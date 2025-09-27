# -*- coding: utf-8 -*-
"""
@author: Liaoyh
@contact: liaoyh21@mails.tsinghua.edu.cn
@file: dist_based_sampling.py
@time: 2025/4/12 1:10
"""
import os
import os.path as osp
import logging
import argparse
from pdb import set_trace as here
import math
from copy import deepcopy
from tqdm import tqdm
from scipy.linalg import eigh

import numpy as np
import pandas as pd

import proc_samples
from exp_acc_scaling_law import assign_unit, train_once


class HomoBasedSampler(object):
    def __init__(self, tr_samples, val_samples, feats, num_min_samples=6 * (6 + 1) / 2,
                 max_grid_size=30, min_grid_size=3, crs='epsg:4326', seed=42, epsilon=1e-6):
        self.tr_samples = tr_samples
        self.val_samples = val_samples
        self.feats = feats
        self.num_min_samples = num_min_samples
        self.max_grid_size = max_grid_size
        self.min_grid_size = min_grid_size
        self.crs = crs
        self.seed = seed
        self.classes = sorted(self.tr_samples.Class.unique().tolist())
        self.num_classes = len(self.classes)
        self.num_feats = len(feats)
        self.epsilon = epsilon

        self.valid_sample_size = self.num_min_samples * 0.5
        self.stats = {}
        self.already_sampled = {}
        self.df_weights = None
        self.total_tr_sample_size = 0

    def init_samples(self):
        # assign units
        tr_samples = assign_unit(self.tr_samples, self.max_grid_size)
        val_samples = assign_unit(self.val_samples, self.max_grid_size, ref=tr_samples)

        # filter invalid units with max class sample size less than self.num_min_samples
        invalid_units = tr_samples.groupby('unit').apply(
            lambda x: x.Class.value_counts().iloc[0] < self.num_min_samples
        )
        invalid_units = invalid_units[invalid_units].index.tolist()
        tr_samples = tr_samples[~tr_samples.unit.isin(invalid_units)]

        # for each unit, sampling training points until its dominant class reaches the num-min-samples
        units = tr_samples.unit.unique()
        for unit in tqdm(units, desc='Init Sampling units'):
            unit_tr_samples = tr_samples[tr_samples.unit == unit]
            unit_val_samples = val_samples[val_samples.unit == unit]
            val_tr_ratio = unit_val_samples.shape[0] / unit_tr_samples.shape[0]

            # init unit sampled data
            unit_sampled_data = {'tr': None, 'val': None}

            # sampling training points
            max_class_sample_size = unit_tr_samples.Class.value_counts().values[0]
            reached = False
            while not reached:
                # sample training points
                sampled_element = unit_tr_samples.sample(n=1, random_state=self.seed)
                unit_tr_samples = unit_tr_samples.drop(index=sampled_element.index)

                if unit_sampled_data['tr'] is None:
                    unit_sampled_data['tr'] = sampled_element
                else:
                    unit_sampled_data['tr'] = pd.concat([unit_sampled_data['tr'], sampled_element], axis=0)

                # update dominated class
                if unit_sampled_data['tr']['ClassName'].value_counts().iloc[0] >= min(self.num_min_samples,
                                                                                      max_class_sample_size):
                    reached = True

            # sampling validation points
            tr_sample_size = unit_sampled_data['tr'].shape[0]
            val_sample_size = min(int(tr_sample_size * val_tr_ratio), unit_val_samples.shape[0])
            unit_sampled_data['val'] = unit_val_samples.sample(n=val_sample_size, random_state=self.seed)

            self.already_sampled[unit] = unit_sampled_data

        # update stats
        self.update_stats(units, self.max_grid_size)


        # update tr sample size
        self.count_total_tr_sample_size()

    def update_stats(self, units, grid_size):
        for unit in units:
            unit_sampled_data = self.already_sampled[unit]
            tr_samples = unit_sampled_data['tr']

            # init a stats
            stats = {
                # node id
                'id': unit,
                # grid size
                'grid_size': grid_size,
                # sample features, category and coords
                'samples': tr_samples,
                # count of sample numbers of each category, shape: of K * 1
                'counts': np.zeros(self.num_classes),
                # weight of sample numbers of each category, shape: of K * 1
                'weights': np.zeros(self.num_classes),
                # mask of each category, shape: of K * 1
                'mask': np.zeros(self.num_classes, dtype=bool),
                # mean of d-features of each category, shape: K * d, < 0 once take log otherwise > 0
                'means': np.zeros((self.num_classes, self.num_feats)),
                # covariance of d-features of each category, shape: K * d * d
                'covs': np.zeros((self.num_classes, self.num_feats, self.num_feats)),
                # variance d-features of each category (diagonal of covs), shape: K * d
                'vars': np.zeros((self.num_classes, self.num_feats)),
                # variance within categories (vwc) of each category, shape: K * d * d
                'vwcs': np.zeros((self.num_classes, self.num_feats, self.num_feats)),
                # variance between categories (vwc)  of each category, shape: K * d * d
                'vbcs': np.zeros((self.num_classes, self.num_feats, self.num_feats)),
                # total variance within categories (vwc), shape: d * d
                't_vwc': np.zeros((self.num_feats, self.num_feats)),
                # total variance between categories (vbc), shape: d * d
                't_vbc': np.zeros((self.num_feats, self.num_feats)),
                # gaussian homogeneous criterion (ghc), shape: K * d > 0
                'ghc': np.zeros(self.num_classes),
                # fisher discriminant criterion (fdc), shape: d - 1
                'fdc': np.zeros(self.num_feats - 1),
                't_ghc': np.nan,
                't_fdc': np.nan,
                't_weights': np.nan,
                'split_times': np.log2(self.max_grid_size / grid_size) if grid_size > 0 else -1,
            }
            overall_mean = tr_samples[self.feats].mean().values
            for idx, cat in enumerate(self.classes):
                cat_samples = tr_samples[tr_samples['Class'] == cat]
                cat_counts = cat_samples.shape[0]

                stats['counts'][idx] = cat_counts

                if cat_counts >= self.valid_sample_size:
                    cat_feats = np.log(cat_samples[self.feats])
                    cat_mean = cat_feats.mean().values
                    cat_cov = cat_feats.cov().values + np.eye(self.num_feats) * self.epsilon
                    variances = cat_cov[np.eye(cat_cov.shape[0], dtype=bool)]

                    # ghc = cat_mean.T @ np.linalg.inv(cat_cov) @ cat_mean / self.num_feats
                    ghc = np.mean(cat_mean ** 2 / variances)

                    delta_mean = cat_mean - overall_mean
                    delta_mean = delta_mean.reshape(-1, 1)
                    dev = cat_feats.values - cat_mean
                    vbc = cat_counts * delta_mean @ delta_mean.T
                    vwc = dev.T @ dev

                    stats['mask'][idx] = True
                    stats['means'][idx] = cat_mean
                    stats['covs'][idx] = cat_cov
                    stats['vars'][idx] = variances
                    stats['vbcs'][idx] = vbc
                    stats['vwcs'][idx] = vwc
                    stats['ghc'][idx] = ghc

            # compute other attributes
            if stats['mask'].any():
                stats['weights'] = stats['counts'] / stats['counts'].sum()
                stats['t_vbc'] = stats['vbcs'].sum(axis=0)
                stats['t_vwc'] = stats['vwcs'].sum(axis=0) + np.eye(self.num_feats) * self.epsilon
                inv_var_w = self._inv_mat(stats['t_vwc'])
                matrix = inv_var_w @ stats['t_vbc']
                eigenvalues, _ = eigh(matrix)
                fdc = sorted(eigenvalues, reverse=True)
                stats['fdc'] = np.array(fdc)

                stats['t_ghc'] = (stats['ghc'][stats['mask']] ** 0.5).mean()
                stats['t_fdc'] = stats['fdc'][0] ** 0.1
                stats['t_weights'] = stats['t_ghc'] * stats['t_fdc']
                if stats['t_fdc'] < 1e-6:
                    stats['t_fdc'] = 1
            self.stats[unit] = stats

    def update_weights(self):
        valid_units = [unit for unit in self.stats.keys() if self.stats[unit]['grid_size'] != -1]
        self.df_weights = {
            'unit': valid_units,
            'ghc': [stats['t_ghc'] for unit, stats in self.stats.items() if unit in valid_units],
            'fdc': [stats['t_fdc'] for unit, stats in self.stats.items() if unit in valid_units],
            'weights': [stats['t_weights'] for unit, stats in self.stats.items() if unit in valid_units],
            'split_times': [stats['split_times'] for unit, stats in self.stats.items() if unit in valid_units]
        }
        self.df_weights = pd.DataFrame.from_dict(self.df_weights, orient='index').T

        # normalize ghc and fdc and re-compute weights
        self.df_weights['re_weighted_ghc'] = self.df_weights['ghc'] / self.df_weights['ghc'].sum()
        self.df_weights['re_weighted_fdc'] = self.df_weights['fdc'] / self.df_weights['fdc'].sum()
        self.df_weights['re_weighted_weights'] = (self.df_weights['ghc'] * self.df_weights['fdc']) * 3 ** (
            self.df_weights['split_times'])
        self.df_weights['re_weighted_weights'] = self.df_weights['re_weighted_weights'] / self.df_weights[
            're_weighted_weights'].sum()
        self.df_weights.sort_values(by='re_weighted_weights', ascending=True, inplace=True)

        return self.df_weights

    def split_grids(self, unit_name):
        # get the unit
        lon, lat = unit_name.split(' ')
        lon, lat = float(lon), float(lat)
        org_grid_size = self.stats[unit_name]['grid_size']
        new_grid_size = org_grid_size / 2
        new_unit_names = [
            f'{lon} {lat}',
            f'{lon} {lat + new_grid_size}',
            f'{lon + new_grid_size} {lat}',
            f'{lon + new_grid_size} {lat + new_grid_size}'
        ]

        # detach org unit
        unit_samples = self.already_sampled.pop(unit_name)
        self.stats.pop(unit_name)
        # print(f'unit {unit_name} is split into {new_unit_names}')

        for new_unit_name in new_unit_names:
            # get lon and lat
            lon, lat = new_unit_name.split(' ')
            lon, lat = float(lon), float(lat)

            # assign unit_samples into new_unit_samples according to spatial range
            new_tr_unit_samples = self.spatial_filter(lon, lat, new_grid_size, unit_samples['tr'])
            new_val_unit_samples = self.spatial_filter(lon, lat, new_grid_size, unit_samples['val'])

            # get corresponding tr_samples and valid samples
            tr_samples = self.spatial_filter(lon, lat, new_grid_size, self.tr_samples)
            val_samples = self.spatial_filter(lon, lat, new_grid_size, self.val_samples)
            tr_samples = tr_samples.drop(index=new_tr_unit_samples.index)
            val_samples = val_samples.drop(index=new_val_unit_samples.index)
            if tr_samples.shape[0] == 0:
                # print(f'unit {new_unit_name} has no training samples')
                continue
            val_tr_ratio = val_samples.shape[0] / tr_samples.shape[0]

            # init unit sampled data
            unit_sampled_data = {'tr': new_tr_unit_samples, 'val': new_val_unit_samples}

            # sampling training points
            max_class_sample_size = tr_samples.Class.value_counts().values[0]
            if max_class_sample_size < self.num_min_samples:
                unit_sampled_data['tr'] = pd.concat([unit_sampled_data['tr'], tr_samples], axis=0)
                unit_sampled_data['val'] = pd.concat([unit_sampled_data['val'], val_samples], axis=0)
                final_grid_size = -1
                # print(f'unit {new_unit_name} has few training samples, stop splitting this unit')
            else:
                reached = False
                while not reached:
                    # sample training points
                    sampled_element = tr_samples.sample(n=1, random_state=self.seed)
                    tr_samples = tr_samples.drop(index=sampled_element.index)

                    unit_sampled_data['tr'] = pd.concat([unit_sampled_data['tr'], sampled_element], axis=0)

                    # update dominated class
                    if unit_sampled_data['tr']['ClassName'].value_counts().iloc[0] >= self.num_min_samples:
                        reached = True

                # sampling validation points
                inc_tr_sample_size = unit_sampled_data['tr'].shape[0] - new_tr_unit_samples.shape[0]
                inc_val_sample_size = min(int(inc_tr_sample_size * val_tr_ratio), val_samples.shape[0])
                unit_sampled_data['val'] = pd.concat(
                    [new_val_unit_samples, val_samples.sample(n=inc_val_sample_size, random_state=self.seed)], axis=0
                )
                final_grid_size = new_grid_size

            # update already sampled
            self.already_sampled[new_unit_name] = unit_sampled_data

            # update stats
            self.update_stats([new_unit_name], final_grid_size)

    def update_samples(self, tr_sample_size):
        self.count_total_tr_sample_size()
        while self.total_tr_sample_size < tr_sample_size:
            # update weights
            self.update_weights()

            # find the unit with the smallest weight
            min_weight_unit = self.df_weights.loc[self.df_weights['re_weighted_weights'].idxmin()]['unit']

            # split it into four quadrants and add samples to each quadrant
            self.split_grids(min_weight_unit)

            # update the total training sample size
            self.count_total_tr_sample_size()

    def spatial_filter(self, lon, lat, grid_size, samples):
        return samples.query(f'lon >= {lon} and lon < {lon + grid_size} and lat >= {lat} and lat < {lat + grid_size}')

    def count_total_tr_sample_size(self):
        self.total_tr_sample_size = sum(list(map(lambda x: x['tr'].shape[0], self.already_sampled.values())))

    def _inv_mat(self, mat):
        try:
            inv_mat = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            inv_mat = np.linalg.pinv(mat)

        return inv_mat


def train_samples(sampler, fraction, model, sampling_method, idx, device):
    tr_samples = pd.concat([elem['tr'] for elem in sampler.already_sampled.values()], axis=0)
    partial_val_samples = pd.concat([elem['val'] for elem in sampler.already_sampled.values()], axis=0)
    val_samples = sampler.val_samples

    ipts = tr_samples, (partial_val_samples, val_samples), fraction, model, sampling_method, idx, device

    return train_once(ipts)


def main():
    # set data and model
    if args.samples == 'fast':
        samples = proc_samples.load_fast(season=args.season, level=args.level)
    elif args.samples == 'coasttrain':
        samples = proc_samples.load_coasttrain(season=args.season, level=args.level)
    else:
        raise ValueError('invalid sample name')
    feats = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    tr_samples = samples[samples.subset.isin(['train'])]
    val_samples = samples[samples.subset.isin(['valid'])]

    out_dir = osp.join('./outs/homo_sampling', f'{args.samples}{str(args.level) if args.level > 1 else ""}')
    os.makedirs(out_dir, exist_ok=True)
    num_min_samples = 6 * (6 + 1) / 2 * 0.5
    fold_base = 1.2

    # set minimum fraction so that even the most rare class have
    max_fold = math.ceil(
        math.log(num_min_samples * 0.5 / tr_samples.Class.value_counts().values[-1]) / math.log(fold_base)
    )
    fractions = [fold_base ** fold for fold in range(max_fold, 1)]

    sampling_method = {'name': 'homo_based_sampling'}
    max_grid_size = 60
    min_grid_size = 3
    sampler = HomoBasedSampler(
        tr_samples, val_samples, feats=feats,
        num_min_samples=num_min_samples, max_grid_size=max_grid_size,
        min_grid_size=min_grid_size
    )

    # initialize sampler
    sampler.init_samples()

    # update fractions
    init_sample_size = sum(list(map(lambda x: x['tr'].shape[0], sampler.already_sampled.values())))
    init_fraction = init_sample_size / tr_samples.shape[0]
    fractions = [fraction for fraction in fractions if fraction > init_fraction]

    # init results
    results = [train_samples(sampler, init_fraction, args.model, sampling_method, 0, args.device)]

    # update sampler
    for idx, fraction in tqdm(enumerate(fractions), total=len(fractions)):
        idx += 1

        tr_sample_size = int(tr_samples.shape[0] * fraction)
        sampler.update_samples(tr_sample_size)

        print(sampler.df_weights)
        print(sampler.df_weights['split_times'].value_counts())

        results.append(train_samples(sampler, fraction, args.model, sampling_method, idx, args.device))

    # convert a list of dict into dataframe
    results = pd.DataFrame(results)
    results.sort_values(by='idx', inplace=True)

    # save results to .csv
    results.to_csv(osp.join(out_dir, f'{args.season}_{args.model}_acc_sl.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-samples', type=str, default='fast', help='season to use')
    parser.add_argument('-season', type=str, default='summer', help='season to use')
    parser.add_argument('-level', type=int, default=1, help='level of classification system')
    parser.add_argument('-model', type=str, default='xgboost', help='model to use')
    parser.add_argument('-device', type=str, default='cuda:0', help='cuda to use')
    args = parser.parse_args()
    # ======================================================================
    main()
    # ======================================================================
