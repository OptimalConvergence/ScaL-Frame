# -*- coding: utf-8 -*-
"""
@author: Liaoyh
@contact: liaoyh21@mails.tsinghua.edu.cn
@file: dist_based_sampling.py
@time: 2025/5/6 16:44
"""
import os.path as osp
import logging
import argparse
from pdb import set_trace as here
import math
from tqdm import tqdm

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from shapely.geometry import box

import proc_samples
from exp_acc_scaling_law import train_once, assign_unit


class DistBasedSampler(object):
    def __init__(self, region, tr_samples, val_samples, feats, base_gmm_cfg, beta_factor, num_min_samples,
                 min_grid_size=5, max_grid_size=30, crs='epsg:4326', seed=42, epsilon=1e-6):
        self.region = region
        self.tr_samples = self.df2gdf(tr_samples)
        self.val_samples = self.df2gdf(val_samples)
        self.feats = feats
        self.base_gmm_cfg = base_gmm_cfg
        self.beta_factor = beta_factor
        self.num_min_samples = num_min_samples
        self.max_grid_size = max_grid_size
        self.min_grid_size = min_grid_size
        self.classes = sorted(self.tr_samples.Class.unique().tolist())
        self.num_classes = len(self.classes)
        self.num_feats = len(feats)

        self.crs = crs
        self.seed = seed
        self.epsilon = epsilon
        self.already_sampled = {'tr': [], 'val': []}
        self.gmm_cfg = base_gmm_cfg
        self.homo_grids = {}
        self.grids = self.init_grids()
        self.cat_ratio = {}

    def randomly_sampling(self, fraction):
        # set data
        tr_sample_size = int(self.tr_samples.shape[0] * fraction)
        val_sample_size = int(self.val_samples.shape[0] * fraction)

        tr_samples = self.tr_samples.sample(tr_sample_size, random_state=self.seed)
        partial_val_samples = self.val_samples.sample(val_sample_size, random_state=self.seed)

        self.already_sampled['tr'] = tr_samples
        self.already_sampled['val'] = partial_val_samples

        # update cat ratio
        self.update_cat_ratio()

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

            self.already_sampled['tr'].append(unit_sampled_data['tr'])
            self.already_sampled['val'].append(unit_sampled_data['val'])

        self.already_sampled['tr'] = pd.concat(self.already_sampled['tr'], axis=0)
        self.already_sampled['val'] = pd.concat(self.already_sampled['val'], axis=0)

        # update cat ratio
        self.update_cat_ratio()

    def update_cat_ratio(self):
        tr_data = self.already_sampled['tr']
        for cat in self.classes:
            cat_data = tr_data.query(f'Class == {cat}')
            cat_ratio = cat_data.shape[0] / tr_data.shape[0]
            self.cat_ratio[cat] = cat_ratio

    def init_grids(self):
        # Get the bounds of the region
        minx, miny, maxx, maxy = self.region.total_bounds
        minx = self.min_grid_size * (minx // self.min_grid_size)
        miny = self.min_grid_size * (miny // self.min_grid_size)
        maxx = self.min_grid_size * (maxx // self.min_grid_size + 1)
        maxy = self.min_grid_size * (maxy // self.min_grid_size + 1)

        # build grids
        grids = []
        for x in np.arange(minx, maxx + self.min_grid_size, self.min_grid_size):
            for y in np.arange(miny, maxy + self.min_grid_size, self.min_grid_size):
                min_x = max(x, -180)
                max_x = min(x + self.min_grid_size, 180)
                min_y = max(y, -90)
                max_y = min(y + self.min_grid_size, 90)
                grid = box(min_x, min_y, max_x, max_y)
                grid_id = f'{x} {y}'
                grids.append({'grid_id': grid_id, 'geometry': grid,
                              'center_x': (min_x + max_x) / 2, 'center_y': (min_y + max_y) / 2})
        grids = gpd.GeoDataFrame(grids, crs=self.crs)

        # preprocess grids
        grids = gpd.sjoin(grids, self.region, how='inner', predicate='intersects')
        grids = grids[['grid_id', 'center_x', 'center_y', 'geometry']]
        return grids

    def update_samples(self, tr_sample_size):
        # unmixing
        self.unmixing()

        # sampling according to gmm weights
        for idx, cat in enumerate(self.classes):
            inc_sampled_size = int(tr_sample_size * self.cat_ratio[cat]) - \
                               self.already_sampled['tr'].query(f'Class == {cat}').shape[0]
            val_tr_ratio = self.already_sampled['val'].query(f'Class == {cat}').shape[0] / \
                           self.already_sampled['tr'].query(f'Class == {cat}').shape[0]
            # assign the number of samples of each gmm types
            weights = self.homo_grids[cat]['weight']
            weights = (weights ** self.beta_factor) / (weights ** self.beta_factor).sum()
            weighted_tr_inc_sample_size = (weights * inc_sampled_size).astype(int)
            weighted_val_inc_sample_size = (val_tr_ratio * weighted_tr_inc_sample_size).astype(int)

            for gmm_type in weights.index:
                # extract the grids
                cat_homo_grids = self.homo_grids[cat]['grids'].query(f'gmm_type == {gmm_type}')

                # get the tr samples
                tr_samples = self.tr_samples.drop(index=self.already_sampled['tr'].index)
                tr_samples = gpd.sjoin(tr_samples, cat_homo_grids, how='inner', predicate='intersects')
                cat_tr_samples = tr_samples.query(f'Class == {cat}')
                inc_tr_samples = cat_tr_samples.sample(
                    n=min(weighted_tr_inc_sample_size.at[gmm_type], len(cat_tr_samples))
                )

                # get the val samples
                val_samples = self.val_samples.drop(index=self.already_sampled['val'].index)
                val_samples = gpd.sjoin(val_samples, cat_homo_grids, how='inner', predicate='intersects')
                cat_val_samples = val_samples.query(f'Class == {cat}')
                inc_val_samples = cat_val_samples.sample(
                    n=min(weighted_val_inc_sample_size.at[gmm_type], len(cat_val_samples))
                )

                # append it to
                self.already_sampled['tr'] = pd.concat([self.already_sampled['tr'], inc_tr_samples], axis=0)
                self.already_sampled['val'] = pd.concat([self.already_sampled['val'], inc_val_samples], axis=0)

        # update cat ratio
        self.update_cat_ratio()

    def df2gdf(self, data):
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['lon'], data['lat']))
        data.set_crs(epsg=4326, inplace=True)
        return data

    def unmixing(self):
        print(self.already_sampled['tr']['Class'].value_counts() // (self.num_min_samples * 5))
        for idx, category in tqdm(enumerate(self.classes), total=self.num_classes):
            # extract data
            cat_data = self.already_sampled['tr'].query(f'Class == {category}')

            # determine the number of components
            # num_components = int(cat_data.shape[0] // (self.num_min_samples * 5))
            # num_components = min(num_components, 20)
            num_components = 4

            if num_components > 1:
                # perform bvgmm
                # bv_gmm = BayesianGaussianMixture(
                #     n_components=num_components,
                #     init_params=self.gmm_cfg['init_params'],
                #     weight_concentration_prior_type=self.gmm_cfg['weight_concentration_prior_type'],
                #     covariance_type='full',
                #     weight_concentration_prior=self.gmm_cfg['weight_concentration_prior'],
                #     max_iter=self.gmm_cfg['bv_gmm_max_iter'],
                #     random_state=self.seed
                # )
                bv_gmm = GaussianMixture(
                    n_components=num_components,
                    init_params=self.gmm_cfg['init_params'],
                    covariance_type='full',
                    max_iter=self.gmm_cfg['max_iter'],
                    warm_start=self.gmm_cfg['warm_start'],
                    random_state=self.seed
                )
                features = cat_data[['lon', 'lat'] + self.feats]
                features = features.copy()
                features.loc[:, self.feats] = np.log(features[self.feats] + 0.01)
                bv_gmm.fit(features)
                labels = bv_gmm.predict(features)

                # assign grids according to the labels
                cat_data.loc[:, ['gmm_type']] = labels
            else:
                cat_data.loc[:, ['gmm_type']] = 0

            self.homo_grids[category] = self.allocate_grids(cat_data.loc[:, ['geometry', 'gmm_type']])

    def allocate_grids(self, data):
        grids = self.grids
        grids.loc[:, ['gmm_type']] = -1

        # compute the original weight
        org_weight = data['gmm_type'].value_counts()

        # if intersected
        intersected = gpd.sjoin(grids, data, how='inner', predicate='intersects')
        intersected.rename(columns={'gmm_type_right': 'gmm_type'}, inplace=True)
        grid_id2gmm_type = intersected.groupby('grid_id').apply(lambda x: x.gmm_type.value_counts().index[0],
                                                                include_groups=False)
        grids['gmm_type'] = grids['grid_id'].apply(lambda x: grid_id2gmm_type[x] if x in grid_id2gmm_type.index else -1)

        # Handle non-intersected grids
        non_intersected = grids[grids['gmm_type'] == -1]
        intersected = grids[grids['gmm_type'] != -1]
        if not non_intersected.empty:
            for idx, row in non_intersected.iterrows():
                lon, lat = row.grid_id.split(' ')
                lon, lat = float(lon), float(lat)

                # Find the nearest grid
                distances = intersected.apply(lambda x: (x.center_x - lon) ** 2 + (x.center_y - lat), axis=1)
                nearest = distances.idxmin()

                grids.at[idx, 'gmm_type'] = grids.at[nearest, 'gmm_type']

        return {'weight': org_weight, 'grids': grids}


def train_samples(sampler, fraction, model, sampling_method, idx, device):
    tr_samples = sampler.already_sampled['tr']
    partial_val_samples = sampler.already_sampled['val']
    val_samples = sampler.val_samples

    ipts = tr_samples, (partial_val_samples, val_samples), fraction, model, sampling_method, idx, device

    return train_once(ipts)


def main():
    # set data and model
    if args.samples == 'fast':
        samples = proc_samples.load_fast(season=args.season, level=args.level)
    else:
        raise ValueError('invalid sample name')

    feats = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    tr_samples = samples[samples.subset.isin(['train'])]
    val_samples = samples[samples.subset.isin(['valid'])]

    region_fp = osp.join('../../assets/GeoData/Continents/World_Continents.shp')
    region = gpd.read_file(region_fp)
    region.to_crs(epsg=4326, inplace=True)
    region = region.dissolve()

    out_dir = osp.join('./outs/dist_sampling', f'{args.samples}{str(args.level) if args.level > 1 else ""}')
    num_min_samples = 6 * (6 + 1) / 2 * 0.5
    fold_base = 1.2
    max_fold = math.ceil(
        math.log(num_min_samples * 0.5 / tr_samples.Class.value_counts().values[-1]) / math.log(fold_base)
    )
    fractions = [fold_base ** fold for fold in range(max_fold, 1)]

    # inst a sampler
    sampling_method = {'name': 'dist_based_sampling'}
    base_gmm_cfg = {'max_iter': 10000, 'init_params': 'k-means++', 'warm_start': True}
    max_grid_size = 20
    min_grid_size = 5
    beta_factor = 0.8
    sampler = DistBasedSampler(
        region, tr_samples, val_samples, feats=feats, base_gmm_cfg=base_gmm_cfg, beta_factor=beta_factor,
        num_min_samples=num_min_samples, min_grid_size=min_grid_size, max_grid_size=max_grid_size,
    )

    # initialize sampler
    # sampler.init_samples()

    # update fractions
    # init_fraction = sampler.already_sampled['tr'].shape[0] / tr_samples.shape[0]
    # fractions = [fraction for fraction in fractions if fraction > init_fraction]

    # init results
    results = []
    # results = [train_samples(sampler, init_fraction, args.model, sampling_method, 0, args.device)]

    # update sampler
    for idx, fraction in tqdm(enumerate(fractions), total=len(fractions)):
        if fraction < 0.015:
            sampler.randomly_sampling(fraction)
        else:
            tr_sample_size = int(tr_samples.shape[0] * fraction)
            sampler.update_samples(tr_sample_size)

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
    logging.info("start running")
    # ======================================================================
    main()
    # ======================================================================
    logging.info("everything is under control")
