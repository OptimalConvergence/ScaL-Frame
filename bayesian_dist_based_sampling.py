# -*- coding: utf-8 -*-
"""
@author: Liaoyh
@contact: liaoyh21@mails.tsinghua.edu.cn
@file: bayesian_dist_based_sampling.py
@time: 2025/5/7 15:56
"""
import os
import os.path as osp
import logging
import argparse
from pdb import set_trace as here
from tqdm import tqdm
import math
from copy import deepcopy
import geopandas as gpd

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from shapely.geometry import box

import proc_samples
from dist_based_sampling import train_samples
from utils import process_in_batch
from visualization import fit_ll

import warnings

# Suppress specific DeprecationWarning
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="DataFrameGroupBy.apply operated on the grouping columns.*"
)


class BayesianDistBasedSampler(object):
    def __init__(self, region, tr_samples, val_samples, num_min_samples, min_grid_size,
                 feats, label, gmm_cfg, beta_factor, out_dir,
                 crs='epsg:4326', seed=42):
        self.region = region
        self.tr_samples = tr_samples
        self.val_samples = val_samples
        self.num_min_samples = num_min_samples
        self.min_grid_size = min_grid_size
        self.feats = feats
        self.label = label
        self.gmm_cfg = gmm_cfg
        self.beta_factor = beta_factor
        self.crs = crs
        self.seed = seed
        self.out_dir = out_dir
        self.classes = sorted(self.tr_samples.Class.unique().tolist())
        self.class_names = [self.query_category(self.tr_samples, self.label, cat).iloc[0]['ClassName'] for cat in
                            self.classes]
        self.num_classes = len(self.classes)
        self.num_feats = len(feats)

        self.homo_grids = {}
        self.grids = self.init_grids()
        self.already_sampled = {'tr': [], 'val': []}
        self.cat_ratio = self.tr_samples[label].value_counts(normalize=True).to_dict()

        self.cur_tr_sample_size = 0
        self.cur_val_sample_size = 0

    def randomly_sampling(self, fraction):
        # set data
        tr_sample_size = int(self.tr_samples.shape[0] * fraction)
        val_sample_size = int(self.val_samples.shape[0] * fraction)

        tr_samples = self.tr_samples.sample(tr_sample_size, random_state=self.seed)
        partial_val_samples = self.val_samples.sample(val_sample_size, random_state=self.seed)

        self.already_sampled['tr'] = tr_samples
        self.already_sampled['val'] = partial_val_samples

    def query_category(self, data, label, cat):
        if isinstance(self.classes[0], int):
            return data.query(f'{label} == {cat}')
        elif isinstance(self.classes[0], str):
            return data.query(f'{label} == "{cat}"')
        else:
            raise ValueError('invalid class type')

    def create_gmm(self, label_col, use_eval=False, overwrite=False):
        tr_gmm_results_fp = osp.join(self.out_dir, f'tr_gmm_results_{label_col}.csv')
        val_gmm_results_fp = osp.join(self.out_dir, f'val_gmm_results_{label_col}.csv')
        if osp.exists(tr_gmm_results_fp) and not overwrite:
            return

        self.tr_samples.loc[:, ['gmm_type']] = -1
        self.val_samples.loc[:, ['gmm_type']] = -1
        for idx, category in tqdm(enumerate(self.classes), total=self.num_classes):
            # extract data
            tr_cat_data = self.query_category(self.tr_samples, label_col, category)
            val_cat_data = self.query_category(self.val_samples, label_col, category)

            if use_eval:
                n_components = val_cat_data.shape[0] / (self.num_min_samples * 4)
            else:
                n_components = tr_cat_data.shape[0] / (self.num_min_samples * 10)
            n_components = np.clip(n_components, 8, self.gmm_cfg['n_components']).astype(int)
            print(self.class_names[self.classes.index(category)], n_components)

            if self.gmm_cfg['bayesian']:
                # Bayesian GMM
                bv_gmm = BayesianGaussianMixture(
                    n_components=n_components,
                    weight_concentration_prior_type=self.gmm_cfg['weight_concentration_prior_type'],
                    weight_concentration_prior=self.gmm_cfg['weight_concentration_prior'],
                    init_params=self.gmm_cfg['init_params'],
                    covariance_type='full',
                    max_iter=self.gmm_cfg['max_iter'],
                    warm_start=self.gmm_cfg['warm_start'],
                    random_state=self.seed
                )
            else:
                bv_gmm = GaussianMixture(
                    n_components=n_components,
                    init_params=self.gmm_cfg['init_params'],
                    covariance_type='full',
                    max_iter=self.gmm_cfg['max_iter'],
                    warm_start=self.gmm_cfg['warm_start'],
                    random_state=self.seed
                )

            tr_features = self.extract_feats(tr_cat_data)
            val_features = self.extract_feats(val_cat_data)
            if use_eval:
                bv_gmm.fit(val_features)
            else:
                bv_gmm.fit(tr_features)
            tr_gmm_type = bv_gmm.predict(tr_features)
            val_gmm_type = bv_gmm.predict(val_features)

            # assign gmm labels to tr and valid samples
            self.tr_samples.loc[tr_cat_data.index, 'gmm_type'] = tr_gmm_type
            self.val_samples.loc[val_cat_data.index, 'gmm_type'] = val_gmm_type

        # save gmm results
        tr_gmm_results = self.tr_samples[['gmm_type']]
        val_gmm_results = self.val_samples[['gmm_type']]
        tr_gmm_results.to_csv(tr_gmm_results_fp)
        val_gmm_results.to_csv(val_gmm_results_fp)

    def init_grids(self):
        # Get the bounds of the region
        minx, miny, maxx, maxy = self.region.total_bounds
        minx = np.floor(self.min_grid_size * (minx // self.min_grid_size))
        miny = np.floor(self.min_grid_size * (miny // self.min_grid_size))
        maxx = np.ceil(self.min_grid_size * (maxx // self.min_grid_size + 1))
        maxy = np.ceil(self.min_grid_size * (maxy // self.min_grid_size + 1))

        # build grids
        ipts = [(x, y)
                for x in np.arange(minx, maxx + self.min_grid_size, self.min_grid_size)
                for y in np.arange(miny, maxy + self.min_grid_size, self.min_grid_size)]
        grids = process_in_batch(self.create_grid, ipts, para=False, num_workers=20)
        grids = gpd.GeoDataFrame(grids, crs=self.crs)

        # preprocess grids
        grids = gpd.sjoin(grids, self.region, how='inner', predicate='intersects')
        grids = grids[['grid_id', 'center_x', 'center_y', 'geometry']]

        # filter the grids that neither contains tr samples nor val samples
        all_samples = gpd.GeoDataFrame(pd.concat([self.tr_samples, self.val_samples])['geometry'])
        grids = gpd.sjoin(grids, all_samples, how='inner', predicate='intersects')
        grids.drop_duplicates(['grid_id'], inplace=True)
        grids.drop(columns=['index_right'], inplace=True)

        return grids

    def create_grid(self, coords):
        x, y = coords
        min_x = max(x, -180)
        max_x = min(x + self.min_grid_size, 180)
        min_y = max(y, -90)
        max_y = min(y + self.min_grid_size, 90)
        grid = box(min_x, min_y, max_x, max_y)
        grid_id = f'{x} {y}'

        return {'grid_id': grid_id, 'geometry': grid,
                'center_x': (min_x + max_x) / 2, 'center_y': (min_y + max_y) / 2}

    def extract_feats(self, data):
        if 'bm_lon' in data.columns and 'bm_lat' in data.columns:
            features = data[['bm_lon', 'bm_lat'] + self.feats]
        else:
            features = data[['lon', 'lat'] + self.feats]
        features = features.copy()
        features.loc[:, self.feats] = np.log(features[self.feats] + 0.01)
        return features

    def df2gdf(self, data):
        data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['lon'], data['lat']))
        data.set_crs(epsg=4326, inplace=True)
        return data

    def dist_based_sampling_pointly(self, fraction, label_col, use_eval=False):
        tr_gmm_results_fp = osp.join(self.out_dir, f'tr_gmm_results_{label_col}.csv')
        val_gmm_results_fp = osp.join(self.out_dir, f'val_gmm_results_{label_col}.csv')
        tr_gmm_results = pd.read_csv(tr_gmm_results_fp)
        val_gmm_results = pd.read_csv(val_gmm_results_fp)

        self.tr_samples.loc[:, ['gmm_type']] = tr_gmm_results['gmm_type'].values
        self.val_samples.loc[:, ['gmm_type']] = val_gmm_results['gmm_type'].values

        tr_df_stratified = []
        val_df_stratified = []
        for cat in self.classes:
            # extract data
            tr_cat_data = self.query_category(self.tr_samples, label_col, cat)
            val_cat_data = self.query_category(self.val_samples, label_col, cat)

            tr_frac_assigned = (tr_cat_data['gmm_type'].value_counts() ** self.beta_factor) / (
                    tr_cat_data['gmm_type'].value_counts() ** self.beta_factor).sum()
            tr_frac_assigned = (tr_cat_data.shape[0] * fraction) * tr_frac_assigned
            val_frac_assigned = (val_cat_data['gmm_type'].value_counts() ** self.beta_factor) / (
                    val_cat_data['gmm_type'].value_counts() ** self.beta_factor).sum()
            val_frac_assigned = (val_cat_data.shape[0] * fraction) * val_frac_assigned

            # extract data
            tr_cat_data = self.query_category(tr_cat_data, self.label, cat)
            val_cat_data = self.query_category(val_cat_data, self.label, cat)

            if use_eval:
                gmm_types = val_cat_data['gmm_type'].unique()
            else:
                gmm_types = tr_cat_data['gmm_type'].unique()

            for gmm_type in gmm_types:
                tr_gmm_type_data = tr_cat_data.query(f'gmm_type == {gmm_type}')
                val_gmm_type_data = val_cat_data.query(f'gmm_type == {gmm_type}')
                if use_eval:
                    val_df_stratified.append(val_gmm_type_data.sample(
                        n=round(val_frac_assigned.loc[gmm_type]), random_state=self.seed,
                        replace=True if val_frac_assigned.loc[gmm_type] > val_gmm_type_data.shape[0] else False
                    ))
                    if gmm_type in tr_frac_assigned.index and tr_gmm_type_data.shape[0] > 0:
                        tr_df_stratified.append(tr_gmm_type_data.sample(
                            n=round(tr_frac_assigned.loc[gmm_type]), random_state=self.seed,
                            replace=True if tr_frac_assigned.loc[gmm_type] > tr_gmm_type_data.shape[0] else False
                        ))
                else:
                    tr_df_stratified.append(tr_gmm_type_data.sample(
                        n=round(tr_frac_assigned.loc[gmm_type]), random_state=self.seed,
                        replace=True if tr_frac_assigned.loc[gmm_type] > tr_gmm_type_data.shape[0] else False
                    ))
                    if gmm_type in val_frac_assigned.index and val_gmm_type_data.shape[0] > 0:
                        val_df_stratified.append(val_gmm_type_data.sample(
                            n=round(val_frac_assigned.loc[gmm_type]), random_state=self.seed,
                            replace=True if val_frac_assigned.loc[gmm_type] > val_gmm_type_data.shape[0] else False
                        ))

        self.already_sampled['tr'] = pd.concat(tr_df_stratified, axis=0)
        self.already_sampled['val'] = pd.concat(val_df_stratified, axis=0)

    def update_weights(self, beta_factor, label_col, use_eval):
        if beta_factor is None:
            beta_factor = self.beta_factor
        print(beta_factor)
        for cat in self.classes:
            if use_eval:
                alloc_gmm_dist = self.homo_grids[label_col][cat]['alloc_val_weight']
            else:
                alloc_gmm_dist = self.homo_grids[label_col][cat]['alloc_tr_weight']

            self.homo_grids[label_col][cat]['re_weight'] = alloc_gmm_dist ** beta_factor / (
                    alloc_gmm_dist ** beta_factor).sum()

    def init_homo_grids(self, use_eval, label_col, beta_factor=None, random_assign=False):
        tr_gmm_results_fp = osp.join(self.out_dir, f'tr_gmm_results_{label_col}.csv')
        val_gmm_results_fp = osp.join(self.out_dir, f'val_gmm_results_{label_col}.csv')
        tr_gmm_results = pd.read_csv(tr_gmm_results_fp)
        val_gmm_results = pd.read_csv(val_gmm_results_fp)

        self.tr_samples.loc[:, ['gmm_type']] = tr_gmm_results['gmm_type'].values
        self.val_samples.loc[:, ['gmm_type']] = val_gmm_results['gmm_type'].values
        if label_col not in self.homo_grids:
            self.homo_grids.update({label_col: {}})

        for cat in self.classes:
            if use_eval:
                self.homo_grids[label_col][cat] = self.allocate_grids(
                    self.query_category(self.val_samples, self.label, cat).loc[:, ['geometry', 'gmm_type']]
                )
            else:
                self.homo_grids[label_col][cat] = self.allocate_grids(
                    self.query_category(self.tr_samples, label_col, cat).loc[:, ['geometry', 'gmm_type']]
                )

            if random_assign:
                np.random.seed(self.seed)
                self.homo_grids[label_col][cat]['grids'].loc[:, ['gmm_type']] = np.random.randint(
                    low=0, high=self.tr_samples.gmm_type.max(), size=self.homo_grids[label_col][cat]['grids'].shape[0],
                )

            # self.homo_grids[label_col][cat]['grids'].to_file(f'./grids_{self.class_names[self.classes.index(cat)]}.shp')

            alloc_tr_gmm_dist = gpd.sjoin(
                self.query_category(self.tr_samples, label_col, cat),
                self.homo_grids[label_col][cat]['grids'], how='inner', predicate='within'
            ).gmm_type_right.value_counts(normalize=True)

            alloc_val_gmm_dist = gpd.sjoin(
                self.query_category(self.val_samples, label_col, cat),
                self.homo_grids[label_col][cat]['grids'], how='inner', predicate='within'
            ).gmm_type_right.value_counts(normalize=True)

            if (beta_factor == self.beta_factor) and not random_assign:
                print(
                    f'{self.class_names[self.classes.index(cat)]} gmm_types distance: {np.abs(alloc_val_gmm_dist - alloc_tr_gmm_dist).sum()}'
                )

            self.homo_grids[label_col][cat]['alloc_tr_weight'] = alloc_tr_gmm_dist
            self.homo_grids[label_col][cat]['alloc_val_weight'] = alloc_val_gmm_dist

    def dist_based_sampling_regionly(self, fraction, label_col, use_eval=False):
        tr_sample_size = self.tr_samples.shape[0] * fraction
        val_sample_size = self.val_samples.shape[0] * fraction

        tr_df_stratified = []
        val_df_stratified = []

        inc_tr_sample_size = tr_sample_size - self.cur_tr_sample_size
        inc_val_sample_size = val_sample_size - self.cur_val_sample_size

        # sampling according to gmm weights
        for idx, cat in enumerate(self.classes):
            # self.homo_grids[label_col][cat]['grids'].to_file('farm_grids.shp')
            # self.query_category(self.val_samples, self.label, cat)

            weights = self.homo_grids[label_col][cat]['re_weight']
            weighted_tr_sample_size = round(weights * inc_tr_sample_size * self.cat_ratio[cat]).astype(int)
            weighted_val_sample_size = round(weights * inc_val_sample_size * self.cat_ratio[cat]).astype(int)
            for gmm_type in weights.index:
                # extract the grids
                cat_homo_grids = self.homo_grids[label_col][cat]['grids'].query(f'gmm_type == {gmm_type}')

                # get the tr samples
                cat_tr_samples = gpd.sjoin(self.tr_samples, cat_homo_grids, how='inner', predicate='within')
                cat_tr_samples = self.query_category(cat_tr_samples, self.label, cat)
                cat_tr_samples = cat_tr_samples.sample(frac=1, random_state=self.seed)
                if len(self.already_sampled['tr']) > 0:
                    existing_samples = self.already_sampled['tr']
                    existing_samples = existing_samples[existing_samples.index.isin(cat_tr_samples.index)]
                    cat_tr_samples = cat_tr_samples.drop(index=existing_samples.index)
                cat_tr_samples = cat_tr_samples.iloc[:min(weighted_tr_sample_size.at[gmm_type], len(cat_tr_samples))]
                # cat_tr_samples = cat_tr_samples.sample(
                #     n=min(weighted_tr_sample_size.at[gmm_type], len(cat_tr_samples)), random_state=self.seed
                # )

                # get the val samples
                cat_val_samples = gpd.sjoin(self.val_samples, cat_homo_grids, how='inner', predicate='within')
                cat_val_samples = self.query_category(cat_val_samples, self.label, cat)
                cat_tr_samples = cat_tr_samples.sample(frac=1, random_state=self.seed)
                if len(self.already_sampled['val']) > 0:
                    existing_samples = self.already_sampled['val']
                    existing_samples = existing_samples[existing_samples.index.isin(cat_val_samples.index)]
                    cat_val_samples = cat_val_samples.drop(index=existing_samples.index)
                cat_val_samples = cat_val_samples.iloc[
                                  :min(weighted_val_sample_size.at[gmm_type], len(cat_val_samples))]
                # cat_val_samples = cat_val_samples.sample(
                #     n=min(weighted_val_sample_size.at[gmm_type], len(cat_val_samples)), random_state=self.seed
                # )

                tr_df_stratified.append(cat_tr_samples)
                val_df_stratified.append(cat_val_samples)

        # append it to
        if len(self.already_sampled['tr']) > 0:
            self.already_sampled['tr'] = pd.concat([self.already_sampled['tr'], pd.concat(tr_df_stratified, axis=0)])
            self.already_sampled['val'] = pd.concat([self.already_sampled['val'], pd.concat(val_df_stratified, axis=0)])
        else:
            self.already_sampled['tr'] = pd.concat(tr_df_stratified, axis=0)
            self.already_sampled['val'] = pd.concat(val_df_stratified, axis=0)

        assert self.already_sampled['tr'].index.is_unique
        assert self.already_sampled['val'].index.is_unique

        self.cur_tr_sample_size = self.already_sampled['tr'].shape[0]
        self.cur_val_sample_size = self.already_sampled['val'].shape[0]

    def allocate_grids(self, data):
        grids = deepcopy(self.grids)
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

        # Compute the pair wise distance
        if not non_intersected.empty:
            coords_intersected = intersected[['center_x', 'center_y']]
            coords_non_intersected = non_intersected[['center_x', 'center_y']]
            # Compute the distance matrix
            distances = np.linalg.norm(coords_intersected.values[:, np.newaxis] - coords_non_intersected.values, axis=2)
            # Find the nearest grid for each non-intersected grid
            nearest_indices = np.argmin(distances, axis=0)
            # Assign the gmm_type of the nearest grid to the non-intersected grid
            non_intersected.loc[:, ['gmm_type']] = intersected.iloc[nearest_indices]['gmm_type'].values

        grids.loc[non_intersected.index, ['gmm_type']] = non_intersected['gmm_type']

        return {'org_weight': org_weight, 'grids': grids}


def determine_n_components(samples, season):
    try:
        season = int(season)
    except:
        pass

    if samples in ['fast']:
        base_components = 20
    elif samples in ['coasttrain']:
        base_components = 20
    elif samples in ['lucas']:
        base_components = 20
    else:
        raise ValueError('invalid sample name')

    if season in ['spring', 'summer', 'fall', 'winter']:
        n_components = base_components * 2
    elif season in range(1, 13):
        n_components = base_components
    elif season == 'all_year':
        n_components = base_components * 4
    else:
        raise ValueError('invalid season name')

    return int(n_components)


def determine_feats(samples):
    if samples in ['fast', 'coasttrain']:
        feats = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    elif samples in ['lucas']:
        feats = ['Blue', 'Green', 'Red', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2']
    else:
        raise ValueError('invalid sample name')

    return feats


def determine_beta_factor(samples):
    if samples in ['fast']:
        beta_factor = 1.5
    elif samples in ['coasttrain']:
        beta_factor = 1.1
    elif samples in ['lucas']:
        beta_factor = 1.3
    else:
        raise ValueError('invalid sample name')

    return beta_factor


def determine_grid_size(samples):
    if samples in ['fast']:
        grid_size = 3
    elif samples in ['coasttrain']:
        grid_size = 5
    elif samples in ['lucas']:
        grid_size = 0.1
    else:
        raise ValueError('invalid sample name')

    return grid_size


def load_region(samples):
    region_fp = osp.join('../../assets/GeoData/Continents/World_Continents.shp')
    region = gpd.read_file(region_fp)
    region.to_crs(epsg=4326, inplace=True)
    if samples in ['fast', 'coasttrain']:
        region = region.dissolve()
    elif samples in ['lucas']:
        region = region.query('CONTINENT == "Europe"')
    else:
        raise ValueError('invalid sample name')

    return region


class LogisticLawAdaptor(object):
    def __init__(self, beta_factor, delta=0.05, n_obs=6):
        self.beta_factor = beta_factor
        self.sample_sizes = []
        self.accuracies = []
        self.delta = delta
        self.n_obs = n_obs

        self.cur_rmse = 1
        self.cur_fit = 0

    def update_records(self, sample_size, accuracy):
        self.sample_sizes.append(sample_size)
        self.accuracies.append(accuracy)

    def fit_logistic_law(self):
        if len(self.sample_sizes) < self.n_obs:
            return

        log10_sample_size = np.log10(self.sample_sizes)
        accuracies = np.array(self.accuracies)
        fit_result, rmse = fit_ll(log10_sample_size, accuracies)

        self.cur_rmse = rmse
        self.cur_fit = fit_result

        print(f'Logistic law fit RMSE: {rmse}')
        print(f'{self.cur_fit[-1] - self.accuracies[-1]}')

    def adjust_beta_factor(self):
        if len(self.sample_sizes) < self.n_obs:
            return

        # if the last prediction exceeds real accuracy, increase beta_factor by delta else, decrease by delta
        if self.accuracies[-1] > self.cur_fit[-1]:
            self.beta_factor -= self.delta
        else:
            self.beta_factor += self.delta
        print(f'Adjust beta_factor: {self.beta_factor}')


def main():
    # set data and model
    if args.samples == 'fast':
        samples = proc_samples.load_fast(season=args.season, level=args.level, predict=args.predict)
    elif args.samples == 'coasttrain':
        samples = proc_samples.load_coasttrain(season=args.season, level=args.level)
    elif args.samples == 'lucas':
        # samples = proc_samples.load_lucas(season=args.season, level=args.level)
        samples = proc_samples.load_lucas18sum(season=args.season, level=args.level)
    else:
        raise ValueError('invalid sample name')
    region = load_region(args.samples)

    feats = determine_feats(args.samples)
    label = 'Class'
    tr_samples = samples[samples.subset.isin(['train'])]
    val_samples = samples[samples.subset.isin(['valid'])]
    out_dir = osp.join('./outs/bayesian_dist_sampling',
                       f'{args.samples}{str(args.level) if args.level > 1 else ""}-{args.season}')
    os.makedirs(out_dir, exist_ok=True)

    num_min_samples = 6 * (6 + 1) / 2 * 0.5
    fold_base = 1.2
    max_fold = math.ceil(
        math.log(num_min_samples * 0.5 / tr_samples.Class.value_counts().values[-1]) / math.log(fold_base)
    )
    fractions = [fold_base ** fold for fold in range(max_fold, 1)]

    # inst a sampler
    sampling_method = {'name': 'bayesian_dist_based_sampling'}
    n_components = determine_n_components(args.samples, args.season)
    gmm_cfg = {
        'bayesian': True, 'weight_concentration_prior_type': 'dirichlet_process', 'weight_concentration_prior': 1e5,
        'n_components': n_components, 'max_iter': 10000, 'init_params': 'kmeans', 'warm_start': True
    }
    beta_factor = determine_beta_factor(args.samples)
    min_grid_size = determine_grid_size(args.samples)
    sampler = BayesianDistBasedSampler(
        region, tr_samples, val_samples, num_min_samples=num_min_samples, min_grid_size=min_grid_size,
        feats=feats, label=label, gmm_cfg=gmm_cfg, beta_factor=beta_factor, out_dir=out_dir
    )

    label_col = label if args.predict is None else args.predict
    print('label_col:', label_col)
    if args.use_eval:
        sampler.cat_ratio = sampler.val_samples[label_col].value_counts(normalize=True).to_dict()
    sampler.create_gmm(label_col=args.predict, overwrite=args.overwrite, use_eval=args.use_eval)
    sampler.init_homo_grids(use_eval=args.use_eval, label_col=label_col, beta_factor=None)
    sampler.update_weights(beta_factor=None, use_eval=args.use_eval, label_col=label_col)
    sampler1 = deepcopy(sampler)
    sampler2 = deepcopy(sampler)
    sampler3 = deepcopy(sampler)
    sampler4 = deepcopy(sampler)
    sampler2.init_homo_grids(use_eval=args.use_eval, label_col=label_col, beta_factor=None, random_assign=True)
    sampler4.init_homo_grids(use_eval=args.use_eval, label_col=label_col, beta_factor=None, random_assign=True)

    # evaluate training samples
    results = []
    idx = -1
    adaptor = LogisticLawAdaptor(beta_factor + 1e-6)
    for fraction in tqdm(fractions, total=len(fractions)):

        # if fraction < 0.003:
        #     continue

        print('R.P.S.')
        idx += 1
        sampler.randomly_sampling(fraction)
        outs = train_samples(sampler, fraction, args.model, sampling_method, idx, args.device)
        results.append(outs)

        print('D.B.S. 1')
        idx += 1
        if args.modality == 'poi':
            sampler1.dist_based_sampling_pointly(fraction, label_col=label_col, use_eval=args.use_eval)
        else:
            temp_beta_factor = beta_factor + 1e-6
            # if args.samples == 'coasttrain':
            #     if fraction > 0.0073:
            #         temp_beta_factor = 1.1
            if sampler1.tr_samples.shape[0] * fraction > 25000:
                temp_beta_factor = 1
            sampler1.update_weights(beta_factor=temp_beta_factor, use_eval=args.use_eval, label_col=label_col)
            sampler1.dist_based_sampling_regionly(fraction, label_col=label_col, use_eval=args.use_eval)
        outs = train_samples(sampler1, fraction, args.model, sampling_method, idx, args.device)
        results.append(outs)

        # print('D.B.S. 2')
        # idx += 1
        # if args.modality == 'poi':
        #     sampler2.dist_based_sampling_pointly(fraction, label_col=args.predict, use_eval=args.use_eval)
        # else:
        #     sampler2.update_weights(beta_factor=temp_beta_factor, use_eval=args.use_eval, label_col=label_col)
        #     sampler2.dist_based_sampling_regionly(fraction, label_col=label_col, use_eval=args.use_eval)
        # outs = train_samples(sampler2, fraction, args.model, sampling_method, idx, args.device)
        # results.append(outs)

        # print('D.B.S. 3')
        # idx += 1
        # if args.modality == 'poi':
        #     sampler3.dist_based_sampling_pointly(fraction, label_col=label_col, use_eval=args.use_eval)
        # else:
        #     sampler3.update_weights(beta_factor=adaptor.beta_factor, use_eval=args.use_eval, label_col=label_col)
        #     sampler3.dist_based_sampling_regionly(fraction, label_col=label_col, use_eval=args.use_eval)
        # outs = train_samples(sampler3, fraction, args.model, sampling_method, idx, args.device)
        # results.append(outs)
        # adaptor.update_records(sample_size=outs['sample_size'], accuracy=outs['val_acc'])
        # adaptor.fit_logistic_law()
        # adaptor.adjust_beta_factor()

        # print('D.B.S. 4')
        # idx += 1
        # if args.modality == 'poi':
        #     sampler4.dist_based_sampling_pointly(fraction, label_col=label_col, use_eval=args.use_eval)
        # else:
        #     sampler4.update_weights(beta_factor=1, use_eval=args.use_eval, label_col=label_col)
        #     sampler4.dist_based_sampling_regionly(fraction, label_col=label_col, use_eval=args.use_eval)
        # outs = train_samples(sampler4, fraction, args.model, sampling_method, idx, args.device)
        # results.append(outs)

    # convert a list of dict into dataframe
    results = pd.DataFrame(results)

    # save results to .csv
    results.to_csv(osp.join(out_dir, f'{args.season}_{args.model}_acc_sl.csv'), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-samples', type=str, default='fast', help='season to use')
    parser.add_argument('-season', type=str, default='summer', help='season to use')
    parser.add_argument('-level', type=int, default=1, help='level of classification system')
    parser.add_argument('-predict', type=str, default=None, help='pseudo label for classification')
    parser.add_argument('-use_eval', action='store_true', help='use eval set')
    parser.add_argument('-modality', type=str, choices=['poi', 'roi'], default='poi')
    parser.add_argument('-overwrite', action='store_true', help='overwrite gmm results')
    parser.add_argument('-model', type=str, default='xgboost', help='model to use')
    parser.add_argument('-device', type=str, default='cuda:0', help='cuda to use')

    args = parser.parse_args()
    logging.info("start running")
    # ======================================================================
    main()
    # ======================================================================
    logging.info("everything is under control")
