# -*- coding: utf-8 -*-
"""
@author: Liaoyh
@contact: liaoyh21@mails.tsinghua.edu.cn
@file: proc_samples.py
@time: 2025/4/11 14:50
"""
import os
import os.path as osp
from pdb import set_trace as here
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

from utils import process_in_batch

sample_dir = '../../assets/LCSamples'

CAT_TRANS = {
    'GLC_FCS30D': {
        0: 0, 250: 0, 255: 0,
        10: 10, 11: 10, 12: 10, 20: 10,
        50: 20, 51: 20, 52: 20, 61: 20, 62: 20, 71: 20, 72: 20, 81: 20, 82: 20, 91: 20, 92: 20, 90: 20, 60: 20, 70: 20,
        80: 20,
        120: 40, 121: 40, 122: 40,
        130: 30,
        140: 70,
        181: 50, 182: 50, 183: 50, 184: 50, 185: 50, 186: 50, 187: 50, 180: 50,
        190: 80, 150: 90, 152: 90, 153: 90, 200: 90, 201: 90, 202: 90, 210: 60, 220: 100
    },
    'ESA_WC': {
        0: -2, 10: 20, 20: 40, 30: 30, 40: 10, 50: 80, 60: 90, 70: 100, 80: 60, 90: 50, 95: 50, 100: 70
    },
    'IO_WC': {
        0: -2, 1: 60, 2: 20, 3: -2, 4: 50, 5: 10, 6: -2, 7: 80, 8: 90, 9: 100, 10: -2, 11: 30
    },
    'DW': {
        0: 60, 1: 20, 2: 30, 3: 50, 4: 10, 5: 40, 6: 80, 7: 90, 8: 100
    }
}


def pd2gpd(data, coords=['lon', 'lat'], crs='epsg:4326'):
    """
    Convert a plain DataFrame with numeric coords into a GeoDataFrame.
    """
    return gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[coords[0]], data[coords[1]]), crs=crs)


def load_fast(season='', level=1, predict=None, randomly_split=True):
    fp = osp.join(sample_dir, 'FAST', 'sr', 'prod_enhanced_fast.csv')
    samples = pd.read_csv(fp, index_col=0)

    bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    samples = samples.loc[(samples[bands] > 0).all(axis=1)]
    # samples.ID = samples.ID.astype(str) + '-' + samples.subset

    # remove drop class
    samples = samples[~samples['Cat1'].isin([12])]

    if not predict:
        predict = 'Class'

    # split samples into train and valid
    tr_samples = samples.query('subset == "train"').copy()
    val_samples = samples.query('subset == "valid"').copy()
    if predict is not None:
        if predict == 'Class':
            pass
        elif predict == 'ESA':
            tr_samples['Cat1'] = tr_samples['ESA_WC2021']
        elif predict == 'GLC_FCS':
            tr_samples['Cat1'] = tr_samples['GLC_FCS2015']
        elif predict == 'DW':
            tr_samples['Cat1'] = tr_samples['DW_2017']
        elif predict == 'Composite2':
            tr_samples['Cat1'] = tr_samples['Composite2']
        elif predict == 'Composite3':
            tr_samples['Cat1'] = tr_samples['Composite3']
        else:
            raise ValueError("predict must be None, 'Class', 'ESA', 'GLC_FCS', 'DW', 'Composite2' or 'Composite3'")
    tr_samples = tr_samples[tr_samples['Cat1'] != 0]
    tr_samples = tr_samples[~pd.isna(tr_samples['Cat1'])]
    samples = pd.concat([tr_samples, val_samples], ignore_index=True)

    try:
        season = int(season)
    except:
        pass

    # filtering month
    if isinstance(season, str):
        if season == 'spring':
            nh_months = [3, 4, 5]
        elif season == 'summer':
            nh_months = [6, 7, 8]
        elif season == 'fall':
            nh_months = [9, 10, 11]
        elif season == 'winter':
            nh_months = [12, 1, 2]
        else:
            nh_months = list(range(1, 13))
    elif isinstance(season, int):
        nh_months = [season]
    else:
        raise ValueError("season must be a string or an integer")

    samples['month'] = samples['date'].apply(lambda x: x.split('-')[1]).astype(int)
    sh_months = [(month + 6) % 12 for month in nh_months]
    sh_months = [month if month != 0 else 12 for month in sh_months]

    samples = samples.query('(lat >= 0 and month in @nh_months) or (lat < 0 and month in @sh_months)')

    # change the name of Category (according to level)
    samples.loc[:, ['Class']] = samples[f'Cat{level}']
    samples.loc[:, ['ClassName']] = samples[f'Cat{level}Label']

    if level == 1:
        if season == 'winter':
            samples = samples.loc[~samples.ClassName.isin(['Tundra'])]

    if level == 2:
        samples = samples.loc[~samples.ClassName.isin([
            'Greenhouse', 'Mixedleaf(leaf-off)', 'Marshland(leaf-off)', 'ShrubTundra', 'Needleleaf(leaf-off)'
        ])]

    if randomly_split:
        # randomly split
        grid_size = 0.001  # test good
        samples.loc[:, ['grid_lon']] = (samples['lon'] // grid_size).astype(int)
        samples.loc[:, ['grid_lat']] = (samples['lat'] // grid_size).astype(int)
        samples.loc[:, ['grid_id']] = samples['grid_lon'].astype(str) + '_' + samples['grid_lat'].astype(str)

        # Assign grids to train or valid
        unique_grids = samples['grid_id'].unique()

        # set random seed
        np.random.seed(42)
        train_grids = set(np.random.choice(unique_grids, size=int(0.8 * len(unique_grids)), replace=False))
        samples.loc[:, ['subset']] = samples['grid_id'].apply(lambda grid: 'train' if grid in train_grids else 'valid')
        samples.index = samples.ID.astype(str) + '-' + samples.subset + '-' + samples.date
        assert samples.index.is_unique

    # to geo-df
    samples = pd2gpd(samples)

    return samples


def compute_spectral_indices(samples):
    # add NDVI, mNDWI and NBR
    samples['NDVI'] = (samples['NIR'] - samples['Red']) / (samples['NIR'] + samples['Red'])
    samples['mNDWI'] = (samples['Green'] - samples['NIR']) / (samples['Green'] + samples['NIR'])
    samples['NBR'] = (samples['NIR'] - samples['SWIR1']) / (samples['NIR'] + samples['SWIR1'])

    return samples


def load_coasttrain(season='summer', level=1):
    fp = osp.join(sample_dir, 'CoastTrain', 'sr', 'sr_ts.csv')
    samples = pd.read_csv(fp, index_col=0)
    bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    samples = samples.loc[(samples[bands] > 0).all(axis=1)]

    # filtering month
    if season == 'spring':
        nh_months = [3, 4, 5]
    elif season == 'summer':
        nh_months = [6, 7, 8]
    elif season == 'fall':
        nh_months = [9, 10, 11]
    elif season == 'winter':
        nh_months = [12, 1, 2]
    else:
        nh_months = list(range(1, 13))

    samples['month'] = samples['date'].apply(lambda x: x.split('-')[1]).astype(int)
    sh_months = [(month + 6) % 12 for month in nh_months]
    sh_months = [month if month != 0 else 12 for month in sh_months]

    samples = samples.query('(lat >= 0 and month in @nh_months) or (lat < 0 and month in @sh_months)')

    # change the name of Category (according to level)
    samples.loc[:, ['Class']] = samples[f'Cat{level}'].astype(int)
    samples.loc[:, ['ClassName']] = samples[f'Cat{level}Label']

    # Divide samples into 0.5° x 0.5° grids
    # grid_size = 0.1 # test good
    grid_size = 0.001  # test good
    samples.loc[:, ['grid_lon']] = (samples['lon'] // grid_size).astype(int)
    samples.loc[:, ['grid_lat']] = (samples['lat'] // grid_size).astype(int)
    samples.loc[:, ['grid_id']] = samples['grid_lon'].astype(str) + '_' + samples['grid_lat'].astype(str)

    # Assign grids to train or valid
    unique_grids = samples['grid_id'].unique()

    # set random seed
    np.random.seed(42)
    train_grids = set(np.random.choice(unique_grids, size=int(0.8 * len(unique_grids)), replace=False))
    samples.loc[:, ['subset']] = samples['grid_id'].apply(lambda grid: 'train' if grid in train_grids else 'valid')

    # Assign unique index
    samples.index = samples.ID.astype(str) + '-' + samples.date

    # Compute spectral indices
    samples = compute_spectral_indices(samples)

    # to geo-df
    samples = pd2gpd(samples)

    return samples


def load_lucas18sum(season='summer', level=1):
    fp = osp.join(sample_dir, 'LUCAS', 'sr', 'lucas_2018_sum.csv')
    band_names = ['Blue', 'Green', 'Red', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2']
    if not osp.exists(fp):
        # read all files
        by_id_dir = '/home/liaoyh/DESS/workplace/project/GEELinker/tests/out_lucas_s2/integrated/by_id'
        by_id_files = [osp.join(by_id_dir, file) for file in os.listdir(by_id_dir)]
        by_id_files = [file for file in by_id_files if file.endswith('.csv')]

        # divide files into 100-files slices
        by_id_files = [by_id_files[i:i + 1000] for i in range(0, len(by_id_files), 1000)]
        temp_out_dir = osp.join(sample_dir, 'LUCAS', 'sr', 'temp')
        os.makedirs(temp_out_dir, exist_ok=True)
        for idx_files, files in enumerate(tqdm(by_id_files)):
            temp_out_fp = osp.join(temp_out_dir, f'{idx_files}.csv')
            if osp.exists(temp_out_fp):
                continue

            # combine all files
            all_samples = []
            for file in files:
                df = pd.read_csv(file)
                all_samples.append(df)
            all_samples = pd.concat(all_samples, ignore_index=True)

            # filter invalid data
            all_samples = all_samples[np.isfinite(all_samples['QA60'])]

            # preprocess
            # Define bitmasks
            cloud_bit_mask = 1 << 10  # Bit 10
            cirrus_bit_mask = 1 << 11  # Bit 11
            # Apply bitmask logic
            all_samples['QA60'] = all_samples['QA60'].astype(np.uint16)
            clear_conditions = (all_samples['QA60'] & cloud_bit_mask == 0) & (
                    all_samples['QA60'] & cirrus_bit_mask == 0)
            all_samples = all_samples[clear_conditions]

            # renaming bands
            all_samples.rename(
                columns={'B2': 'Blue', 'B3': 'Green', 'B4': 'Red', 'B5': 'RE1', 'B6': 'RE2', 'B7': 'RE3',
                         'B8': 'NIR', 'B11': 'SWIR1', 'B12': 'SWIR2'},
                inplace=True)
            all_samples[band_names] = all_samples[band_names] / 10000

            all_samples = all_samples[['ID', 'lon', 'lat'] + band_names]

            # group by id and aggregate bands by median
            all_samples.ID = all_samples.ID.astype(int).astype(str)
            all_samples = all_samples.groupby('ID').agg(
                lon=('lon', 'first'),
                lat=('lat', 'first'),
                **{band: (band, 'median') for i, band in enumerate(band_names)}
            ).reset_index()

            # load sample infos
            sample_infos = pd.read_csv(
                '/home/liaoyh/DESS/workplace/project/GEELinker/tests/out_lucas_s2/infos.csv', index_col=0
            )
            sample_infos.rename(columns={'survey_date': 'date', 'lc1': 'Cat2', 'lc1_label': 'Cat2Label'}, inplace=True)
            sample_infos = sample_infos[['ID', 'date', 'Cat2', 'Cat2Label']]
            cat_sys = {'A': 'Building', 'B': 'Croplands', 'C': 'Forest', 'D': 'Shrubland', 'E': 'Grassland',
                       'F': 'Bareland', 'G': 'Water', 'H': 'Wetlands'}
            sample_infos['Cat1'] = sample_infos['Cat2'].apply(lambda x: x[0])
            sample_infos['Cat1Label'] = sample_infos['Cat1'].apply(lambda x: cat_sys[x])
            sample_infos['ID'] = sample_infos['ID'].astype(str)

            # merge sample infos into all-samples
            all_samples = all_samples.merge(sample_infos, on='ID', how='left')

            # save to csv
            all_samples.to_csv(temp_out_fp, index=False)

        # merge all temp files and clear temp dir
        temp_samples = [pd.read_csv(osp.join(temp_out_dir, fn)) for fn in os.listdir(temp_out_dir)]
        all_samples = pd.concat(temp_samples)
        all_samples.to_csv(fp, index=False)

    samples = pd.read_csv(fp)
    samples = samples.loc[(samples[band_names] > 0).all(axis=1)]

    # filtering month
    if season == 'spring':
        nh_months = [3, 4, 5]
    elif season == 'summer':
        nh_months = [6, 7, 8]
    elif season == 'fall':
        nh_months = [9, 10, 11]
    elif season == 'winter':
        nh_months = [12, 1, 2]
    else:
        nh_months = list(range(1, 13))

    samples['month'] = samples['date'].apply(lambda x: x.split('/')[1]).astype(int)
    sh_months = [(month + 6) % 12 for month in nh_months]
    sh_months = [month if month != 0 else 12 for month in sh_months]

    samples = samples.query('(lat >= 0 and month in @nh_months) or (lat < 0 and month in @sh_months)')

    # change the name of Category (according to level)
    samples.loc[:, ['Class']] = samples[f'Cat{level}']
    samples.loc[:, ['ClassName']] = samples[f'Cat{level}Label']

    if level == 2:
        samples = samples[samples.Class.isin(samples.Class.value_counts()[samples.Class.value_counts() >= 21].index)]

    # Divide samples into 0.1° x 0.1° grids
    grid_size = 0.001
    samples.loc[:, ['grid_lon']] = (samples['lon'] // grid_size).astype(int)
    samples.loc[:, ['grid_lat']] = (samples['lat'] // grid_size).astype(int)
    samples.loc[:, ['grid_id']] = samples['grid_lon'].astype(str) + '_' + samples['grid_lat'].astype(str)

    # Assign grids to train or valid
    unique_grids = samples['grid_id'].unique()
    train_grids = set(np.random.choice(unique_grids, size=int(0.8 * len(unique_grids)), replace=False))
    samples.loc[:, ['subset']] = samples['grid_id'].apply(lambda grid: 'train' if grid in train_grids else 'valid')

    # to geo-df
    samples = pd2gpd(samples)

    return samples


def load_lucas(season='summer', level=1, use_agg=True):
    fp = osp.join(sample_dir, 'LUCAS', 'sr', f'lucas_{season}{" " if not use_agg else "_agg"}.csv')
    band_names = ['Blue', 'Green', 'Red', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2']

    if not osp.exists(fp):
        # read all year
        all_year_fp = osp.join(sample_dir, 'LUCAS', 'sr', f'lucas_all_year.csv')
        if not osp.exists(all_year_fp):
            proc_lucas(all_year_fp)

        samples = pd.read_csv(all_year_fp)
        samples = samples.loc[(samples[band_names] > 0).all(axis=1)]

        try:
            season = int(season)
        except:
            pass

        # filtering month
        if isinstance(season, str):
            if season == 'spring':
                nh_months = [3, 4, 5]
            elif season == 'summer':
                nh_months = [6, 7, 8]
            elif season == 'fall':
                nh_months = [9, 10, 11]
            elif season == 'winter':
                nh_months = [12, 1, 2]
            else:
                nh_months = list(range(1, 13))
        elif isinstance(season, int):
            nh_months = [season]
        else:
            raise ValueError("season must be a string or an integer")

        sh_months = [(month + 6) % 12 for month in nh_months]
        sh_months = [month if month != 0 else 12 for month in sh_months]

        samples = samples.query('(lat >= 0 and month in @nh_months) or (lat < 0 and month in @sh_months)')

        # samples.year = samples['image_date'].apply(lambda x: x.split('-')[0]).astype(int)
        if use_agg:
            if season == 'all_year':
                samples.loc[:, ['img_month']] = samples['image_date'].apply(lambda x: x.split('-')[1]).astype(int)
                samples = samples.query(
                    '(lat >= 0 and img_month in @nh_months) or (lat < 0 and img_month in @sh_months)')
                samples = samples.loc[samples.month == samples.img_month]

            # take the median according to the ID keep the columns except band_names the first
            samples = samples.groupby('ID').agg(
                **{elem: (elem, 'median') if elem in band_names else (elem, 'first') for elem in
                   samples.columns.difference(['ID'])}
            ).reset_index()

        # Divide samples into 0.1° x 0.1° grids
        # grid_size = 0.05
        # grid_size = 0.5
        # grid_size = 0.001
        # grid_size = 0.05
        grid_size = 0.001
        samples.loc[:, ['grid_lon']] = (samples['lon'] // grid_size).astype(int)
        samples.loc[:, ['grid_lat']] = (samples['lat'] // grid_size).astype(int)
        samples.loc[:, ['grid_id']] = samples['grid_lon'].astype(str) + '_' + samples['grid_lat'].astype(str)

        # Assign grids to train or valid
        # set random seed
        np.random.seed(42)
        unique_grids = samples['grid_id'].unique()
        train_grids = set(np.random.choice(unique_grids, size=int(0.8 * len(unique_grids)), replace=False))
        samples.loc[:, ['subset']] = samples['grid_id'].apply(lambda grid: 'train' if grid in train_grids else 'valid')

        samples.to_csv(fp, index=False)

    # to geo-df
    samples = pd.read_csv(fp)

    # change the name of Category (according to level)
    samples.loc[:, ['Class']] = samples[f'Cat{level}']
    samples.loc[:, ['ClassName']] = samples[f'Cat{level}Label']

    if level == 2:
        samples = samples[samples.Class.isin(samples.Class.value_counts()[samples.Class.value_counts() > 500].index)]

    # Compute spectral indices
    samples = compute_spectral_indices(samples)

    # box-muller transform
    # samples['bm_lon'] = (samples['lon'] - samples['lon'].min()) / (samples['lon'].max() - samples['lon'].min())
    # samples['bm_lat'] = (samples['lat'] - samples['lat'].min()) / (samples['lat'].max() - samples['lat'].min())
    # samples.loc[samples['bm_lon'] <= 0, ['bm_lon']] = 1e-6
    # samples.loc[samples['bm_lat'] <= 0, ['bm_lat']] = 1e-6
    # samples['bm_lon'] = np.cos(2 * np.pi * samples['bm_lon']) * (-2 * np.log(samples['bm_lat'])) ** 0.5
    # samples['bm_lat'] = np.sin(2 * np.pi * samples['bm_lon']) * (-2 * np.log(samples['bm_lat'])) ** 0.5

    samples = pd2gpd(samples)

    return samples


def proc_lucas(fp):
    band_names = ['Blue', 'Green', 'Red', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2']
    base_dir = '/home/liaoyh/DESS/workplace/project/GEELinker/tests/'

    for year in [2018, 2022]:
        out_dir = f'out_lucas_s2_{str(year)[-2:]}'
        org_by_id_dir = osp.join(base_dir, out_dir, 'integrated', 'by_id')
        fix_by_id_dir = org_by_id_dir.replace(out_dir, f'{out_dir}_fix_mis')
        fps = []
        for file in os.listdir(org_by_id_dir):
            if file.endswith('.csv'):
                org_fp = osp.join(org_by_id_dir, file)
                fix_fp = osp.join(fix_by_id_dir, file)
                if not osp.exists(fix_fp):
                    fps.append(fix_fp)
                else:
                    fps.append(org_fp)

        # load sample infos
        sample_infos = pd.read_csv(osp.join(base_dir, out_dir, 'infos.csv'), index_col=0)
        sample_infos.rename(columns={'survey_date': 'date', 'lc1': 'Cat2', 'lc1_label': 'Cat2Label'}, inplace=True)
        sample_infos = sample_infos[['ID', 'date', 'Cat2', 'Cat2Label']]
        cat_sys = {'A': 'Building', 'B': 'Croplands', 'C': 'Forest', 'D': 'Shrubland', 'E': 'Grassland',
                   'F': 'Bareland', 'G': 'Water', 'H': 'Wetlands'}
        sample_infos['Cat1'] = sample_infos['Cat2'].apply(lambda x: x[0])
        sample_infos['Cat1Label'] = sample_infos['Cat1'].apply(lambda x: cat_sys[x])

        # Convert date format
        if year == 2018:
            sample_infos['date'] = sample_infos['date'].apply(
                lambda x: datetime.strptime(x, '%d/%m/%y').strftime('%Y-%m-%d')
            )
        elif year == 2022:
            sample_infos['date'] = sample_infos['date'].apply(lambda x: x.split(' ')[0])
        else:
            raise ValueError("year must be 2018 or 2022")
        sample_infos['month'] = sample_infos['date'].apply(lambda x: x.split('-')[1]).astype(int)

        # divide files into 100-files slices
        by_id_files = [fps[i:i + 1000] for i in range(0, len(fps), 1000)]
        temp_out_dir = osp.join(sample_dir, 'LUCAS', 'sr', 'temp')
        os.makedirs(temp_out_dir, exist_ok=True)
        ipts_set = []
        for idx_files, files in enumerate(tqdm(by_id_files)):
            ipts_set.append([idx_files, files, year, temp_out_dir, band_names, sample_infos])
        process_in_batch(_proc_lucas_batch_files, ipts_set, para=True, num_workers=20)

    # merge all temp files and clear temp dir
    temp_samples = [pd.read_csv(osp.join(temp_out_dir, fn)) for fn in os.listdir(temp_out_dir)]
    all_samples = pd.concat(temp_samples)
    all_samples.to_csv(fp, index=False)


def _proc_lucas_batch_files(ipts):
    idx_files, files, year, temp_out_dir, band_names, sample_infos = ipts
    temp_out_fp = osp.join(temp_out_dir, f'{year}-{idx_files}.csv')
    if osp.exists(temp_out_fp):
        return

    # combine all files
    all_samples = []
    for file in files:
        df = pd.read_csv(file)
        if df.shape[0] > 0:
            all_samples.append(df)
    if len(all_samples) == 0:
        return
    all_samples = pd.concat(all_samples, ignore_index=True)

    # filter invalid data
    all_samples = all_samples[np.isfinite(all_samples['QA60'])]

    # preprocess
    # Define bitmasks
    cloud_bit_mask = 1 << 10  # Bit 10
    cirrus_bit_mask = 1 << 11  # Bit 11
    # Apply bitmask logic
    all_samples['QA60'] = all_samples['QA60'].astype(np.uint16)
    clear_conditions = (all_samples['QA60'] & cloud_bit_mask == 0) & (
            all_samples['QA60'] & cirrus_bit_mask == 0)
    all_samples = all_samples[clear_conditions]

    # renaming bands
    all_samples.rename(
        columns={'B2': 'Blue', 'B3': 'Green', 'B4': 'Red', 'B5': 'RE1', 'B6': 'RE2', 'B7': 'RE3',
                 'B8': 'NIR', 'B11': 'SWIR1', 'B12': 'SWIR2'},
        inplace=True)
    all_samples[band_names] = all_samples[band_names] / 10000
    all_samples['image_date'] = all_samples['id'].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}")

    all_samples = all_samples[['ID', 'image_date', 'lon', 'lat'] + band_names]

    # merge sample infos into all-samples
    all_samples = all_samples.merge(sample_infos, on='ID', how='left')

    # save to csv
    all_samples.to_csv(temp_out_fp, index=False)


def enhance_fast_by_products():
    # load original fast
    fp = osp.join(sample_dir, 'FAST', 'sr', 'enhanced_fast.csv')
    samples = pd.read_csv(fp, index_col=0)
    samples.ID = samples.ID.astype(str) + '-' + samples.subset
    tr_samples = samples.query('subset == "train"').copy()
    val_samples = samples.query('subset == "valid"').copy()

    # 1: combine ESA WC data
    esa_out = osp.join(sample_dir, 'FAST', 'prod', 'FAST_ESA_WC_AllYear_2021.csv')
    # load esa data
    esa_data = pd.read_csv(esa_out, index_col=0)
    esa_data = esa_data[esa_data.subset == 'train']
    esa_data = esa_data[~pd.isna(esa_data.Map)]
    esa_data['Map'] = esa_data['Map'].apply(lambda x: CAT_TRANS['ESA_WC'][x]) // 10
    esa_data = esa_data.drop_duplicates(['ID'])
    esa_data = esa_data[['ID', 'Map']]
    # merge esa data to tr_samples
    tr_samples = tr_samples.merge(esa_data, on='ID', how='left')
    tr_samples.rename(columns={'Map': 'ESA_WC2021'}, inplace=True)

    # 2: combine GLC_FCS30D data
    glc_out = osp.join(sample_dir, 'FAST', 'prod', 'GLC_FCS30-2015.csv')
    # load glc data
    glc_data = pd.read_csv(glc_out, index_col=0)
    glc_data = glc_data[glc_data.subset == 'train']
    glc_data.ID = glc_data.ID.astype(str) + '-' + glc_data.subset
    glc_data.rename(columns={"org-GLC_FCS30-2015": "GLC_FCS2015"}, inplace=True)
    glc_data = glc_data[~pd.isna(glc_data["GLC_FCS2015"])]
    glc_data['GLC_FCS2015'] = glc_data['GLC_FCS2015'].apply(lambda x: CAT_TRANS['GLC_FCS30D'][x]) // 10
    glc_data = glc_data.drop_duplicates(['ID'])
    glc_data = glc_data[['ID', 'GLC_FCS2015']]
    # merge esa data to tr_samples
    tr_samples = tr_samples.merge(glc_data, on='ID', how='left')

    # 3: combine DW data
    dw_out = osp.join(sample_dir, 'FAST', 'prod', 'FAST_DynamicWorld_17.csv')
    # load dw data
    dw_data = pd.read_csv(dw_out, index_col=0)
    dw_data = dw_data[dw_data.subset == 'train']
    dw_data.rename(columns={"label_mode": "DW_2017"}, inplace=True)
    dw_data = dw_data[~pd.isna(dw_data["DW_2017"])]
    dw_data['DW_2017'] = dw_data['DW_2017'].apply(lambda x: CAT_TRANS['DW'][x]) // 10
    dw_data = dw_data.drop_duplicates(['ID'])
    dw_data = dw_data[['ID', 'DW_2017']]
    # merge esa data to tr_samples
    tr_samples = tr_samples.merge(dw_data, on='ID', how='left')

    # 4: composite of three kinds of land cover
    def calculate_composites(row):
        values = [row['GLC_FCS2015'], row['DW_2017']]
        if row['GLC_FCS2015'] != 10 and row['DW_2017'] != 10:
            values.append(row['ESA_WC2021'])
        unique_values = list(set(values))
        consistent_count = len(unique_values)

        composite2 = unique_values[0] if consistent_count <= 2 else np.nan
        composite3 = unique_values[0] if consistent_count == 1 else np.nan

        return pd.Series([composite2, composite3])

    tr_samples[['Composite2', 'Composite3']] = tr_samples.apply(calculate_composites, axis=1)

    # Ensure all extra columns in tr_samples are added to val_samples with None
    extra_columns = set(tr_samples.columns) - set(val_samples.columns)
    for col in extra_columns:
        val_samples[col] = None

    # Combine tr_samples and val_samples
    combined_samples = pd.concat([tr_samples, val_samples], ignore_index=True)

    # save the combined samples
    combined_samples.to_csv(osp.join(sample_dir, 'FAST', 'sr', 'prod_enhanced_fast.csv'))
