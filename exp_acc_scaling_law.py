# -*- coding: utf-8 -*-
"""
@author: Liaoyh
@contact: liaoyh21@mails.tsinghua.edu.cn
@file: exp_acc_scaling_law.py
@time: 2025/4/11 15:10
"""
import os
import os.path as osp
import logging
import argparse
from pdb import set_trace as here
import math
from tqdm import tqdm
import random
import warnings
from functools import partial

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from utils import process_in_batch
import proc_samples

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def assign_unit(samples, grid_size, ref=None):
    """
    Assign samples into different units.
    """
    samples.loc[:, ['unit_x']] = (samples['lon'] // grid_size * grid_size).astype(int)
    samples.loc[:, ['unit_y']] = (samples['lat'] // grid_size * grid_size).astype(int)
    samples.loc[:, ['unit']] = samples['unit_x'].astype(str) + ' ' + samples['unit_y'].astype(str)
    if ref is not None:
        ref_units = ref['unit'].unique()
        sample_units = samples['unit'].unique()

        # Find units in samples that are not in ref
        missing_units = set(sample_units) - set(ref_units)

        if missing_units:
            # Map missing units to the closest unit in ref
            ref_coords = pd.Series(ref_units).str.split(' ', expand=True).astype(int).to_numpy()
            for missing_unit in missing_units:
                missing_coords = np.array(missing_unit.split(' ')).astype(int)
                distances = np.sqrt(np.sum((ref_coords - missing_coords) ** 2, axis=1))
                closest_unit = ref_units[np.argmin(distances)]
                samples.loc[samples['unit'] == missing_unit, ['unit']] = closest_unit
    return samples


def randomly_point_sampling(tr_samples, val_samples, tr_sample_size, val_sample_size, seed):
    """
    Randomly sample a subset of the data.
    """
    tr_samples = tr_samples.sample(tr_sample_size, random_state=seed)
    partial_val_samples = val_samples.sample(val_sample_size, random_state=seed)
    return tr_samples, partial_val_samples


def randomly_unit_sampling(tr_samples, val_samples, tr_sample_size, val_sample_size, grid_size, seed):
    # assign samples into different units
    tr_samples = assign_unit(tr_samples, grid_size)
    val_samples = assign_unit(val_samples, grid_size, ref=tr_samples)

    # randomly select unit
    unit_sample_sizes = pd.DataFrame(tr_samples['unit'].value_counts())
    unit_sample_sizes = unit_sample_sizes.sample(frac=1, random_state=seed)
    unit_sample_sizes.reset_index(inplace=True)

    unit_sample_sizes['acc_sample_sizes'] = unit_sample_sizes.cumsum()['count']
    end_idx = unit_sample_sizes[unit_sample_sizes['acc_sample_sizes'] >= tr_sample_size].index[0]
    selected_units = unit_sample_sizes.iloc[:end_idx + 1]['unit']

    # sample from selected units
    tr_samples = tr_samples[tr_samples['unit'].isin(selected_units)]
    partial_val_samples = val_samples[val_samples['unit'].isin(selected_units)]

    return tr_samples, partial_val_samples


def sequentially_unit_sampling(tr_samples, val_samples, tr_sample_size, val_sample_size, grid_size, seed):
    # assign samples into different units
    tr_samples = assign_unit(tr_samples, grid_size)
    val_samples = assign_unit(val_samples, grid_size, ref=tr_samples)

    # predefine center of each continent
    centers = {
        'NA': [-103.9, 43.14], 'SA': [-59.96, -15.87],
        'AS': [87.09, 37.09], 'EU': [50.18, 58.5], 'AF': [21.7, 8.84],
        # 'AU': [132.44, -26.03]
    }
    # randomly choose one
    random.seed(seed)
    continent = random.choice(list(centers.keys()))
    center_x, center_y = centers[continent]

    # Get the coordinates of each unit
    unit_coords = tr_samples['unit'].unique()
    unit_coords = pd.DataFrame(unit_coords, columns=['unit'])
    unit_coords[[0, 1]] = unit_coords['unit'].str.split(' ', expand=True).astype(int)

    # Find the initial seed unit (closest to the center)
    unit_coords.loc[:, ['distance']] = np.sqrt((unit_coords[0] - center_x) ** 2 + (unit_coords[1] - center_y) ** 2)
    initial_unit = unit_coords.sort_values('distance').iloc[0]['unit']

    # Sequentially select nearest units
    selected_units = [initial_unit]
    remaining_units = set(tr_samples['unit'].unique()) - set(selected_units)
    cumulative_sample_size = tr_samples[tr_samples['unit'].isin(selected_units)].shape[0]

    while cumulative_sample_size <= tr_sample_size and remaining_units:
        current_unit = selected_units[-1]
        current_coords = np.array(current_unit.split(' ')).astype(int)

        # Calculate distances to remaining units
        distances = []
        for unit in remaining_units:
            unit_coords = np.array(unit.split(' ')).astype(int)
            distance = np.sqrt(np.sum((unit_coords - current_coords) ** 2))
            distances.append((unit, distance))

        # Find the nearest unit(s)
        distances = sorted(distances, key=lambda x: x[1])
        nearest_distance = distances[0][1]
        nearest_units = [unit for unit, dist in distances if dist == nearest_distance]

        # Randomly select one of the nearest units
        next_unit = random.choice(nearest_units)
        selected_units.append(next_unit)
        remaining_units.remove(next_unit)

        # Update cumulative sample size
        cumulative_sample_size = tr_samples[tr_samples['unit'].isin(selected_units)].shape[0]

    # Sample from selected units
    tr_samples = tr_samples[tr_samples['unit'].isin(selected_units)]
    partial_val_samples = val_samples[val_samples['unit'].isin(selected_units)]

    return tr_samples, partial_val_samples


def sampling(tr_samples, val_samples, tr_sample_size, val_sample_size, sampling_method, seed=42):
    if sampling_method['name'] == 'randomly_point_sampling':
        sampled_tr_samples, sampled_val_samples = randomly_point_sampling(tr_samples, val_samples, tr_sample_size,
                                                                          val_sample_size, seed)
        return sampled_tr_samples, sampled_val_samples
    elif sampling_method['name'] == 'randomly_unit_sampling':
        sampled_tr_samples, sampled_val_samples = randomly_unit_sampling(tr_samples, val_samples, tr_sample_size,
                                                                         val_sample_size,
                                                                         sampling_method['grid_size'], seed)
        return sampled_tr_samples, sampled_val_samples
    elif sampling_method['name'] == 'sequentially_unit_sampling':
        sampled_tr_samples, sampled_val_samples = sequentially_unit_sampling(tr_samples, val_samples, tr_sample_size,
                                                                             val_sample_size,
                                                                             sampling_method['grid_size'], seed)
        return sampled_tr_samples, sampled_val_samples


def train_once(ipts):
    """
    Train a model once and evaluate its performance.
    """
    tr_samples, val_samples, fraction, model, sampling_method, idx, device, seed = ipts

    if sampling_method['name'] in ['homo_based_sampling', 'dist_based_sampling', 'bayesian_dist_based_sampling']:
        partial_val_samples, val_samples = val_samples
        tr_sample_size = tr_samples.shape[0]
        val_sample_size = partial_val_samples.shape[0]
    else:
        # set data
        tr_sample_size = int(tr_samples.shape[0] * fraction)
        val_sample_size = int(val_samples.shape[0] * fraction)

        # sample data
        tr_samples, partial_val_samples = sampling(tr_samples, val_samples, tr_sample_size, val_sample_size,
                                                   sampling_method, seed=seed)
        tr_sample_size = tr_samples.shape[0]

    feats_names = sampling_method.get('feats', ['lon', 'lat', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'])

    tr_X = tr_samples[feats_names]
    tr_y = tr_samples['Class']
    val_samples = val_samples[val_samples['Class'].isin(tr_y.unique())]
    partial_val_samples = partial_val_samples[partial_val_samples['Class'].isin(tr_y.unique())]
    val_X = val_samples[feats_names]
    val_y = val_samples['Class']
    p_val_X = partial_val_samples[feats_names]
    p_val_y = partial_val_samples['Class']

    if 'lon' in feats_names and 'lat' in feats_names:
        tr_X = tr_X.copy()
        val_X = val_X.copy()
        p_val_X = p_val_X.copy()
        tr_X.loc[:, ['lon', 'lat']] = tr_X.loc[:, ['lon', 'lat']] // 5 * 5
        val_X.loc[:, ['lon', 'lat']] = val_X.loc[:, ['lon', 'lat']] // 5 * 5
        p_val_X.loc[:, ['lon', 'lat']] = p_val_X.loc[:, ['lon', 'lat']] // 5 * 5

    # tr_X.drop(columns=['lon', 'lat'], inplace=True)
    # val_X.drop(columns=['lon', 'lat'], inplace=True)
    # p_val_X.drop(columns=['lon', 'lat'], inplace=True)

    # Encode labels, use val-y to transform data since it's fixed
    label_encoder = LabelEncoder()
    val_y = label_encoder.fit_transform(val_y)
    tr_y = label_encoder.transform(tr_y)
    p_val_y = label_encoder.transform(p_val_y)

    # set model
    if model == 'xgboost':
        # Convert data to DMatrix and set device to GPU
        clf = xgb.XGBClassifier(
            n_estimators=1000,  # Increase the number of trees
            learning_rate=0.01,  # Lower learning rate
            max_depth=10,  # Increase the maximum depth of trees
            subsample=0.8,  # Subsample ratio of the training instances
            colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
            tree_method='hist',  # Use GPU for training
            # predictor='gpu_predictor',
            device=device,
            random_state=42,
            eval_metric='mlogloss',
            n_jobs=-1,
        )
        # Train the model
        clf.fit(tr_X, tr_y)
    elif model == 'rf':
        if 'model_params' in sampling_method:
            model_params = sampling_method['model_params']
            clf = RandomForestClassifier(random_state=seed, **model_params)
        else:
            clf = RandomForestClassifier(n_estimators=500, random_state=seed)
        clf.fit(tr_X, tr_y)
    elif model == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(tr_X, tr_y)
    elif model == 'nn':
        clf = MLPClassifier(hidden_layer_sizes=(512,),
                            # learning_rate_init=0.0001, learning_rate='adaptive',
                            max_iter=10000,
                            # early_stopping=True,
                            random_state=42)
        clf.fit(tr_X, tr_y)
    else:
        ValueError('')

    # evaluate
    tr_preds = clf.predict(tr_X)
    val_preds = clf.predict(val_X)

    # convert predictions to original labels
    tr_y = label_encoder.inverse_transform(tr_y)
    val_y = label_encoder.inverse_transform(val_y)
    tr_preds = label_encoder.inverse_transform(tr_preds)
    val_preds = label_encoder.inverse_transform(val_preds)

    tr_acc = (tr_preds == tr_y).mean()
    val_acc = (val_preds == val_y).mean()

    if len(p_val_X) > 0:
        p_val_preds = clf.predict(p_val_X)
        p_val_y = label_encoder.inverse_transform(p_val_y)
        p_val_preds = label_encoder.inverse_transform(p_val_preds)
        p_val_acc = (p_val_preds == p_val_y).mean()
    else:
        print('No Partial Val Samples Fetched')
        p_val_acc = 0

    # Compute precision, recall, and F1-score
    val_report = classification_report(val_y, val_preds, output_dict=True)
    p_val_report = classification_report(p_val_y, p_val_preds, output_dict=True) if len(p_val_X) > 0 else {}

    val_precision = {k: round(v['precision'], 4) for k, v in val_report.items() if k.isdigit()}
    val_recall = {k: round(v['recall'], 4) for k, v in val_report.items() if k.isdigit()}
    val_f1 = {k: round(v['f1-score'], 4) for k, v in val_report.items() if k.isdigit()}
    p_val_precision = {k: round(v['precision'], 4) for k, v in p_val_report.items() if
                       k.isdigit()} if p_val_report else {}
    p_val_recall = {k: round(v['recall'], 4) for k, v in p_val_report.items() if k.isdigit()} if p_val_report else {}
    p_val_f1 = {k: round(v['f1-score'], 4) for k, v in p_val_report.items() if k.isdigit()} if p_val_report else {}

    val_prec = {f'val_prec_{k}': v for k, v in val_precision.items()}
    val_rec = {f'val_rec_{k}': v for k, v in val_recall.items()}
    val_f1 = {f'val_f1_{k}': v for k, v in val_f1.items()}
    p_val_prec = {f'p_val_prec_{k}': v for k, v in p_val_precision.items()}
    p_val_rec = {f'p_val_rec_{k}': v for k, v in p_val_recall.items()}
    p_val_f1 = {f'p_val_f1_{k}': v for k, v in p_val_f1.items()}

    macro_val_prec = val_report['macro avg']['precision']
    macro_val_rec = val_report['macro avg']['recall']
    macro_val_f1 = val_report['macro avg']['f1-score']
    weighted_val_prec = val_report['weighted avg']['precision']
    weighted_val_rec = val_report['weighted avg']['recall']
    weighted_val_f1 = val_report['weighted avg']['f1-score']

    if len(p_val_X) > 0:
        weighted_p_val_prec = p_val_report['weighted avg']['precision']
        weighted_p_val_rec = p_val_report['weighted avg']['recall']
        weighted_p_val_f1 = p_val_report['weighted avg']['f1-score']
        macro_p_val_prec = p_val_report['macro avg']['precision']
        macro_p_val_rec = p_val_report['macro avg']['recall']
        macro_p_val_f1 = p_val_report['macro avg']['f1-score']
    else:
        weighted_p_val_prec = 0
        weighted_p_val_rec = 0
        weighted_p_val_f1 = 0
        macro_p_val_prec = 0
        macro_p_val_rec = 0
        macro_p_val_f1 = 0

    overall_results_pval = {
        'macro_p_val_prec': macro_p_val_prec,
        'macro_p_val_rec': macro_p_val_rec,
        'macro_p_val_f1': macro_p_val_f1,
        'weighted_p_val_prec': weighted_p_val_prec,
        'weighted_p_val_rec': weighted_p_val_rec,
        'weighted_p_val_f1': weighted_p_val_f1
    }

    overall_results_val = {
        'macro_val_prec': macro_val_prec,
        'macro_val_rec': macro_val_rec,
        'macro_val_f1': macro_val_f1,
        'weighted_val_prec': weighted_val_prec,
        'weighted_val_rec': weighted_val_rec,
        'weighted_val_f1': weighted_val_f1,
    }

    abb_sampling_method = ''.join(list(map(lambda x: x[0].upper(), sampling_method['name'].split('_'))))
    print(
        f"Fract.: {fraction:.4f}, Sampling method: {abb_sampling_method}, "
        f"Tr. samples: {tr_sample_size}, Par. samples: {partial_val_samples.shape[0]} v.s. {val_sample_size} (Exp) "
        f"Tr. Acc.: {tr_acc:.4f}, Val. Acc.: {val_acc:.4f}, Par. Acc.: {p_val_acc:.4f}")
    # print(val_f1)
    if 'desc' in sampling_method:
        sampling_method_str = f"{sampling_method['name']}-{sampling_method['desc']}"
    else:
        sampling_method_str = sampling_method['name']

    out = {
        'idx': idx,
        'fraction': fraction,
        'sample_size': tr_sample_size,
        'sampling_method': sampling_method_str,
        'tr_acc': tr_acc,
        'val_acc': val_acc,
        'p_val_acc': p_val_acc
    }
    out.update(val_prec)
    out.update(val_rec)
    out.update(val_f1)
    out.update(p_val_prec)
    out.update(p_val_rec)
    out.update(p_val_f1)
    out.update(overall_results_val)
    out.update(overall_results_pval)

    return out


def main():
    # set data and model
    if args.samples == 'fast':
        samples = proc_samples.load_fast(season=args.season, level=args.level, randomly_split=False,
                                         predict=args.predict)
    elif args.samples == 'coasttrain':
        samples = proc_samples.load_coasttrain(season=args.season, level=args.level)
    elif args.samples == 'lucas':
        samples = proc_samples.load_lucas(season=args.season, level=args.level)
        # samples = proc_samples.load_lucas18sum(season=args.season, level=args.level)
    else:
        raise ValueError

    tr_samples = samples[samples.subset.isin(['train'])]
    val_samples = samples[samples.subset.isin(['valid'])]
    out_dir = osp.join('./outs/acc_sl', f'{args.samples}{str(args.level) if args.level > 1 else ""}')
    os.makedirs(out_dir, exist_ok=True)
    num_min_samples = 6 * (6 + 1) / 2 * 0.25
    fold_base = 1.1

    # set minimum fraction so that even the most rare class have
    max_fold = math.ceil(math.log(num_min_samples / tr_samples.Class.value_counts().values[-1]) / math.log(fold_base))
    fractions = [fold_base ** fold for fold in range(max_fold, 1)]

    # get n_repeats sequence
    max_n_trails = 30
    min_n_trials = 3

    # # for each fraction and sampling method, train and evaluate model
    ipts_set = []
    idx = 0

    sampling_cfgs = {
        'rps': [{'name': 'randomly_point_sampling'}],
        'biased':
            [
                {'name': 'randomly_unit_sampling', 'desc': 1},
                {'name': 'randomly_unit_sampling', 'desc': 5},
                {'name': 'randomly_unit_sampling', 'desc': 10},
                {'name': 'sequentially_unit_sampling', 'desc': 1},
                {'name': 'sequentially_unit_sampling', 'desc': 5},
                {'name': 'sequentially_unit_sampling', 'desc': 10},
            ],
        'eval_impact': [
            {'name': 'randomly_point_sampling', 'model_params': {'n_estimators': 8}, 'desc': 'mc8'},
            {'name': 'randomly_point_sampling', 'model_params': {'n_estimators': 32}, 'desc': 'mc32'},
            {'name': 'randomly_point_sampling', 'model_params': {'n_estimators': 128}, 'desc': 'mc128'},
            {'name': 'randomly_point_sampling', 'model_params': {'n_estimators': 512}, 'desc': 'mc512'},
            {'name': 'randomly_point_sampling', 'model_params': {'n_estimators': 128},
             'feats': ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'], 'desc': 'wo_coords'},
            {'name': 'randomly_point_sampling', 'model_params': {'n_estimators': 128},
             'feats': ['Blue', 'Green', 'Red'], 'desc': 'rgb'},
            {'name': 'randomly_point_sampling', 'model_params': {'n_estimators': 128},
             'feats': ['lon', 'lat', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
                , 'desc': 'org'},
            {'name': 'randomly_point_sampling', 'model_params': {'n_estimators': 128},
             'feats': ['lon', 'lat', 'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'mNDWI', 'NBR']
                , 'desc': 'spetral_indices'},
        ]
    }
    sampling_cfgs = sampling_cfgs[args.sampling]

    for fraction in tqdm(fractions):
        if args.repeat:
            n_trials = int((max_n_trails - min_n_trials) * (1 - fraction) + min_n_trials)
        else:
            n_trials = 1
        for seed in range(42, 42 + n_trials):
            for sampling_method in sampling_cfgs:
                ipts = (tr_samples, val_samples, fraction, args.model, sampling_method, idx, args.device, seed)
                ipts_set.append(ipts)
                idx += 1

    # process in batch
    results = process_in_batch(
        iterator=ipts_set,
        func=train_once,
        para=True if args.model in ['rf', 'knn'] else False,
        num_workers=args.n_workers,
    )
    # convert a list of dict into dataframe
    results = pd.DataFrame(results)
    results.sort_values(by='idx', inplace=True)

    # save results to .csv
    if args.predict:
        out_fp = osp.join(out_dir, f'{args.predict}_{args.sampling}-{args.season}_{args.model}_acc_sl.csv')
    else:
        out_fp = osp.join(out_dir, f'{args.sampling}-{args.season}_{args.model}_acc_sl.csv')
    results.to_csv(out_fp, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-samples', type=str, default='fast', help='season to use')
    parser.add_argument('-season', type=str, default='summer', help='season to use')
    parser.add_argument('-level', type=int, default=1, help='level of classification system')
    parser.add_argument('-predict', type=str, default='', help='predict value or human labels')
    parser.add_argument('-model', type=str, default='xgboost', help='model to use')
    parser.add_argument('-device', type=str, default='cuda', help='cuda to use')
    parser.add_argument('-repeat', action='store_true', help='repeat the experiment')
    parser.add_argument('-sampling', type=str, default='rps', help='sampling method')
    parser.add_argument('-n_workers', type=int, default=10, help='number of workers')
    args = parser.parse_args()
    logging.info("start running")
    # ======================================================================
    main()
    # ======================================================================
    logging.info("everything is under control")
