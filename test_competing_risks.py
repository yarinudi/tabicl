"""
Test script for TabICLCompetingRisks implementation.

Generates synthetic competing risks data and evaluates the model.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter

from src.tabicl.sklearn.competing_risks import TabICLCompetingRisks
from src.tabicl.sklearn.competing_risks_multimodal import TabICLCompetingRisksMultiModal
from src.tabicl.train.gait_dataset import get_datasets, get_dataloaders, SurvivalDataset

import xgboost as xgb
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tqdm import tqdm


def load_target_events():
    fpath = '/home/yarinod/PythonProjects/WHS/MWD/resources/data/whs/whs_accel_fract050924_fid.csv'
    df = pd.read_csv(fpath).set_index('newid')
    
    # first fall after the acce wear
    cols_fractpyr = [col for col in df.columns if 'fractp' in col]
    df['first_fall_post'] = df[df[cols_fractpyr] > 0][cols_fractpyr].min(axis=1)
    df['first_fall_post_label'] = (~df['first_fall_post'].isna()).astype(int)
    df['first_fall_post'] = df['first_fall_post'].fillna(df['randyears'])

    # define events of interest
    cols_events = [
        # K = 11
        'randyears', 
        # 'depressyrs', 
        # 'Fibromyalgiayrs',
        'jointyrs', 
        # 'MSyrs', 
        'Osteoarthritisyrs', 
        'parkyrs', 
        # 'Rheumayrs', 
        'strokeyears', 
        'tiayrs', 
        'first_fall_post',
    ]

    cols_labels = [
        'death',
        # 'depression',
        # 'Fibromyalgia',
        'jointreplace',
        # 'Multiple_sclerosis',
        'Osteoarthritis',
        'parkinson',
        # 'Rheuma_arthritis',
        'stkconf',
        'tiaunr',
        'first_fall_post_label'
    ]

    def format_events(events, labels):
        temp_events = {}

        for i, (e, t) in enumerate(events.items()):
            label = labels.iloc[i] if t > 0 else 0
            t_censored = events['randyears']
            t = t_censored if label == 0 else t
            temp_events[f'{e}_time'] = t
            temp_events[f'{e}_label'] = label

        return temp_events

    targets = {}
    for idx, events in df.iterrows():
        targets[idx] = format_events(events[cols_events], events[cols_labels])

    targets = pd.DataFrame.from_dict(targets, orient='index')
    return targets


# ##### XGBoost Survival Helpers
def build_cause_specific_labels(times, events, idx=None):
    """Build cause-specific labels for XGBoost survival model."""
    K = 7
    time_k, event_k = [], []
    
    first_events = True  # True for legacy use | False for using all available event tags
    if not first_events:
        # ALL EVENTS
        targets = load_target_events()
        events = ['randyears','jointyrs', 'Osteoarthritisyrs', 'parkyrs', 'strokeyears', 'tiayrs', 'first_fall_post']
        targets = targets.iloc[idx, :]

        for e in events:
            col_time = "_".join([e, 'time'])
            col_label = "_".join([e, 'label'])
            times = targets.loc[:, col_time].values.reshape(-1, 1)
            events = targets.loc[:, col_label].values.reshape(-1, 1)
            time_k.append(times)
            event_k.append(events.astype(np.int64))

        return time_k, event_k

    # First Event Only [Legacy]
    for k in range(1, K + 1):
        time_k.append(times)
        event_k.append((events == k).astype(np.int64))

    return time_k, event_k


# Recover Breslow baselines and build CIF/Survival predictors
def breslow_baseline(times, events, risk_scores):
    if times.shape[1] == 1:
        times = times.squeeze(1)
    if events.shape[1] == 1:        
        events = events.squeeze(1)
    order = np.argsort(times)
    t = times[order]; e = events[order]; r = np.exp(risk_scores[order])

    # unique event times
    t_event = np.unique(t[e==1])
    H0 = []; dH0 = []
    for te in t_event:
        # events at te
        d = ((t == te) & (e == 1)).sum()

        # risk set: those with time >= te
        at_risk = (t >= te)
        R = r[at_risk].sum()
        inc = d / R
        dH0.append(inc)
        H0.append(inc if len(H0) == 0 else H0[-1] + inc)
    return t_event, np.asarray(H0, float), np.asarray(dH0, float)


def fit_baselines_for_all_causes(models, x_train, time_k, event_k):
    baselines = []
    for k, bst in enumerate(models):
        dtrain = xgb.DMatrix(x_train)
        # xgboost Cox predict() returns log risk (margin). exp(.) -> HR
        f = bst.predict(dtrain, output_margin=True)  # (N,)
        t_ev, H0, dH0 = breslow_baseline(np.asarray(time_k[k]), np.asarray(event_k[k]), f)
        baselines.append((t_ev, H0, dH0))
    return baselines


def predict_cif_and_survival(models, baselines, x, t_grid=None):
    K = len(models)
    M = x.shape[0]

    # 1) unified time grid
    if t_grid is None:
        all_times = np.unique(np.concatenate([b[0] for b in baselines if len(b[0]) > 0]))
    
    else:
        all_times = np.asarray(t_grid, float)
    
    T = all_times.size

    # 2) baseline increments aligned to grid
    dH0_grid = np.zeros((K, T), dtype=np.float64)
    for k, (t_ev, _, dH0) in enumerate(baselines):
        if len(t_ev) == 0: 
            continue
        idx = np.searchsorted(all_times, t_ev)
        dH0_grid[k, idx] = dH0
    
    # 3) subject-specific HR per cause
    dm = xgb.DMatrix(x)
    f = np.stack([bst.predict(dm, output_margin=True) for bst in models], axis=1)  # (M, K)
    HR = np.exp(f)

    # 4) forward recursion to get S and CIFs
    F = np.zeros((M, K, T), dtype=np.float64)
    S = np.zeros((M, T), dtype=np.float64)
    for t in tqdm(range(T)):
        S_prev = S[:, t-1] if t > 0 else np.ones(M, dtype=np.float64)  # (M,)
        dH_k_t = dH0_grid[:, t]

        # increments per cause: (M, K)
        dF_k = S_prev[:, None] * HR * dH_k_t[None, :]

        # update overall survival multiplicatively
        dH_tot = (HR * dH_k_t[None, :]).sum(axis=1)  # (M,)
        S[:, t] = S_prev * np.exp(-dH_tot)

        # accumulate CIFs
        F[:, :, t] = dF_k if t == 0 else F[:, :, t-1] + dF_k
    
    return F, S, all_times


def train_xgb_cox_survival(x, times, events, val=None, params=None, num_boost_round=1000, train_idx=None, val_idx=None):
    """Train XGBoost Cox survival model with cause-specific labels."""
    if params is None:
        params = {
            "seed": 42,
            "objective": 'survival:cox',
            "eval_metric": 'cox-nloglik',
            "tree_method": 'hist',
            "learning_rate": 0.05,
            "max_depth": 1,  # Decision stumps
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
    
    time_k, event_k = build_cause_specific_labels(times, events, idx=train_idx)
    if val:
        x_val, (times_val, events_val) = val
        time_k_val, event_k_val = build_cause_specific_labels(times_val, events_val, idx=val_idx)
    
    models = []
    watchlist = None
    for k in range(len(time_k)):
        # Build signed labels: positive for events, negative for censored
        y_signed = np.where(event_k[k] == 1, time_k[k], -time_k[k]).astype(np.float32)
        dtrain = xgb.DMatrix(x, label=y_signed)
        
        if val:
            y_signed_val = np.where(event_k_val[k] == 1, time_k_val[k], -time_k_val[k]).astype(np.float32)
            dval = xgb.DMatrix(x_val, label=y_signed_val)
            watchlist = [(dtrain, 'train'), (dval, 'eval')]
        
        bst = xgb.train(
            params, 
            dtrain, 
            evals=watchlist, 
            early_stopping_rounds=50, 
            verbose_eval=10,
            num_boost_round=num_boost_round
        )
        models.append(bst)

    return models, (time_k, event_k)


def predict_xgb_cox(models, x):
    """Predict risk scores from XGBoost Cox models."""
    risks = []
    for m in models:
        dtest = xgb.DMatrix(x)
        risks_temp = m.predict(dtest)
        risks.append(risks_temp)

    return np.array(risks).T


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    
def train_val_split(X, Y, group, val_size=0.125):
    num_split = 1
    folds = GroupShuffleSplit(
        num_split, test_size=val_size, random_state=42
    ).split(X, Y, groups=group)
    train_idx, val_idx = next(folds)
    Y_train = Y[train_idx] if isinstance(Y, np.ndarray) else Y.iloc[train_idx]
    Y_val = Y[val_idx] if isinstance(Y, np.ndarray) else Y.iloc[val_idx]
    return X.iloc[train_idx], X.iloc[val_idx], Y_train, Y_val, train_idx, val_idx


def load_data():
    Y = pd.read_pickle('survival_data_target.pkl')

    fpath = '/home/yarinod/PythonProjects/WHS/MWD/resources/data/whs/whs_accel_data_fid.csv'
    dates_wear = pd.read_csv(fpath)[['newid', 'date']].set_index('newid')
    dates_wear['date'] = pd.to_datetime(dates_wear['date']).apply(lambda t: t.weekday())
    dir_ts = '/data/WHS/WHS/WHS_Processed_Data/yarinudi/processed/mtl_ssl/data/ssl/whs/data/valid_windws_timestamps'

    cols_tabular = [
        'hearing_trouble', 'hearing_loss_expos', 
        'sit_work', 'sit_hometv', 'sithome', 
        'RACE', 'ageaccel', 'genhealth', 'bmi', 'histhtn',
        'smoke', 'weight', 'alcuse', 'heightbs',
        'SF12_vigact', 
        'SF12_modact', 'SF12_lifting', 'SF12_climbsev', 'SF12_climbone',
        'SF12_bending', 'SF12_walkmile', 'SF12_walkblocks', 'SF12_walkblock',
        'SF12_bath', 'SF12_cut_time', 'SF12_phys_accless', 'SF12_limitedwork',
        'SF12_diff_work', 'SF12_emot_accless', 'SF12_not_careful', 'SF12_pain',
        'SF12_felt_calm', 'SF12_energy', 'SF12_felt_down', 'SF12_soc_act', 
        'SF12year',
        'EDUC', 
        'tmethrst', 'walkpace',
        'stairclimb', 'running', 'biking', 'jogging', 'swim', 'aerobic',
        'walking', 'lifting', 'lowexer', 'othrexer'
    ]
    dict_conditions = {
        'cols_events': ['jointyrs', 'Osteoarthritisyrs', 'parkyrs', 'strokeyears', 'tiayrs'],
        'cols_labels': ['jointreplace', 'Osteoarthritis', 'parkinson', 'stkconf', 'tiaunr']
    }
    fpath = '/home/yarinod/PythonProjects/WHS/MWD/resources/data/whs/whs_accel_fract050924_fid.csv'
    X_feats = pd.read_csv(fpath).set_index('newid')

    # I extracted the TableOne table1 table 1 from here
    # generate_table1(X_feats, paper="both", save=True, out_dir='/home/yarinod/PythonProjects/python_modules/TransmitorJ/data_for_papers')

    # Add number of windows 
    n_winds_mapping = {fname.split('.')[0]: np.load(os.path.join(dir_ts, fname)).shape[0] for fname in os.listdir(dir_ts)}
    X_feats['n_winds'] = X_feats.index.map(n_winds_mapping)

    # Normalize N winds with the length of experiments
    dates_wear = pd.read_csv('/home/yarinod/PythonProjects/WHS/MWD/resources/data/whs/whs_accel_data_fid.csv')
    dates_wear['date'] = pd.to_datetime(dates_wear['date']).apply(lambda t: t.weekday())
    total_steps_mapping = {id: gr['total_steps'].sum() for id, gr in dates_wear.groupby('newid')}
    X_feats['total_steps'] = X_feats.index.map(total_steps_mapping)

    n_days_mapping = {id: int(gr['date'].count()) for id, gr in dates_wear.groupby('newid')}
    X_feats['n_days_wear'] = X_feats.index.map(n_days_mapping)
    X_feats['n_winds_norm'] = X_feats['n_winds'] / X_feats['n_days_wear']
    X_feats['total_steps_norm'] = X_feats['total_steps'] / X_feats['n_days_wear']

    # Cadence
    with open('cadence_mapping_v2.pkl', 'rb') as file:
        map = pickle.load(file)

    mean_spm = {k: v['mean_spm'] for k, v in map.items()}
    X_feats['mean_spm'] = X_feats.index.map(mean_spm)

    mean_std_spm = {k: v['mean_std_spm'] for k, v in map.items()}
    X_feats['mean_std_spm'] = X_feats.index.map(mean_std_spm)

    sem_spm = {k: v['sem_spm'] for k, v in map.items()}
    X_feats['sem_spm'] = X_feats.index.map(sem_spm)

    cols_tabular.extend([
        'n_winds_norm', 'total_steps_norm', 'n_days_wear',
        'mean_spm', 'mean_std_spm', 'sem_spm'
    ])

    # Add conditions features
    def format_events(events, labels):
        temp_events = {}
        for i, (e, t) in enumerate(events.items()):
            label = labels.iloc[i] if t <= 0 else 0
            t_index_day = np.inf
            t = t_index_day if label == 0 else t
            temp_events[f'had_{e}_time'] = 1 / (1 + np.abs(t))
            
        return temp_events

    feats_conds = {}
    for idx, events in X_feats.iterrows():
        feats_conds[idx] = format_events(events[dict_conditions['cols_events']], 
                                            events[dict_conditions['cols_labels']])

    feats_conds = pd.DataFrame.from_dict(feats_conds, orient='index')
    X_feats = pd.concat([X_feats, feats_conds], axis=1)

    cols_tabular.extend([
        *[col for col in X_feats.columns if 'had' in col]
    ])

    X_feats = X_feats.loc[:, cols_tabular]

    # Add sway feats
    df_feats_sway = pd.read_pickle('df_feats_sway.pkl').sort_index().astype(float)
    X_feats = pd.concat([X_feats, df_feats_sway], axis=1)

    with open('groups.pkl', 'rb') as file:
        groups = pickle.load(file)

    # Time to first negative event evaluation
    any_event = False
    if any_event:
        Y['event_k'] = (Y['event_k'] > 0).astype(int)

    # # Add Amit feats - 20260309
    # df_feats_amit = pd.read_pickle('df_feats_amit_20260309.pkl').sort_index()
    # df_feats_amit = df_feats_amit.drop(columns='Subject')
    # X_feats = pd.merge(X_feats, df_feats_amit, left_index=True, right_index=True)
    # groups = groups[np.isin(groups, X_feats.index.to_numpy())]
    # Y = Y.loc[groups]

    # Ablation Study - Remove subjects of 1 year, 2 year, 5 year
    t_ablation = None  # [year]
    if t_ablation is not None:
        print(f"Removing subjects with bad events before {t_ablation}")
        Y['time_star'] = Y['time_star'] - t_ablation
        Y = Y[Y['time_star'] > 0]
        idx_keep = Y.index
        X_feats = X_feats.loc[idx_keep]
        groups = groups[np.isin(groups, idx_keep)]

    num_splits = 5
    folds = GroupShuffleSplit(
        num_splits, test_size=0.2, random_state=42
    ).split(X_feats, Y, groups=groups)

    # Informative NaNs handling
    # cols_nans = X_feats.loc[:, (X_feats.isna().mean() > 0)].columns
    cols_fill_0 = [
        'othrexer', 
        'hearing_trouble', 'hearing_loss_expos',
    ]
    cols_fill_mean = [
        'sit_work', 'sit_hometv','sithome', 
        'bmi', 'weight', 
        'tmethrst', 
        'RACE', 'genhealth', 'smoke', 'alcuse', 
        'SF12_vigact', 'SF12_modact', 'SF12_lifting', 'SF12_climbsev',
        'SF12_climbone', 'SF12_bending', 'SF12_walkmile', 'SF12_walkblocks',
        'SF12_walkblock', 'SF12_bath', 'SF12_cut_time', 'SF12_phys_accless',
        'SF12_limitedwork', 'SF12_diff_work', 'SF12_emot_accless',
        'SF12_not_careful', 'SF12_pain', 'SF12_felt_calm', 'SF12_energy',
        'SF12_felt_down', 'SF12_soc_act', 
        'SF12year',
        'walkpace', 'stairclimb', 'running', 'biking', 'jogging', 'swim',
        'aerobic', 'walking', 'lifting', 'lowexer', 
        'EDUC'
    ]

    # # Add Amit cols for mean
    # cols_fill_mean.extend(df_feats_amit.columns)

    if all([col in X_feats.columns for col in cols_fill_0]):
        X_feats[cols_fill_0] = X_feats[cols_fill_0].fillna(0)

    return X_feats, Y, groups, folds, cols_fill_mean


def get_fold_data(X_feats, Y, groups, train_idxs, test_idxs, cols_fill_mean, scale_feats=True):
    tmp_X_train, X_test = X_feats.iloc[train_idxs], X_feats.iloc[test_idxs]
    tmp_Y_train, Y_test = Y[train_idxs] if isinstance(Y, np.ndarray) else Y.iloc[train_idxs], Y[test_idxs] if isinstance(Y, np.ndarray) else Y.iloc[test_idxs]
    group_train = groups[train_idxs]

    # We further divide up train into 70/10 train/val split
    X_train, X_val, Y_train, Y_val, fold_train_idx, fold_val_idx = train_val_split(
        tmp_X_train, tmp_Y_train, group_train
    )

    # Normalize times according to maximal training time
    cols_time = [col for col in Y_train.columns if 'time' in col]
    t_max = Y_train[cols_time].values.max()
    Y_train.loc[:, cols_time] = (Y_train[cols_time] / t_max).values
    Y_val.loc[:, cols_time] = (Y_val[cols_time] / t_max).values
    Y_test.loc[:, cols_time] = (Y_test[cols_time] / t_max).values
    print("Normalized times to [0, 1] according to maximal time in train...")

    # Fill NaNs with mean value
    if all([col in X_train.columns for col in cols_fill_mean]):
        mean_vals = X_train[cols_fill_mean].mean()
        X_train[cols_fill_mean] = X_train[cols_fill_mean].fillna(mean_vals)
        X_val[cols_fill_mean] = X_val[cols_fill_mean].fillna(mean_vals)
        X_test[cols_fill_mean] = X_test[cols_fill_mean].fillna(mean_vals)

    # Standard scale tabular features
    if scale_feats:
        tabular_scaler = StandardScaler().set_output(transform='pandas')
        X_train = tabular_scaler.fit_transform(X_train)
        X_val = tabular_scaler.transform(X_val)
        X_test = tabular_scaler.transform(X_test)
        print("Normalized tabular features...")

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
        X_val = X_val.to_numpy()
        X_test = X_test.to_numpy()

    get_target = lambda df: (df['time_star'].values, df['event_k'].values)
    y_train = get_target(Y_train)
    y_val = get_target(Y_val)
    y_test = get_target(Y_test)

    t_train, e_train = y_train
    t_val, e_val = y_val
    t_test, e_test = y_test

    argsortttest = np.argsort(t_test)
    t_test = t_test[argsortttest]
    e_test = e_test[argsortttest]
    X_test = X_test[argsortttest,:]

    return X_train, (t_train, e_train), X_val, (t_val, e_val), X_test, (t_test, e_test), fold_train_idx, fold_val_idx, t_max


if __name__ == "__main__":
    set_seed(42)

    print("="*70)
    print("TABICL COMPETING RISKS TEST")
    print("="*70)

    X_feats, Y, groups, folds, cols_fill_mean = load_data()
    n_event_types = K = len(np.unique(Y['event_k'].values)) - 1  # 7

    if n_event_types == 7:
        cols_events = [
            'randyears', 
            # 'depressyrs', 
            # 'Fibromyalgiayrs',
            'jointyrs', 
            # 'MSyrs', 
            'Osteoarthritisyrs', 
            'parkyrs', 
            # 'Rheumayrs', 
            'strokeyears', 
            'tiayrs', 
            'first_fall_post',
        ]
    else:
        cols_events = ['1st_negative_event']

    # Subjective - self-reports
    cols_subjective = [
        'hearing_trouble', 'hearing_loss_expos', 
        'sit_work', 'sit_hometv', 'sithome', 
        'SF12_vigact', 
        'SF12_modact', 'SF12_lifting', 'SF12_climbsev', 'SF12_climbone',
        'SF12_bending', 'SF12_walkmile', 'SF12_walkblocks', 'SF12_walkblock',
        'SF12_bath', 'SF12_cut_time', 'SF12_phys_accless', 'SF12_limitedwork',
        'SF12_diff_work', 'SF12_emot_accless', 'SF12_not_careful', 'SF12_pain',
        'SF12_felt_calm', 'SF12_energy', 'SF12_felt_down', 'SF12_soc_act', 
        'SF12year',
        'tmethrst', 'walkpace',
        'stairclimb', 'running', 'biking', 'jogging', 'swim', 'aerobic',
        'walking', 'lifting', 'lowexer', 'othrexer',
        'genhealth','smoke', 'alcuse'
    ]
        
    # Objective - EHR
    cols_objective = [
        'RACE', 'ageaccel', 'bmi', 'histhtn', 'weight', 'heightbs', 'EDUC', 
    ]

    # Sway features - accelerometer derived
    cols_sway = pd.read_pickle('df_feats_sway.pkl').columns.to_list()
    # cols_amit = pd.read_pickle('df_feats_amit_20260309.pkl').columns.to_list()
    # cols_amit.remove('Subject')

    cols_subset_label = 'oqs'
    dict_feature_sets = {
        'oqs': None,
        'oq': [*cols_objective, *cols_subjective],
        'o': cols_objective,
        's': cols_sway,
        'os': [*cols_objective, *cols_sway]
    }
    cols_subset = dict_feature_sets[cols_subset_label]

    def get_feats_set(cols, cols_subset, X_train, X_val, X_test):
        cols_list = cols.to_list()
        cols_idxs = [cols_list.index(c) for c in cols_subset]
        print(f'Subset of Features - # Feats: {len(cols_idxs)}')
        return X_train[:, cols_idxs], X_val[:, cols_idxs], X_test[:, cols_idxs]

    gams_only = False
    scale_feats = not gams_only

    # # DCA Arguments - run when the csv was generated for each feature set
    dca_path = '/home/yarinod/PythonProjects/ssl-wearables/tabicl/dca/data/folds'
    # folds_dirs = os.listdir(dca_path)

    # for fold_name in folds_dirs:
    #     print("="*70)
    #     print(f'Working on Fold {fold_name}')
    #     print("="*70)
    #     dca_path_fold = os.path.join(dca_path, fold_name)
    #     fold_files = [os.path.join(root, f) for root, _, filenames in os.walk(dca_path_fold) for f in filenames]
        
    #     for e in cols_events:
    #         e_files = {f.split('/')[-2]: f for f in fold_files if e in f}
    #         df_dca_fold = pd.DataFrame(columns=['time_years', 'event'])
    #         for k, f in e_files.items():
    #             df_temp = pd.read_csv(f)
    #             temp_times, temp_event = df_temp['time_years'], df_temp['event']
    #             cols_risks = [col for col in df_temp.columns if 'risk' in col]
    #             df_temp = df_temp[cols_risks]
    #             new_cols = {col: "_".join([col, k]) for col in cols_risks}
    #             df_temp = df_temp.rename(columns=new_cols)
    #             df_dca_fold = pd.concat([df_dca_fold, df_temp], axis=1)
            
    #         df_dca_fold['time_years'] = temp_times
    #         df_dca_fold['event'] = temp_event
            
    #         # Save to event path
    #         df_dca_fold.to_csv(os.path.join(dca_path_fold, f'df_dca_{e}.csv'), index=False)
    
    # # Concatenate all folds to the same csv for DCA curve generation
    # for e in cols_events:
    #     print("="*70)
    #     print(f'Working on Outcome {e}')
    #     print("="*70)
    #     df_all_folds = []
        
    #     for fold_name in folds_dirs:
    #         f = os.path.join(dca_path, fold_name, f'df_dca_{e}.csv')
    #         df_temp = pd.read_csv(f)
    #         df_all_folds.append(df_temp)

    #     # Save concatenated dataframe per event
    #     df_all_folds = pd.concat(df_all_folds)
    #     df_all_folds.to_csv(os.path.join(dca_path, f'df_dca_{e}.csv'), index=False)

    # # 3-year
    # python run_dca.py --data-path data/df_dca_pd_input.csv --time-col time_years --event-col event_pd --horizon 5 --event-value 1 --model "Q:risks_q_5y" --model "Q+S:risks_qs_5y" --threshold-min 0.001 --threshold-max 0.008 --threshold-step 0.005 --nri --nri-reference "Q" --nri-cutoffs 0,0.02 --outdir data/output_5y --figure-title "Decision Curve Analysis (5-year PD risk)" --calibration-plot-title "Calibration (5-year PD risk)"

    total_results = []
    total_specificity_results = []
    run_specificity_tests = False
    run_dca_risks_generation = False

    for i, (train_idxs, fold_test_idxs) in enumerate(folds):
        print("\n" + "="*70)
        print(f'##### Fold {i} #####')

        X_train, (t_train, e_train), X_val, (t_val, e_val), X_test, (t_test, e_test), fold_train_idx, fold_val_idx, t_max_train = get_fold_data(X_feats, Y, groups, train_idxs, fold_test_idxs, cols_fill_mean, scale_feats=scale_feats)
        if cols_subset is not None:
            X_train, X_val, X_test = get_feats_set(cols=X_feats.columns, cols_subset=cols_subset, X_train=X_train, X_val=X_val, X_test=X_test)

        if not gams_only:
            dca_path_fold = os.path.join(dca_path, f'fold_{i}', cols_subset_label)
            os.makedirs(dca_path_fold, exist_ok=True)

            ################################# Dataloaders for Multim-modal Training
            train_dataset, val_dataset, test_dataset = get_datasets(
                fold_train_idx, fold_val_idx, fold_test_idxs, 
                X_train, X_val, X_test, 
                t_train, e_train, t_val, e_val, t_test, e_test)

            train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_dataset, val_dataset, test_dataset)

            # Train TabICL model with Gait Time-series Signals
            print("\n" + "="*70)
            print("TRAINING TABICL COMPETING RISKS MULTI MODAL MODEL")
            print("="*70)
            model_tabicl_multi = TabICLCompetingRisksMultiModal(
                n_event_types=n_event_types,
                backbone="tabicl",
                checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
                hidden=128,
                epochs=100,
                patience=5,  # 15,
                lr=1e-3,
                verbose=True,
                device='cuda:1',
                multi_cox_heads=True  # True := Train one model for all cause; False := train a different model (K models)
            )
            
            print("\nFitting TabICL model...")
            model_tabicl_multi.fit(
                train_dataloader, val_dataloader
            )

            # Compare with XGBoost Cox survival model (max_depth=1)
            print("\n" + "-"*60)
            print("XGBOOST COX SURVIVAL (max_depth=1, decision stumps)")
            print("-"*60)

            # Prepare data for XGBoost Cox model (needs shape (n_samples, 1) for times/events)
            times_train = t_train.reshape(-1, 1)
            events_train = e_train.reshape(-1, 1)
            times_val = t_val.reshape(-1, 1)
            events_val = e_val.reshape(-1, 1)
            times_test = t_test.reshape(-1, 1)
            events_test = e_test.reshape(-1, 1)
            
            print("Training XGBoost Cox model...")
            xgb_models, (time_k, event_k) = train_xgb_cox_survival(
                X_train, 
                times_train, 
                events_train,
                val=(X_val, (times_val, events_val)),
                params={
                    "seed": 42,
                    "objective": 'survival:cox',
                    "eval_metric": 'cox-nloglik',
                    "tree_method": 'hist',
                    "learning_rate": 0.05,
                    "max_depth": 1,  # 0 - set to not limit depth  # "max_leaves": 20, # 0
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                },
                num_boost_round=1000,
                train_idx=fold_train_idx, 
                val_idx=fold_val_idx
            )

            if run_dca_risks_generation:
                print('Fitting baselines for all causes...')
                baselines = fit_baselines_for_all_causes(xgb_models, X_train, time_k, event_k)

                print('Evaluating XGB models...')
                F_pred, S_pred, t_grid = predict_cif_and_survival(xgb_models, baselines, X_test)
                t_grid_denorm = t_grid * t_max_train

                # Wrapper for DCA
                for idx_event, event in enumerate(cols_events):
                    horizon_years = [1, 3, 5, 7, 10]

                    horizon_indices = {}
                    for h in horizon_years:
                        h_norm = h / t_max_train
                        if h_norm > t_grid.max():
                            print(f'WARNING: {h}-year {h_norm:.3f} exceedes t_grid max ')

                        t_idx = np.argmin(np.abs(t_grid - h_norm))
                        horizon_indices[h] = t_idx
                        print(f"Horizon {h}y -> normalized {h_norm:.4f}, "
                            f"matched t_grad[{t_idx}] = {t_grid[t_idx]:.4f} "
                            f"({t_grid[t_idx] * t_max_train:.2f} years)"
                            )

                    risks_dca = {}
                    for h, t_idx in horizon_indices.items():
                        risks_dca[h] = F_pred[:, idx_event, t_idx]
                
                    df_dca_temp = pd.DataFrame(columns=['time_years', 'event', 'risks_1y', 'risks_3y', 'risks_5y', 'risks_7y', 'risks_10y'])
                    df_dca_temp['time_years'] = (times_test * t_max_train).squeeze(1)
                    df_dca_temp['event'] = (events_test == idx_event + 1).astype(int)
                    df_dca_temp['risks_1y'] = risks_dca[1]
                    df_dca_temp['risks_3y'] = risks_dca[3]
                    df_dca_temp['risks_5y'] = risks_dca[5]
                    df_dca_temp['risks_7y'] = risks_dca[7]
                    df_dca_temp['risks_10y'] = risks_dca[10]

                    df_dca_temp.to_csv(os.path.join(dca_path_fold, f'df_{event}_dca.csv'), index=False)

            # Evaluation
            print("\n" + "="*70)
            print("EVALUATION RESULTS")
            print("="*70)
            
            results = {} 
            risks = model_tabicl_multi.predict_cause_specific_risk(test_dataloader)
            risks_xgb = predict_xgb_cox(xgb_models, X_test)

            from src.tabicl.sklearn.survivor import _concordance_index

            for k in range(1, n_event_types + 1):
                events_k = (e_test == k).astype(float)
                risk_k = risks[:, k - 1]
                c_tabicl_multi = _concordance_index(t_test, events_k, risk_k)
                xgb_c_index = _concordance_index(t_test, events_k, risks_xgb[:, k-1])
            
                print(f"Outcome {k}")
                print(f"\n✓ XGBoost C-Index:  {xgb_c_index:.4f}")
                print(f"✓ TabICL C-Index:   {c_tabicl_multi:.4f}")
                print(f"\nTabICL vs XGBoost:  {c_tabicl_multi - xgb_c_index:+.4f}")
                                
                results[f'k-{k}-TabICL'] = c_tabicl_multi
                results[f'k-{k}-XGBoost'] = xgb_c_index
                
            print("\n" + "="*60)
            total_results.append(results)

            if run_specificity_tests:
                print("COVARIANCE EVALUATION RESULTS - SPECIFICITY")
                print("="*70)
                
                specificity_results = np.zeros((n_event_types, n_event_types)) 
                for idx_m, m in enumerate(xgb_models):
                    risks_xgb = predict_xgb_cox([m]*n_event_types, X_test)
                    for k in range(1, n_event_types + 1):
                        events_k = (e_test == k).astype(float)                   
                        specificity_results[idx_m, k - 1] = _concordance_index(t_test, events_k, risks_xgb[:, k - 1])

                print("\n" + "="*60)
                total_specificity_results.append(specificity_results)
        
        else:
            # Prepare data for XGBoost Cox model (needs shape (n_samples, 1) for times/events)
            times_train = t_train.reshape(-1, 1)
            events_train = e_train.reshape(-1, 1)
            times_val = t_val.reshape(-1, 1)
            events_val = e_val.reshape(-1, 1)
            times_test = t_test.reshape(-1, 1)
            events_test = e_test.reshape(-1, 1)

            # GAM View
            from xgbGAMView import xgbGAMView

            ##### Fit Cause Specific xgbGAMViewer #####
            viewers = []
            cols_feats = X_feats.columns
            df_x_train = pd.DataFrame(X_train, columns=cols_feats)
            df_x_val = pd.DataFrame(X_val, columns=cols_feats)
            df_x_test = pd.DataFrame(X_test, columns=cols_feats)

            time_k, event_k = build_cause_specific_labels(times_train, events_train, idx=fold_train_idx)
            time_k_val, event_k_val = build_cause_specific_labels(times_val, events_val, idx=fold_val_idx)
            time_k_test, event_k_test = build_cause_specific_labels(times_test, events_test, idx=fold_test_idxs)

            fig_bar_plot = make_subplots(rows=n_event_types, cols=1,
                                subplot_titles=[f'Feature Importance (Gain) -- {l}' for l in cols_events])

            for k in range(n_event_types):
                event_label = cols_events[k]
                print(f"Outcome: {k + 1} - {event_label}")
                y_signed = np.where(event_k[k] == 1, time_k[k], -time_k[k]).astype(np.float32)
                dtrain = xgb.DMatrix(df_x_train, label=y_signed)
                y_signed_val = np.where(event_k_val[k] == 1, time_k_val[k], -time_k_val[k]).astype(np.float32)
                dval = xgb.DMatrix(df_x_val, label=y_signed_val)
                watchlist = [(dtrain, 'train'), (dval, 'eval')]
                train_params = {'num_boost_round': 1000, 'evals': watchlist, 'early_stopping_rounds': 50, 'verbose_eval': 10}
                params = {
                    "seed": 42,
                    "objective": 'survival:cox',
                    "eval_metric": 'cox-nloglik',
                    "tree_method": 'hist',
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                }

                # Fit viewer
                viewer = xgbGAMView(param=params)
                viewer.fit(df_x_train, y_signed, train_params)
                viewers.append(viewer)

                # Predict - absolute risks
                preds_val = viewer.predict(df_x_val)
                preds_test = viewer.predict(df_x_test)

                # Contribution of all the features for 1 sample prediction
                # contribution = viewer.feature_contribution(df_x_val.iloc[1: 2])
                # contribution.T[contribution.T['Contribution'] > 0]
                outputs_home = 'plots'
                outputs_dir = f'xgbGAMViewPlots/fold_{i}/outcome_{k}_{event_label}'
                full_outputs_dir = os.path.join(outputs_home, outputs_dir)
                os.makedirs(full_outputs_dir, exist_ok=True)

                # Generate plots for the 10 most important features
                feats = list(viewer.X.columns[viewer.X.isna().mean(axis=0) == 0])
                importance = viewer.model.get_score(importance_type='gain')
                df_importance = pd.DataFrame(importance.items(), columns=['Feature', 'Gain'])
                df_importance = df_importance.sort_values(by='Gain', ascending=False)
                
                # Write all features to csv
                df_importance.to_csv(os.path.join(full_outputs_dir, 'feature_importance.csv'))
                df_importance = df_importance.iloc[:10]
                feats = df_importance['Feature'].values.tolist()

                if viewer.X[feats].isna().any().any():
                    viewer.X[feats] = viewer.X[feats].fillna(viewer.X[feats].mean(axis=0))
                viewer.plot(name=outputs_dir, features=feats)  # Selected features | to get all features set features=None

                # Feature importance Bar Plot
                fig_bar_plot.add_trace(
                    go.Bar(name=f'{cols_events[i]}', x=df_importance['Gain'], y=df_importance['Feature'], orientation='h'),
                    row=k + 1, col=1
                )

            fig_bar_plot.write_html(os.path.join(full_outputs_dir.rsplit('/', 1)[0], f'feature_importance_all_outcomes.html'))

        print("\n" + "="*70)

    #########################################
    df_results = pd.DataFrame(total_results)
    stats = df_results.describe().T
    print(stats)

    print("\n" + "="*70)
    print(stats.round(3))

    ctds_list = [[result[f"ctd-index-k{i}"] for result in total_results] for i in range(1, 8)]
    for a in range(len(ctds_list)):
        print("####################### Event ", a + 1)
        print(f"{np.mean(ctds_list[a]).round(3)} +/- {np.std(ctds_list[a]).round(3)}")

    # Aggregate specificity covariance
    if run_specificity_tests:
        df_specificity = np.array(total_specificity_results)
        means = pd.DataFrame(df_specificity.mean(axis=0).round(3), columns=cols_events, index=cols_events)
        stds = pd.DataFrame(df_specificity.std(axis=0).round(3), columns=cols_events, index=cols_events)

###################### Timestamps Evaluations ######################


# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# fpath = '/home/yarinod/PythonProjects/WHS/MWD/resources/data/whs/whs_accel_data_fid.csv'
# dates_wear = pd.read_csv(fpath).set_index('newid')

# for i in range(2):
#     idx = 14500 + i
#     new_id = f'ZU0{idx}'
#     secs_continuous = torch.from_numpy(np.load(os.path.join(train_dataset.dir_ts, f'ZU0{idx}.npy'))).float()  # [seconds]
#     mins_continuous = secs_continuous // 60 // 60 # [hrs]
#     days = pd.to_datetime(dates_wear.loc[new_id, 'date']).apply(lambda t: t.weekday())
#     days = days if isinstance(days, np.int64) else days.values

#     fig = make_subplots(specs=[[{"secondary_y": True}]])
#     # fig.add_trace(go.Scatter(x=list(range(len(secs_continuous))), y=secs_continuous.numpy(), name='secs_continuous'), secondary_y=False)
#     fig.add_trace(go.Scatter(x=list(range(len(days))), y=days, name='days'), secondary_y=False)
#     fig.add_trace(go.Scatter(x=list(range(len(mins_continuous))), y=mins_continuous.numpy(), name='mins_continuous'), secondary_y=True)
#     fig.update_layout(title=f'ZU0{idx}')
#     fig.show()
"""
    There is an issue with using the days and seconds information.
    We don't know what is the initial time of the wearing. That is, time of day is not accurate.
"""
