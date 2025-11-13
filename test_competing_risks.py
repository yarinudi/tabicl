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

from src.tabicl.sklearn.competing_risks import TabICLCompetingRisks


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    
def train_val_split(X, Y, group, val_size=0.125):
    num_split = 1
    folds = GroupShuffleSplit(
        num_split, test_size=val_size, random_state=42
    ).split(X, Y, groups=group)
    train_idx, val_idx = next(folds)
    Y_train = Y[train_idx] if isinstance(Y, np.ndarray) else Y.iloc[train_idx]
    Y_val = Y[val_idx] if isinstance(Y, np.ndarray) else Y.iloc[val_idx]
    return X.iloc[train_idx], X.iloc[val_idx], Y_train, Y_val


def load_data():
    Y = pd.read_pickle('survival_data_target.pkl')

    fpath = '/home/yarinod/PythonProjects/WHS/MWD/resources/data/whs/whs_accel_data_fid.csv'
    dates_wear = pd.read_csv(fpath)[['newid', 'date']].set_index('newid')
    dates_wear['date'] = pd.to_datetime(dates_wear['date']).apply(lambda t: t.weekday())
    dir_ts = '/data/WHS/WHS/WHS_Processed_Data/yarinudi/processed/mtl_ssl/data/ssl/whs/data/valid_windws_timestamps'

    cols_tabular = [
        'hearing_trouble', 'hearing_loss_expos', 'sit_work', 'sit_hometv', 'sithome', 'RACE', 'ageaccel', 'genhealth', 'bmi', 'histhtn',
        'smoke', 'weight', 'alcuse', 'heightbs','SF12_vigact', 
        'SF12_modact', 'SF12_lifting', 'SF12_climbsev', 'SF12_climbone',
        'SF12_bending', 'SF12_walkmile', 'SF12_walkblocks', 'SF12_walkblock',
        'SF12_bath', 'SF12_cut_time', 'SF12_phys_accless', 'SF12_limitedwork',
        'SF12_diff_work', 'SF12_emot_accless', 'SF12_not_careful', 'SF12_pain',
        'SF12_felt_calm', 'SF12_energy', 'SF12_felt_down', 'SF12_soc_act', 
        'SF12year',
        'EDUC', 'tmethrst', 'walkpace',
        'stairclimb', 'running', 'biking', 'jogging', 'swim', 'aerobic',
        'walking', 'lifting', 'lowexer', 'othrexer'
    ]
    dict_conditions = {
        'cols_events': ['jointyrs', 'Osteoarthritisyrs', 'parkyrs', 'strokeyears', 'tiayrs'],
        'cols_labels': ['jointreplace', 'Osteoarthritis', 'parkinson', 'stkconf', 'tiaunr']
    }
    fpath = '/home/yarinod/PythonProjects/WHS/MWD/resources/data/whs/whs_accel_fract050924_fid.csv'
    X_feats = pd.read_csv(fpath).set_index('newid')

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

    with open('groups.pkl', 'rb') as file:
        groups = pickle.load(file)

    num_splits = 5
    folds = GroupShuffleSplit(
        num_splits, test_size=0.2, random_state=42
    ).split(X_feats, Y, groups=groups)

    # Informative NaNs handling
    # cols_nans = X_feats.loc[:, (X_feats.isna().mean() > 0)].columns
    cols_fill_0 = [
        'othrexer', 'hearing_trouble', 'hearing_loss_expos',
    ]
    cols_fill_mean = [
        'sit_work', 'sit_hometv','sithome', 'bmi', 'weight', 
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
        'aerobic', 'walking', 'lifting', 'lowexer', 'EDUC' 
    ]

    X_feats[cols_fill_0] = X_feats[cols_fill_0].fillna(0)

    return X_feats, Y, groups, folds, cols_fill_mean


if __name__ == "__main__":
    set_seed(42)

    print("="*70)
    print("TABICL COMPETING RISKS TEST")
    print("="*70)

    X_feats, Y, groups, folds, cols_fill_mean = load_data()

    total_results = []
    for i, (train_idxs, test_idxs) in enumerate(folds):
        print(f'##### Fold {i} #####')
        tmp_X_train, X_test = X_feats.iloc[train_idxs], X_feats.iloc[test_idxs]
        tmp_Y_train, Y_test = Y[train_idxs] if isinstance(Y, np.ndarray) else Y.iloc[train_idxs], Y[test_idxs] if isinstance(Y, np.ndarray) else Y.iloc[test_idxs]
        group_train, group_test = groups[train_idxs], groups[test_idxs]

        # We further divide up train into 70/10 train/val split
        X_train, X_val, Y_train, Y_val = train_val_split(
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
        mean_vals = X_train[cols_fill_mean].mean()
        X_train[cols_fill_mean] = X_train[cols_fill_mean].fillna(mean_vals)
        X_val[cols_fill_mean] = X_val[cols_fill_mean].fillna(mean_vals)
        X_test[cols_fill_mean] = X_test[cols_fill_mean].fillna(mean_vals)

        # Standard scale tabular features
        tabular_scaler = StandardScaler().set_output(transform='pandas')
        X_train = tabular_scaler.fit_transform(X_train)
        X_val = tabular_scaler.transform(X_val)
        X_test = tabular_scaler.transform(X_test)

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
            X_val = X_val.to_numpy()
            X_test = X_test.to_numpy()

        print("Normalized tabular features...")

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

        ###################################
        # Train TabICL model
        print("\n" + "="*70)
        print("TRAINING TABICL COMPETING RISKS MODEL")
        print("="*70)
        n_event_types = K = 7
        model_tabicl = TabICLCompetingRisks(
            n_event_types=n_event_types,
            backbone="tabicl",
            checkpoint_version="tabicl-classifier-v1.1-0506.ckpt",
            hidden=128,
            epochs=100,
            patience=15,
            verbose=False, 
            device='cpu'
        )
        
        print("\nFitting TabICL model...")
        model_tabicl.fit(
            X_train, (t_train, e_train),
            X_val=X_val, y_val=(t_val, e_val)
        )
        
        # Train baseline MLP model
        print("\n" + "="*70)
        print("TRAINING BASELINE MLP MODEL")
        print("="*70)
        
        model_baseline = TabICLCompetingRisks(
            n_event_types=n_event_types,
            backbone="mlp",
            hidden=128,
            epochs=100,
            patience=15,
            verbose=False,
            device='cpu'
        )
        
        print("\nFitting baseline model...")
        model_baseline.fit(
            X_train, (t_train, e_train),
            X_val=X_val, y_val=(t_val, e_val)
        )
        
        # Evaluation
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        # Predict cause-specific risks
        risks_tabicl = model_tabicl.predict_cause_specific_risk(X_test)
        risks_baseline = model_baseline.predict_cause_specific_risk(X_test)
        
        print(f"\nPredicted risks shape: {risks_tabicl.shape}")
        
        # Compute C-indices
        print("\nCause-specific C-indices:")
        print(f"{'Cause':<10} {'TabICL':<12} {'Baseline':<12} {'Improvement':<12}")
        print("-" * 50)
        results = {}            
        for k in range(1, n_event_types + 1):
            c_tabicl = model_tabicl.score(X_test, (t_test, e_test), cause=k)
            c_baseline = model_baseline.score(X_test, (t_test, e_test), cause=k)
            
            if not np.isnan(c_tabicl):
                print(f"Cause {k:<5} {c_tabicl:<12.4f} {c_baseline:<12.4f} {c_tabicl - c_baseline:+.4f}")
            else:
                print(f"Cause {k:<5} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

            results[f'tabicl-ctd-index-k{k}'] = c_tabicl
            results[f'baseline-ctd-index-k{k}'] = c_baseline

        c_avg_tabicl = model_tabicl.score(X_test, (t_test, e_test))
        c_avg_baseline = model_baseline.score(X_test, (t_test, e_test))
        
        print("-" * 50)
        print(f"{'Average':<10} {c_avg_tabicl:<12.4f} {c_avg_baseline:<12.4f} {c_avg_tabicl - c_avg_baseline:+.4f}")
        
        # Final summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        print(f"\nTabICL Competing Risks achieves:")
        print(f"  - Average C-index: {c_avg_tabicl:.4f}")
        print(f"  - Improvement over MLP: {c_avg_tabicl - c_avg_baseline:+.4f}")
        
        print("\n" + "="*70)

        print("\n Computing CIFs for test samples...")

        # Get CIFs for all causes
        cifs_tabicl_dict = model_tabicl.predict_cumulative_incidence(X_test, return_array=False)
        cifs, times = [], []
        for k, v in cifs_tabicl_dict.items():
            v = np.stack(v)
            times.append(v[:, 0, :]); cifs.append(v[:, 1, :])
        
        cifs, times = np.array(cifs), np.array(times)  # (K, N, T)
        t_eval = times[0][0]

        # results = {}
        # for i in range(K):
        #     print(f'##### Event {i + 1} #####')
        #     # cause-specific survival S_k = 1 - CIF_k
        #     surv_temp = pd.DataFrame(surv[i], index=t_eval)
        #     ev = EvalSurv(surv_temp, t_test, e_test == i + 1, censor_surv='km')
            
        #     results[f'ctd-index-k{i + 1}'] = np.round(ev.concordance_td(), 3)
        #     results[f'ibrier-score-k{i + 1}'] = ev.integrated_brier_score(time_grid).round(3)
        #     results[f'inbll-k{i + 1}'] = ev.integrated_nbll(time_grid).round(3)

        #     print(f"Ctd-index-{i + 1}: ", results[f'ctd-index-k{i + 1}'])
        #     print(f"Integrated-Brier-Score-{i + 1}: ", results[f'ibrier-score-k{i + 1}'])
        #     print(f"Integrated-NBLL-{i + 1}: ", results[f'inbll-k{i + 1}'])
        

        import plotly.graph_objects as go
        from lifelines import KaplanMeierFitter


        def plot_model_cif(cifs, t_eval):
            """
            Overlay KKM and model mean survival for cause k (0-based)
            """
            K = cifs.shape[0]
            
            fig = go.Figure()
            for k in range(K):
                F_model = cifs.mean(1)[k]

                fig.add_trace(
                    go.Scatter(
                        x=t_eval, y=F_model,
                        mode="lines",
                        name="Model CIF cause " + str(k+1)
                    )
                )
            fig.update_layout(
                title=f"Model CIF - Competing Outcomes - {K} Causes",
                xaxis_title="t",
                yaxis_title="CIF",
                yaxis_range=[0, 1],
                hovermode='x unified'
            )
            fig.show()
            return fig
        
        # Calculate t_eval and mean survival rate from CIFk dictionary

        fig = plot_model_cif(cifs, t_eval)

        # KM vs Overall Survival
        km = KaplanMeierFitter()
        km.fit(durations=t_test, event_observed=(e_test > 0).astype(int))
        t_km, S_km = km.survival_function_.index.values, km.survival_function_['KM_estimate'].values

        S_model = 1 - cifs.sum(axis=0).mean(axis=0)  # mean over N, sum over K
        S_model_std = cifs.sum(axis=0).std(axis=0)
        z_score = 1.96  # [95%]
        upper_bound = S_model + z_score * S_model_std
        lower_bound = S_model - z_score * S_model_std

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_km, y=S_km, mode='lines',name=f"KM overall ({K} events)", line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=t_eval, y=S_model, mode="lines", name=f"Model overall ({K} events)"))
        fig.add_trace(go.Scatter(
            x=t_eval, y=upper_bound, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 200, 0.2)',
            showlegend=False, name='95%CI'
        ))
        fig.add_trace(go.Scatter(
            x=t_eval, y=lower_bound, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 100, 200, 0.2)',
            showlegend=False,
        ))
        fig.update_layout(
                    title=f"Kaplan-Meier vs Model Overall Survival with 95% CI",
                    xaxis_title="t",
                    yaxis_title="S(t)",
                    yaxis_range=[0, 1],
                    hovermode='x unified'
                )
        fig.show()

        total_results.append(results)

    #########################################


    ctds_list = [[result[f"ctd-index-k{i}"] for result in total_results] for i in range(1, 8)]
    for a in range(len(ctds_list)):
        print("####################### Event ", a + 1)
        print(f"{np.mean(ctds_list[a]).round(3)} +/- {np.std(ctds_list[a]).round(3)}")
