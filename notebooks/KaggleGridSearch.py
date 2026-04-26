"""
Systematic grid search: n_folds ∈ {5, 7, 10}  ×  n_seeds ∈ {15, 20, 25}
= 9 configurations evaluated from 3 sequential runs (one per fold count).
Predictions accumulate across seeds so each checkpoint costs nothing extra.

Known baseline: folds=5, seeds=15 → OOF 21,510.65 / Kaggle 21,633.99
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')

# ── Config ─────────────────────────────────────────────────────────
N_FOLDS_LIST      = [5, 7, 10]
SEED_CHECKPOINTS  = [15, 20, 25]

# Original 15 seeds first (preserves comparability), then 10 new seeds
ALL_SEEDS = [
    42, 123, 456, 789, 999, 2024, 31, 77, 314, 1337, 555, 888, 2001, 7777, 9999,  # 15
    11, 22, 33, 44, 55, 66, 100, 200, 300, 400                                      # +10
]
assert len(ALL_SEEDS) == 25 and len(set(ALL_SEEDS)) == 25, "Duplicate seeds!"

# ── Load & preprocess (done once) ──────────────────────────────────
print("Loading data...")
train = pd.read_csv('../data/train.csv', low_memory=False)
test  = pd.read_csv('../data/test.csv',  low_memory=False)
test_ids = test['id'].copy()
print(f"Train: {train.shape}  |  Test: {test.shape}")

drop_cols = ['residential', 'floor_area_sqft', 'storey_range', 'lower', 'upper', 'mid',
             'full_flat_type', 'address', 'Tranc_YearMonth', 'block', 'postal',
             'bus_stop_name', 'street_name']
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols,  inplace=True, errors='ignore')

amenity_cols = ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
                'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']
for col in amenity_cols:
    train[col] = train[col].fillna(0)
    test[col]  = test[col].fillna(0)
train['Mall_Nearest_Distance'] = train['Mall_Nearest_Distance'].fillna(train['Mall_Nearest_Distance'].median())
test['Mall_Nearest_Distance']  = test['Mall_Nearest_Distance'].fillna(test['Mall_Nearest_Distance'].median())

bool_cols = ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion',
             'bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']
for col in bool_cols:
    train[col] = train[col].map({'Y': 1, 'N': 0})
    test[col]  = test[col].map({'Y': 1, 'N': 0})

current_year = 2024
CBD_LAT, CBD_LON = 1.2794, 103.8501
for df in [train, test]:
    df['market_regime']          = df['Tranc_Year'].apply(
        lambda x: 0 if x <= 2013 else (1 if x <= 2016 else (2 if x <= 2018 else 3)))
    df['building_age']           = current_year - df['year_completed']
    df['lease_remaining']        = 99 - (current_year - df['lease_commence_date'])
    df['floor_area_per_room']    = df['floor_area_sqm'] / (df['total_dwelling_units'] + 1)
    df['Tranc_Quarter']          = ((df['Tranc_Month'] - 1) // 3) + 1
    df['age_lease_interaction']  = df['building_age'] * df['lease_remaining']
    df['floor_area_price_proxy'] = df['floor_area_sqm'] * df['total_dwelling_units']
    df['amenity_score']          = (df['Mall_Within_1km'] + df['Hawker_Within_1km']) / (df['Mall_Nearest_Distance'] + 1)
    df['price_per_sqm_proxy']    = df['floor_area_sqm'] / (df['total_dwelling_units'] + 1)
    df['storey_ratio']           = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan).fillna(1)
    df['dist_to_cbd']            = np.sqrt((df['Latitude'] - CBD_LAT)**2 + (df['Longitude'] - CBD_LON)**2)
    df['mrt_accessibility']      = 1.0 / (df['mrt_nearest_distance'] + 1)

y        = train['resale_price'].reset_index(drop=True)
y_log    = np.log1p(y)
X_raw    = train.drop('resale_price', axis=1).reset_index(drop=True)
test_raw = test.reset_index(drop=True)

te_cols = [c for c in ['planning_area', 'mrt_name', 'pri_sch_name', 'sec_sch_name', 'town', 'flat_model']
           if c in X_raw.columns]

sample     = pd.read_csv('../data/sample_sub_reg.csv')
sample_ids = set(sample['Id'])

# ── Helper: one OOF seed pass ───────────────────────────────────────
def run_seed(seed, n_folds, X_raw, test_raw, y, y_log, te_cols):
    """Returns (seed_oof_log [N_train], seed_test_log [N_test])."""
    seed_oof  = np.zeros(len(X_raw))
    seed_test = np.zeros(len(test_raw))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_raw)):
        ftr_X      = X_raw.iloc[tr_idx].reset_index(drop=True)
        fval_X     = X_raw.iloc[val_idx].reset_index(drop=True)
        fte_X      = test_raw.copy()
        ftr_y      = y.iloc[tr_idx].reset_index(drop=True)
        ftr_y_log  = y_log.iloc[tr_idx].reset_index(drop=True)
        fval_y_log = y_log.iloc[val_idx].reset_index(drop=True)
        global_mean = ftr_y.mean()

        for col in te_cols:
            smoothing = 5.0
            stats = (
                pd.DataFrame({'col': ftr_X[col].values, 'target': ftr_y.values})
                .groupby('col')['target']
                .agg(['mean', 'count'])
            )
            stats['smoothed'] = (
                (stats['mean'] * stats['count'] + global_mean * smoothing)
                / (stats['count'] + smoothing)
            )
            enc = f'{col}_enc'
            for df in [ftr_X, fval_X, fte_X]:
                df[enc] = df[col].map(stats['smoothed']).fillna(global_mean)
                df.drop(columns=[col], inplace=True)

        cat_cols = ftr_X.select_dtypes(include=['object']).columns.tolist()
        ftr_X  = pd.get_dummies(ftr_X,  columns=cat_cols, drop_first=True)
        fval_X = pd.get_dummies(fval_X, columns=cat_cols, drop_first=True)
        fte_X  = pd.get_dummies(fte_X,  columns=cat_cols, drop_first=True)
        for df in [fval_X, fte_X]:
            for col in set(ftr_X.columns) - set(df.columns):
                df[col] = 0
        fval_X = fval_X[ftr_X.columns]
        fte_X  = fte_X[ftr_X.columns]

        lgb_m = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.03, num_leaves=127, max_depth=10,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.08, reg_lambda=0.08,
            min_child_samples=15, random_state=seed, verbosity=-1, n_jobs=-1
        )
        lgb_m.fit(ftr_X, ftr_y_log,
                  eval_set=[(fval_X, fval_y_log)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

        xgb_m = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.03, max_depth=7, subsample=0.85,
            colsample_bytree=0.85, reg_alpha=0.08, reg_lambda=0.08, min_child_weight=7,
            random_state=seed, verbosity=0, n_jobs=-1,
            early_stopping_rounds=50, eval_metric='rmse'
        )
        xgb_m.fit(ftr_X, ftr_y_log, eval_set=[(fval_X, fval_y_log)], verbose=False)

        cb_m = cb.CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=1.5,
            subsample=0.85, random_seed=seed, verbose=False, early_stopping_rounds=50
        )
        cb_m.fit(ftr_X, ftr_y_log, eval_set=(fval_X, fval_y_log))

        lgb_oof  = lgb_m.predict(fval_X);  lgb_test = lgb_m.predict(fte_X)
        xgb_oof  = xgb_m.predict(fval_X);  xgb_test = xgb_m.predict(fte_X)
        cb_oof   = cb_m.predict(fval_X);   cb_test  = cb_m.predict(fte_X)

        seed_oof[val_idx] = lgb_oof * 0.35 + xgb_oof * 0.25 + cb_oof * 0.40
        seed_test        += (lgb_test * 0.35 + xgb_test * 0.25 + cb_test * 0.40) / n_folds

    return seed_oof, seed_test


def oof_rmse(oof_log_sum, n_seeds, y):
    return np.sqrt(np.mean((np.expm1(oof_log_sum / n_seeds) - y) ** 2))


def save_submission(test_log_sum, n_seeds, test_ids, sample_ids, tag):
    preds = np.clip(np.expm1(test_log_sum / n_seeds), 150000, 1300000)
    sub   = pd.DataFrame({'Id': test_ids, 'Predicted': preds.round().astype(int)})
    sub   = sub[sub['Id'].isin(sample_ids)]
    path  = f'../submission_{tag}.csv'
    sub.to_csv(path, index=False)
    return path


# ── Grid search ─────────────────────────────────────────────────────
results = {}  # (n_folds, n_seeds) → OOF RMSE

print(f"\n{'='*70}")
print(f"GRID SEARCH: folds={N_FOLDS_LIST}  ×  seeds={SEED_CHECKPOINTS}")
print(f"Total model fits: {sum(f*25*3 for f in N_FOLDS_LIST):,}")
print(f"{'='*70}\n")

for n_folds in N_FOLDS_LIST:
    print(f"\n{'─'*70}")
    print(f"  n_folds = {n_folds}  ({n_folds*25*3} model fits)")
    print(f"{'─'*70}")

    oof_cumsum  = np.zeros(len(X_raw))
    test_cumsum = np.zeros(len(test_raw))

    for seed_idx, seed in enumerate(ALL_SEEDS):
        n = seed_idx + 1
        print(f"    Seed {seed} ({n}/{len(ALL_SEEDS)})  [folds={n_folds}]")
        s_oof, s_test = run_seed(seed, n_folds, X_raw, test_raw, y, y_log, te_cols)
        oof_cumsum  += s_oof
        test_cumsum += s_test

        if n in SEED_CHECKPOINTS:
            rmse = oof_rmse(oof_cumsum, n, y)
            tag  = f'grid_f{n_folds}_s{n}'
            path = save_submission(test_cumsum, n, test_ids, sample_ids, tag)
            results[(n_folds, n)] = rmse
            print(f"    *** checkpoint f={n_folds} s={n} → OOF RMSE: {rmse:,.2f}  →  {path}")

# ── Summary table ────────────────────────────────────────────────────
print(f"\n\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")
print(f"{'':>18}", end='')
for s in SEED_CHECKPOINTS:
    print(f"  seeds={s:>2}", end='')
print()
print(f"  {'─'*60}")
for f in N_FOLDS_LIST:
    print(f"  folds={f:>2}         ", end='')
    for s in SEED_CHECKPOINTS:
        rmse = results.get((f, s), float('nan'))
        print(f"  {rmse:>9,.2f}", end='')
    print()

best_config = min(results, key=results.get)
best_rmse   = results[best_config]
print(f"\n  Best: folds={best_config[0]}, seeds={best_config[1]}  →  OOF RMSE {best_rmse:,.2f}")
print(f"  Submission: submission_grid_f{best_config[0]}_s{best_config[1]}.csv")
print(f"{'='*70}")
