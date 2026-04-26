"""
OOF Ensemble — 20 seeds × 10 folds × 3 models = 600 fits
Target encoding computed inside each fold (zero leakage).
Early stopping uses the OOF validation split to find optimal tree count per fold.
Added: Singapore zone feature (North/NorthEast/East/West/Central) derived from town.
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

print("Loading data...")
train = pd.read_csv('../data/train.csv', low_memory=False)
test  = pd.read_csv('../data/test.csv',  low_memory=False)
test_ids = test['id'].copy()
print(f"Train: {train.shape}  |  Test: {test.shape}")

# ── Drop redundant columns ─────────────────────────────────────────
drop_cols = ['residential', 'floor_area_sqft', 'storey_range', 'lower', 'upper', 'mid',
             'full_flat_type', 'address', 'Tranc_YearMonth', 'block', 'postal',
             'bus_stop_name', 'street_name']
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols,  inplace=True, errors='ignore')

# ── Missing values ─────────────────────────────────────────────────
amenity_cols = ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
                'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']
for col in amenity_cols:
    train[col] = train[col].fillna(0)
    test[col]  = test[col].fillna(0)
train['Mall_Nearest_Distance'] = train['Mall_Nearest_Distance'].fillna(train['Mall_Nearest_Distance'].median())
test['Mall_Nearest_Distance']  = test['Mall_Nearest_Distance'].fillna(test['Mall_Nearest_Distance'].median())

# ── Boolean encoding ───────────────────────────────────────────────
bool_cols = ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion',
             'bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']
for col in bool_cols:
    train[col] = train[col].map({'Y': 1, 'N': 0})
    test[col]  = test[col].map({'Y': 1, 'N': 0})

# ── Feature engineering ────────────────────────────────────────────
current_year = 2024
CBD_LAT, CBD_LON = 1.2794, 103.8501  # Raffles Place

zone_map = {
    'WOODLANDS': 'North', 'YISHUN': 'North', 'SEMBAWANG': 'North',
    'ANG MO KIO': 'NorthEast', 'HOUGANG': 'NorthEast', 'PUNGGOL': 'NorthEast',
    'SENGKANG': 'NorthEast', 'SERANGOON': 'NorthEast',
    'BEDOK': 'East', 'PASIR RIS': 'East', 'TAMPINES': 'East',
    'BUKIT BATOK': 'West', 'BUKIT PANJANG': 'West', 'CHOA CHU KANG': 'West',
    'CLEMENTI': 'West', 'JURONG EAST': 'West', 'JURONG WEST': 'West', 'BUKIT TIMAH': 'West',
    'BISHAN': 'Central', 'BUKIT MERAH': 'Central', 'CENTRAL AREA': 'Central',
    'GEYLANG': 'Central', 'KALLANG/WHAMPOA': 'Central', 'MARINE PARADE': 'Central',
    'QUEENSTOWN': 'Central', 'TOA PAYOH': 'Central',
}

for df in [train, test]:
    df['zone']                   = df['town'].map(zone_map).fillna('Unknown')
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

# ── Targets and raw feature matrix ────────────────────────────────
y      = train['resale_price'].reset_index(drop=True)
y_log  = np.log1p(y)
X_raw  = train.drop('resale_price', axis=1).reset_index(drop=True)
test_raw = test.reset_index(drop=True)

te_cols = [c for c in ['planning_area', 'mrt_name', 'pri_sch_name', 'sec_sch_name', 'town', 'flat_model']
           if c in X_raw.columns]
print(f"\nRaw features: {X_raw.shape[1]}  |  Target-encode cols: {te_cols}")

# ── OOF loop ───────────────────────────────────────────────────────
N_FOLDS = 10
seeds   = [42, 123, 456, 789, 999, 2024, 31, 77, 314, 1337, 555, 888, 2001, 7777, 9999,
           11, 22, 33, 44, 55]

oof_sum  = np.zeros(len(X_raw))
test_sum = np.zeros(len(test_raw))

for seed_idx, seed in enumerate(seeds):
    seed_oof  = np.zeros(len(X_raw))
    seed_test = np.zeros(len(test_raw))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_raw)):
        print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)})  Fold {fold+1}/{N_FOLDS}")

        # Slice fold data with fresh 0-based indices to avoid alignment bugs
        ftr_X = X_raw.iloc[tr_idx].reset_index(drop=True)
        fval_X = X_raw.iloc[val_idx].reset_index(drop=True)
        fte_X  = test_raw.copy()
        ftr_y      = y.iloc[tr_idx].reset_index(drop=True)
        ftr_y_log  = y_log.iloc[tr_idx].reset_index(drop=True)
        fval_y_log = y_log.iloc[val_idx].reset_index(drop=True)
        global_mean = ftr_y.mean()

        # Per-fold target encoding — stats derived only from fold-train
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
            enc_name = f'{col}_enc'
            for df in [ftr_X, fval_X, fte_X]:
                df[enc_name] = df[col].map(stats['smoothed']).fillna(global_mean)
                df.drop(columns=[col], inplace=True)

        # One-hot encode remaining categoricals — aligned to fold-train schema
        cat_cols = ftr_X.select_dtypes(include=['object']).columns.tolist()
        ftr_X  = pd.get_dummies(ftr_X,  columns=cat_cols, drop_first=True)
        fval_X = pd.get_dummies(fval_X, columns=cat_cols, drop_first=True)
        fte_X  = pd.get_dummies(fte_X,  columns=cat_cols, drop_first=True)

        for df in [fval_X, fte_X]:
            for col in set(ftr_X.columns) - set(df.columns):
                df[col] = 0
        fval_X = fval_X[ftr_X.columns]
        fte_X  = fte_X[ftr_X.columns]

        # ── LightGBM ──────────────────────────────────────────────
        lgb_m = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.03, num_leaves=127, max_depth=10,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.08, reg_lambda=0.08,
            min_child_samples=15, random_state=seed, verbosity=-1, n_jobs=-1
        )
        lgb_m.fit(
            ftr_X, ftr_y_log,
            eval_set=[(fval_X, fval_y_log)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )

        # ── XGBoost ───────────────────────────────────────────────
        xgb_m = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.03, max_depth=7, subsample=0.85,
            colsample_bytree=0.85, reg_alpha=0.08, reg_lambda=0.08, min_child_weight=7,
            random_state=seed, verbosity=0, n_jobs=-1,
            early_stopping_rounds=50, eval_metric='rmse'
        )
        xgb_m.fit(
            ftr_X, ftr_y_log,
            eval_set=[(fval_X, fval_y_log)],
            verbose=False
        )

        # ── CatBoost ──────────────────────────────────────────────
        cb_m = cb.CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=1.5,
            subsample=0.85, random_seed=seed, verbose=False,
            early_stopping_rounds=50
        )
        cb_m.fit(
            ftr_X, ftr_y_log,
            eval_set=(fval_X, fval_y_log)
        )

        # ── Blend predictions ─────────────────────────────────────
        lgb_oof  = lgb_m.predict(fval_X)
        xgb_oof  = xgb_m.predict(fval_X)
        cb_oof   = cb_m.predict(fval_X)

        lgb_test = lgb_m.predict(fte_X)
        xgb_test = xgb_m.predict(fte_X)
        cb_test  = cb_m.predict(fte_X)

        fold_oof  = lgb_oof  * 0.35 + xgb_oof  * 0.25 + cb_oof  * 0.40
        fold_test = lgb_test * 0.35 + xgb_test * 0.25 + cb_test * 0.40

        seed_oof[val_idx] = fold_oof
        seed_test        += fold_test / N_FOLDS

    seed_rmse = np.sqrt(np.mean((np.expm1(seed_oof) - y) ** 2))
    print(f"  → Seed {seed} OOF RMSE: {seed_rmse:,.2f}")

    oof_sum  += seed_oof
    test_sum += seed_test

# ── Average and evaluate ───────────────────────────────────────────
final_oof_log  = oof_sum  / len(seeds)
final_test_log = test_sum / len(seeds)

final_oof  = np.expm1(final_oof_log)
final_test = np.expm1(final_test_log)

oof_rmse = np.sqrt(np.mean((final_oof - y) ** 2))
print(f"\n{'='*60}")
print(f"FINAL OOF RMSE (all {len(seeds)} seeds × {N_FOLDS} folds): {oof_rmse:,.2f}")
print(f"{'='*60}")

final_test = np.clip(final_test, 150000, 1300000)

# ── Submission ─────────────────────────────────────────────────────
sample    = pd.read_csv('../data/sample_sub_reg.csv')
sample_ids = set(sample['Id'])
submission = pd.DataFrame({'Id': test_ids, 'Predicted': final_test.round().astype(int)})
submission = submission[submission['Id'].isin(sample_ids)]

print(f"Submission: {submission.shape}")
print(f"Prediction range: {submission['Predicted'].min():,} - {submission['Predicted'].max():,}")
submission.to_csv('../submission_oof_zone.csv', index=False)
print("Saved → ../submission_oof_zone.csv")
