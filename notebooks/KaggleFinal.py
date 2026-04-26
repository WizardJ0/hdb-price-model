"""
FINAL Model - Clean structure: feature engineering once, seed loop only fits models
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
test  = pd.read_csv('../data/test.csv', low_memory=False)
test_ids = test['id'].copy()
print(f"Train: {train.shape}  |  Test: {test.shape}")

# ── Drop redundant columns ──────────────────────────────────────────
drop_cols = ['residential', 'floor_area_sqft', 'storey_range', 'lower', 'upper', 'mid',
             'full_flat_type', 'address', 'Tranc_YearMonth', 'block', 'postal',
             'bus_stop_name', 'street_name']
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

# ── Missing values ──────────────────────────────────────────────────
amenity_cols = ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
                'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']
for col in amenity_cols:
    train[col] = train[col].fillna(0)
    test[col]  = test[col].fillna(0)
train['Mall_Nearest_Distance'] = train['Mall_Nearest_Distance'].fillna(train['Mall_Nearest_Distance'].median())
test['Mall_Nearest_Distance']  = test['Mall_Nearest_Distance'].fillna(test['Mall_Nearest_Distance'].median())

# ── Boolean columns ─────────────────────────────────────────────────
bool_cols = ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion',
             'bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']
for col in bool_cols:
    train[col] = train[col].map({'Y': 1, 'N': 0})
    test[col]  = test[col].map({'Y': 1, 'N': 0})

# ── Feature engineering ─────────────────────────────────────────────
current_year = 2024
for df in [train, test]:
    df['market_regime']       = df['Tranc_Year'].apply(lambda x: 0 if x <= 2013 else (1 if x <= 2016 else (2 if x <= 2018 else 3)))
    df['building_age']        = current_year - df['year_completed']
    df['lease_remaining']     = 99 - (current_year - df['lease_commence_date'])
    df['floor_area_per_room'] = df['floor_area_sqm'] / (df['total_dwelling_units'] + 1)
    df['Tranc_Quarter']       = ((df['Tranc_Month'] - 1) // 3) + 1
    df['age_lease_interaction']  = df['building_age'] * df['lease_remaining']
    df['floor_area_price_proxy'] = df['floor_area_sqm'] * df['total_dwelling_units']
    df['amenity_score']          = (df['Mall_Within_1km'] + df['Hawker_Within_1km']) / (df['Mall_Nearest_Distance'] + 1)
    df['price_per_sqm_proxy']    = df['floor_area_sqm'] / (df['total_dwelling_units'] + 1)
    df['storey_ratio']           = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan).fillna(1)

# ── K-fold target encoding (computed once) ──────────────────────────
y = train['resale_price'].copy()
y_log = np.log1p(y)

def kfold_target_encode(train_data, test_data, col, target, n_splits=5, smoothing=5.0):
    global_mean = train_data[target].mean()
    encoded_train = np.zeros(len(train_data))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_data):
        fold_tr = train_data.iloc[tr_idx]
        agg = fold_tr.groupby(col)[target].agg(['mean', 'count'])
        agg['smoothed'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)
        encoded_train[val_idx] = train_data.iloc[val_idx][col].map(agg['smoothed']).fillna(global_mean).values
    agg_full = train_data.groupby(col)[target].agg(['mean', 'count'])
    agg_full['smoothed'] = (agg_full['mean'] * agg_full['count'] + global_mean * smoothing) / (agg_full['count'] + smoothing)
    encoded_test = test_data[col].map(agg_full['smoothed']).fillna(global_mean).values
    train_data[f'{col}_encoded'] = encoded_train
    test_data[f'{col}_encoded']  = encoded_test
    train_data.drop(columns=[col], inplace=True, errors='ignore')
    test_data.drop(columns=[col], inplace=True, errors='ignore')
    return train_data, test_data

for col in ['planning_area', 'mrt_name', 'pri_sch_name', 'sec_sch_name', 'town', 'flat_model']:
    if col in train.columns:
        print(f"  K-fold encoding: {col}")
        train, test = kfold_target_encode(train, test, col, 'resale_price')

# ── One-hot encode remaining categoricals ───────────────────────────
X = train.drop('resale_price', axis=1)
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
X      = pd.get_dummies(X,    columns=cat_cols, drop_first=True)
test_X = pd.get_dummies(test, columns=cat_cols, drop_first=True)

for col in set(X.columns) - set(test_X.columns):
    test_X[col] = 0
test_X = test_X[X.columns]

print(f"Features: {X.shape[1]}  |  Train rows: {len(X)}")

# ── Seed ensemble ───────────────────────────────────────────────────
seeds = [42, 123, 456, 789, 999, 2024, 31, 77, 314, 1337, 555, 888, 2001, 7777, 9999]
all_predictions = []

for seed_idx, seed in enumerate(seeds):
    print(f"\n🌱 Training with seed {seed} ({seed_idx+1}/{len(seeds)})")

    lgb_model = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.02, num_leaves=127, max_depth=10,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.08, reg_lambda=0.08,
        min_child_samples=15, random_state=seed, verbosity=-1, n_jobs=-1
    )
    lgb_model.fit(X, y_log)
    lgb_pred = lgb_model.predict(test_X)

    xgb_model = xgb.XGBRegressor(
        n_estimators=3000, learning_rate=0.02, max_depth=7, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.08, reg_lambda=0.08, min_child_weight=7,
        random_state=seed, verbosity=0, n_jobs=-1
    )
    xgb_model.fit(X, y_log)
    xgb_pred = xgb_model.predict(test_X)

    cb_model = cb.CatBoostRegressor(
        iterations=3000, learning_rate=0.02, depth=8, l2_leaf_reg=1.5,
        subsample=0.85, random_seed=seed, verbose=False
    )
    cb_model.fit(X, y_log)
    cb_pred = cb_model.predict(test_X)

    y_pred_log = lgb_pred * 0.35 + xgb_pred * 0.25 + cb_pred * 0.40
    all_predictions.append(y_pred_log)

# ── Average and submit ──────────────────────────────────────────────
print(f"\n🎯 Averaging predictions from {len(seeds)} seeds...")
final_pred_log = np.mean(all_predictions, axis=0)
y_pred = np.expm1(final_pred_log)
y_pred = np.clip(y_pred, 150000, 1300000)

sample = pd.read_csv('../data/sample_sub_reg.csv')
sample_ids = set(sample['Id'])
submission = pd.DataFrame({'Id': test_ids, 'Predicted': y_pred.round().astype(int)})
submission = submission[submission['Id'].isin(sample_ids)]

print(f"Submission: {submission.shape}")
print(f"Prediction range: {submission['Predicted'].min():,} - {submission['Predicted'].max():,}")
submission.to_csv('../submission_final.csv', index=False)
print("✅ SAVED → ../submission_final.csv")
