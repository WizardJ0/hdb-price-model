"""
10 folds × 20 seeds × 3 models = 600 fits
Improvements over KaggleGridSearch best (21,615.74 Kaggle / 21,467.43 OOF):
  - Zone (geographic region from town)
  - Ordinal flat_type_num
  - Haversine distances to CBD, Orchard, Jurong (km)
  - Multi-column target encoding: (town,flat_type), (planning_area,flat_model),
    (town,flat_model), (Tranc_Year,town), (Tranc_Year,flat_type)
  - Log-transformed distance features
  - Interaction features: floor_x_storey, dist_cbd_x_storey
  - Lease decay: lease_remaining^2
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

# ── Config ───────────────────────────────────────────────────────────
N_FOLDS   = 10
ALL_SEEDS = [42, 123, 456, 789, 999, 2024, 31, 77, 314, 1337,
             555, 888, 2001, 7777, 9999, 11, 22, 33, 44, 55]
N_SEEDS   = len(ALL_SEEDS)

# ── Load data ────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv('../data/train.csv', low_memory=False)
test  = pd.read_csv('../data/test.csv',  low_memory=False)
test_ids = test['id'].copy()
print(f"Train: {train.shape}  |  Test: {test.shape}")

# ── Drop redundant columns ───────────────────────────────────────────
drop_cols = [
    'id', 'residential', 'floor_area_sqft', 'storey_range', 'lower', 'upper', 'mid',
    'full_flat_type', 'address', 'Tranc_YearMonth', 'block', 'postal',
    'bus_stop_name', 'street_name',
]
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols,  inplace=True, errors='ignore')

# ── Missing values ───────────────────────────────────────────────────
for col in ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
            'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']:
    train[col] = train[col].fillna(0)
    test[col]  = test[col].fillna(0)

for col in ['Mall_Nearest_Distance', 'Hawker_Nearest_Distance',
            'mrt_nearest_distance', 'pri_sch_nearest_distance',
            'sec_sch_nearest_dist', 'bus_stop_nearest_distance', 'cutoff_point']:
    if col in train.columns:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col]  = test[col].fillna(med)

# ── Boolean encoding ─────────────────────────────────────────────────
for col in ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion',
            'bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']:
    train[col] = train[col].map({'Y': 1, 'N': 0})
    test[col]  = test[col].map({'Y': 1, 'N': 0})

# ── Feature engineering ──────────────────────────────────────────────
CURRENT_YEAR = 2024
CBD_LAT,     CBD_LON     = 1.2794, 103.8501   # Raffles Place
ORCHARD_LAT, ORCHARD_LON = 1.3048, 103.8318   # Orchard Road
JURONG_LAT,  JURONG_LON  = 1.3329, 103.7436   # Jurong East

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

ZONE_MAP = {
    'WOODLANDS': 'North',      'YISHUN': 'North',         'SEMBAWANG': 'North',
    'ANG MO KIO': 'NorthEast', 'HOUGANG': 'NorthEast',    'PUNGGOL': 'NorthEast',
    'SENGKANG': 'NorthEast',   'SERANGOON': 'NorthEast',
    'BEDOK': 'East',           'PASIR RIS': 'East',       'TAMPINES': 'East',
    'BUKIT BATOK': 'West',     'BUKIT PANJANG': 'West',   'CHOA CHU KANG': 'West',
    'CLEMENTI': 'West',        'JURONG EAST': 'West',     'JURONG WEST': 'West',
    'BUKIT TIMAH': 'West',
    'BISHAN': 'Central',       'BUKIT MERAH': 'Central',  'CENTRAL AREA': 'Central',
    'GEYLANG': 'Central',      'KALLANG/WHAMPOA': 'Central', 'MARINE PARADE': 'Central',
    'QUEENSTOWN': 'Central',   'TOA PAYOH': 'Central',
}
FLAT_TYPE_ORDER = {
    '1 ROOM': 1, '2 ROOM': 2, '3 ROOM': 3, '4 ROOM': 4,
    '5 ROOM': 5, 'EXECUTIVE': 6, 'MULTI-GENERATION': 7,
}

for df in [train, test]:
    df['zone']               = df['town'].map(ZONE_MAP).fillna('Unknown')
    df['flat_type_num']      = df['flat_type'].map(FLAT_TYPE_ORDER).fillna(4).astype(int)
    df['market_regime']      = df['Tranc_Year'].apply(
        lambda x: 0 if x <= 2013 else (1 if x <= 2016 else (2 if x <= 2018 else 3)))
    df['building_age']       = CURRENT_YEAR - df['year_completed']
    df['lease_remaining']    = 99 - (CURRENT_YEAR - df['lease_commence_date'])
    df['lease_remaining_sq'] = df['lease_remaining'] ** 2
    df['Tranc_Quarter']      = ((df['Tranc_Month'] - 1) // 3) + 1
    df['floor_area_per_room']    = df['floor_area_sqm'] / (df['total_dwelling_units'] + 1)
    df['age_lease_interaction']  = df['building_age'] * df['lease_remaining']
    df['floor_area_price_proxy'] = df['floor_area_sqm'] * df['total_dwelling_units']
    df['amenity_score']      = (df['Mall_Within_1km'] + df['Hawker_Within_1km']) / (df['Mall_Nearest_Distance'] + 1)
    df['price_per_sqm_proxy'] = df['floor_area_sqm'] / (df['total_dwelling_units'] + 1)
    df['storey_ratio']       = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan).fillna(1)
    df['dist_to_cbd']        = haversine_km(df['Latitude'], df['Longitude'], CBD_LAT,     CBD_LON)
    df['dist_to_orchard']    = haversine_km(df['Latitude'], df['Longitude'], ORCHARD_LAT, ORCHARD_LON)
    df['dist_to_jurong']     = haversine_km(df['Latitude'], df['Longitude'], JURONG_LAT,  JURONG_LON)
    df['mrt_accessibility']  = 1.0 / (df['mrt_nearest_distance'] + 1)
    df['log_mrt_dist']       = np.log1p(df['mrt_nearest_distance'])
    df['log_pri_dist']       = np.log1p(df['pri_sch_nearest_distance'])
    df['log_sec_dist']       = np.log1p(df['sec_sch_nearest_dist'])
    df['log_mall_dist']      = np.log1p(df['Mall_Nearest_Distance'])
    df['log_hawker_dist']    = np.log1p(df['Hawker_Nearest_Distance'])
    df['floor_x_storey']     = df['floor_area_sqm'] * df['mid_storey']
    df['dist_cbd_x_storey']  = df['dist_to_cbd'] * df['storey_ratio']

y        = train['resale_price'].reset_index(drop=True)
y_log    = np.log1p(y)
X_raw    = train.drop('resale_price', axis=1).reset_index(drop=True)
test_raw = test.reset_index(drop=True)

# Single-column target encode cols (high-cardinality strings)
TE_SINGLE = [c for c in ['planning_area', 'mrt_name', 'pri_sch_name', 'sec_sch_name', 'town', 'flat_model']
             if c in X_raw.columns]

# Multi-column combos — computed per fold, originals still TE'd/OHE'd separately
TE_MULTI = [
    ['town', 'flat_type'],
    ['planning_area', 'flat_model'],
    ['town', 'flat_model'],
    ['Tranc_Year', 'town'],
    ['Tranc_Year', 'flat_type'],
]

sample     = pd.read_csv('../data/sample_sub_reg.csv')
sample_ids = set(sample['Id'])

print(f"Features after engineering: {X_raw.shape[1]}")
print(f"TE single: {TE_SINGLE}")
print(f"TE multi:  {['+'.join(c) for c in TE_MULTI]}")


# ── Target encoding helper ───────────────────────────────────────────
def smoothed_te(tr_keys, tr_targets, apply_keys, global_mean, smoothing=5.0):
    stats = (pd.DataFrame({'k': tr_keys, 't': tr_targets})
               .groupby('k')['t'].agg(['mean', 'count']))
    stats['enc'] = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
    mapping = stats['enc'].to_dict()
    return np.array([mapping.get(v, global_mean) for v in apply_keys], dtype=np.float64)


def make_key(df, cols):
    return df[cols].fillna('NA').astype(str).agg('__'.join, axis=1).values


# ── One seed pass ────────────────────────────────────────────────────
def run_seed(seed):
    seed_oof  = np.zeros(len(X_raw))
    seed_test = np.zeros(len(test_raw))

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_raw)):
        ftr_X      = X_raw.iloc[tr_idx].reset_index(drop=True)
        fval_X     = X_raw.iloc[val_idx].reset_index(drop=True)
        fte_X      = test_raw.copy()
        ftr_y      = y.iloc[tr_idx].reset_index(drop=True)
        ftr_y_log  = y_log.iloc[tr_idx].reset_index(drop=True)
        fval_y_log = y_log.iloc[val_idx].reset_index(drop=True)
        gm         = ftr_y.mean()

        # Multi-column TE — must run before single-col TE drops the source columns
        for cols in TE_MULTI:
            if not all(c in ftr_X.columns for c in cols):
                continue
            enc = '__'.join(cols) + '_enc'
            k_tr  = make_key(ftr_X,  cols)
            k_val = make_key(fval_X, cols)
            k_te  = make_key(fte_X,  cols)
            ftr_X[enc]  = smoothed_te(k_tr, ftr_y.values, k_tr,  gm)
            fval_X[enc] = smoothed_te(k_tr, ftr_y.values, k_val, gm)
            fte_X[enc]  = smoothed_te(k_tr, ftr_y.values, k_te,  gm)

        # Single-column TE — drops original column after encoding
        for col in TE_SINGLE:
            enc = f'{col}_enc'
            ftr_X[enc]  = smoothed_te(ftr_X[col].values, ftr_y.values, ftr_X[col].values,  gm)
            fval_X[enc] = smoothed_te(ftr_X[col].values, ftr_y.values, fval_X[col].values, gm)
            fte_X[enc]  = smoothed_te(ftr_X[col].values, ftr_y.values, fte_X[col].values,  gm)
            for df in [ftr_X, fval_X, fte_X]:
                df.drop(columns=[col], inplace=True)

        # OHE remaining categoricals, aligned to fold-train schema
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
            min_child_samples=15, random_state=seed, verbosity=-1, n_jobs=-1,
        )
        lgb_m.fit(ftr_X, ftr_y_log,
                  eval_set=[(fval_X, fval_y_log)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

        xgb_m = xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.03, max_depth=7, subsample=0.85,
            colsample_bytree=0.85, reg_alpha=0.08, reg_lambda=0.08, min_child_weight=7,
            random_state=seed, verbosity=0, n_jobs=-1,
            early_stopping_rounds=50, eval_metric='rmse',
        )
        xgb_m.fit(ftr_X, ftr_y_log, eval_set=[(fval_X, fval_y_log)], verbose=False)

        cb_m = cb.CatBoostRegressor(
            iterations=3000, learning_rate=0.03, depth=8, l2_leaf_reg=1.5,
            subsample=0.85, random_seed=seed, verbose=False, early_stopping_rounds=50,
        )
        cb_m.fit(ftr_X, ftr_y_log, eval_set=(fval_X, fval_y_log))

        lgb_oof  = lgb_m.predict(fval_X);  lgb_test = lgb_m.predict(fte_X)
        xgb_oof  = xgb_m.predict(fval_X);  xgb_test = xgb_m.predict(fte_X)
        cb_oof   = cb_m.predict(fval_X);   cb_test  = cb_m.predict(fte_X)

        seed_oof[val_idx] = lgb_oof * 0.35 + xgb_oof * 0.25 + cb_oof * 0.40
        seed_test        += (lgb_test * 0.35 + xgb_test * 0.25 + cb_test * 0.40) / N_FOLDS

    return seed_oof, seed_test


# ── Main loop ─────────────────────────────────────────────────────────
oof_sum  = np.zeros(len(X_raw))
test_sum = np.zeros(len(test_raw))

print(f"\n{'='*70}")
print(f"  {N_FOLDS} folds × {N_SEEDS} seeds × 3 models = {N_FOLDS * N_SEEDS * 3:,} fits")
print(f"  Baseline: Kaggle 21,615.74  |  OOF 21,467.43")
print(f"{'='*70}\n")

for i, seed in enumerate(ALL_SEEDS):
    print(f"  Seed {seed} ({i+1}/{N_SEEDS})")
    s_oof, s_test = run_seed(seed)
    oof_sum  += s_oof
    test_sum += s_test

    if (i + 1) % 5 == 0:
        n    = i + 1
        rmse = np.sqrt(np.mean((np.expm1(oof_sum / n) - y) ** 2))
        print(f"  *** s={n}  OOF RMSE: {rmse:,.2f}")

# ── Final ─────────────────────────────────────────────────────────────
final_oof  = np.expm1(oof_sum  / N_SEEDS)
final_test = np.expm1(test_sum / N_SEEDS)
oof_rmse   = np.sqrt(np.mean((final_oof - y) ** 2))

print(f"\n{'='*70}")
print(f"FINAL OOF RMSE ({N_SEEDS}s × {N_FOLDS}f): {oof_rmse:,.2f}")
print(f"OOF delta vs baseline:       {oof_rmse - 21467.43:+,.2f}")
print(f"{'='*70}")

final_test = np.clip(final_test, 150000, 1300000)
sub = pd.DataFrame({'Id': test_ids, 'Predicted': final_test.round().astype(int)})
sub = sub[sub['Id'].isin(sample_ids)]
sub.to_csv('../submission_kaggle.csv', index=False)
print(f"Saved → ../submission_kaggle.csv  |  rows: {sub.shape[0]}")
print(f"Prediction range: {sub['Predicted'].min():,} – {sub['Predicted'].max():,}")
