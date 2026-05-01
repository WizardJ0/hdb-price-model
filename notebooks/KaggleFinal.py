"""
KaggleFinal.py — Combined pipeline: Kaggle.py ensemble depth + notebook v15 features & encodings
Run from the notebooks/ directory: python -u KaggleFinal.py

Architecture:
  Level-0: LightGBM (CPU, native cats) + XGBoost (GPU, ordinal) + CatBoost (GPU, string cats)
  Level-1: RidgeCV meta-learner with OOF preds + disagreement features + key raw features
  Outer:   20-seed x 10-fold averaging

Key improvements over Kaggle.py:
  - Native categorical handling per model — no TE+OHE pipeline
  - 130+ features vs 93 (CPF eligibility, school quality tiers, postal sector,
    unit-mix ratios, 7 geographic centres, school/MRT lat-lon distances, etc.)
  - RidgeCV (auto alpha-tuned) + disagreement features + raw features in meta-learner
  - Deduplication of training data
  - Early stopping patience 100 (vs 50)

Reference scores:
  Kaggle.py best  : 21,488.50
  Notebook v15    : 21,382.41
  Blend 60/40     : 21,313.46  <- current best
  1st place       : 21,225.31
"""

import sys, warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
warnings.filterwarnings('ignore')

# Tee stdout → console + run.log
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()

_logfile = open('run.log', 'w', buffering=1)
sys.stdout = _Tee(sys.__stdout__, _logfile)

# ── Config ────────────────────────────────────────────────────────────
TEST_MODE    = False  # True = 1 seed (~30 min) | False = 5 seeds full run
N_FOLDS      = 10
ALL_SEEDS    = [42, 123, 456, 789, 999]
N_SEEDS      = 1 if TEST_MODE else len(ALL_SEEDS)
N_CLUSTERS   = 30
CURRENT_YEAR = 2024

# 7 geographic reference centres (Euclidean proxy distances)
COORDS = {
    'cbd':        (1.2835, 103.8510),
    'orchard':    (1.3048, 103.8318),
    'marina_bay': (1.2800, 103.8590),
    'jurong_e':   (1.3329, 103.7436),
    'tampines':   (1.3540, 103.9440),
    'woodlands':  (1.4382, 103.7891),
    'serangoon':  (1.3496, 103.8729),
}
CBD_LAT, CBD_LON         = COORDS['cbd']
ORCHARD_LAT, ORCHARD_LON = COORDS['orchard']
JURONG_LAT, JURONG_LON   = COORDS['jurong_e']

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

# Categorical columns — native for LGB/CB, ordinal-encoded for XGB
CAT_COLS = [
    'town', 'flat_type', 'street_name', 'storey_range', 'flat_model', 'full_flat_type',
    'planning_area', 'mrt_name', 'pri_sch_name', 'sec_sch_name',
    'postal_sector', 'lease_bucket', 'mrt_walk_cat', 'sec_sch_tier',
    'dist_nearest_centre_name', 'zone',
]

# Raw numeric features passed into the meta-learner alongside OOF predictions
META_RAW_COLS = [
    'floor_area_sqm', 'mid_storey', 'lease_at_tranc', 'dist_to_cbd',
    'tranc_time_idx', 'pri_sch_nearest_distance', 'mrt_nearest_distance',
    'cutoff_point', 'pri_sch_dist_cbd',
]

# ── Load & deduplicate ────────────────────────────────────────────────
print('Loading data...')
train = pd.read_csv('../data/train.csv', low_memory=False)
test  = pd.read_csv('../data/test.csv',  low_memory=False)
test_ids = test['id'].copy()
print(f'Train: {train.shape}  |  Test: {test.shape}')

n_before = len(train)
train = train.drop_duplicates(
    subset=[c for c in train.columns if c != 'id']).reset_index(drop=True)
print(f'Dedup: {n_before} -> {len(train)} rows')

# ── Missing values ────────────────────────────────────────────────────
for col in ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
            'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']:
    train[col] = train[col].fillna(0)
    test[col]  = test[col].fillna(0)

mall_max = max(train['Mall_Nearest_Distance'].max(), test['Mall_Nearest_Distance'].max())
for df in (train, test):
    df['Mall_Nearest_Distance'] = df['Mall_Nearest_Distance'].fillna(mall_max)

for col in ['Hawker_Nearest_Distance', 'mrt_nearest_distance', 'pri_sch_nearest_distance',
            'sec_sch_nearest_dist', 'bus_stop_nearest_distance', 'cutoff_point', 'vacancy']:
    if col in train.columns:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col]  = test[col].fillna(med)

# ── Boolean encoding ──────────────────────────────────────────────────
bool_cols = ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion',
             'bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']
for col in bool_cols:
    for df in (train, test):
        df[col] = (df[col].astype(str).str.upper()
                   .map({'Y': 1, 'N': 0, '1': 1, '0': 0, 'TRUE': 1, 'FALSE': 0})
                   .fillna(0).astype(np.int8))

# ── Feature engineering ───────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

for df in (train, test):
    lat, lon = df['Latitude'], df['Longitude']
    yr,  mo  = df['Tranc_Year'], df['Tranc_Month']

    # ── Location ──────────────────────────────────────────────────────
    df['zone']          = df['town'].map(ZONE_MAP).fillna('Unknown')
    df['postal_sector'] = df['postal'].astype(str).str.zfill(6).str[:2]

    # Haversine distances to 3 key centres (accurate km)
    df['dist_to_cbd']     = haversine_km(lat, lon, CBD_LAT,     CBD_LON)
    df['dist_to_orchard'] = haversine_km(lat, lon, ORCHARD_LAT, ORCHARD_LON)
    df['dist_to_jurong']  = haversine_km(lat, lon, JURONG_LAT,  JURONG_LON)

    # Euclidean distances to 7 centres (matches notebook v15)
    centre_dist_cols = []
    for name, (clat, clon) in COORDS.items():
        cname = f'dist_{name}'
        df[cname] = np.sqrt((lat - clat) ** 2 + (lon - clon) ** 2)
        centre_dist_cols.append(cname)
    df['dist_nearest_centre']      = df[centre_dist_cols].min(axis=1)
    df['dist_nearest_centre_name'] = (df[centre_dist_cols].idxmin(axis=1)
                                      .str.replace('dist_', '', regex=False))

    # School & MRT coordinate distances to CBD
    df['mrt_dist_cbd']     = np.sqrt((df['mrt_latitude']    - CBD_LAT) ** 2 + (df['mrt_longitude']    - CBD_LON) ** 2)
    df['pri_sch_dist_cbd'] = np.sqrt((df['pri_sch_latitude'] - CBD_LAT) ** 2 + (df['pri_sch_longitude'] - CBD_LON) ** 2)
    df['sec_sch_dist_cbd'] = np.sqrt((df['sec_sch_latitude'] - CBD_LAT) ** 2 + (df['sec_sch_longitude'] - CBD_LON) ** 2)

    # ── Time ──────────────────────────────────────────────────────────
    df['tranc_time_idx'] = (yr - 2000) * 12 + mo
    df['month_sin']      = np.sin(2 * np.pi * mo / 12)
    df['month_cos']      = np.cos(2 * np.pi * mo / 12)
    df['Tranc_Quarter']  = ((mo - 1) // 3) + 1
    df['market_regime']  = yr.apply(
        lambda x: 0 if x <= 2013 else (1 if x <= 2016 else (2 if x <= 2018 else 3)))

    # ── Lease / age ───────────────────────────────────────────────────
    df['age_at_tranc']          = yr - df['year_completed']       # building age at sale
    df['age_at_sale']           = yr - df['lease_commence_date']  # lease age at sale
    df['lease_at_tranc']        = 99 - df['age_at_sale']
    df['lease_at_tranc_sq']     = df['lease_at_tranc'] ** 2
    df['age_x_lease_tranc']     = df['age_at_tranc'] * df['lease_at_tranc']
    df['building_age']          = CURRENT_YEAR - df['year_completed']
    df['lease_remaining']       = 99 - (CURRENT_YEAR - df['lease_commence_date'])
    df['lease_remaining_sq']    = df['lease_remaining'] ** 2
    df['age_lease_interaction'] = df['building_age'] * df['lease_remaining']
    # CPF financing requires >= 60 years remaining — hard price cliff
    df['cpf_eligible']    = (df['lease_at_tranc'] >= 60).astype(np.int8)
    df['lease_above_cpf'] = np.maximum(0, df['lease_at_tranc'] - 60)
    df['lease_bucket']    = pd.cut(
        df['lease_at_tranc'], bins=[0, 50, 60, 70, 80, 90, 99],
        labels=['<50', '50-60', '60-70', '70-80', '80-90', '>90']).astype(str)

    # ── Flat / storey ─────────────────────────────────────────────────
    df['flat_type_num'] = df['flat_type'].map(FLAT_TYPE_ORDER).fillna(4).astype(int)
    df['storey_ratio']  = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan).fillna(1)
    df['storey_rel']    = df['mid_storey'] / (df['max_floor_lvl'] + 1)
    df['high_floor']    = (df['mid_storey'] >= 20).astype(np.int8)

    # ── Floor area ────────────────────────────────────────────────────
    total_units = df['total_dwelling_units'].replace(0, 1)
    df['floor_area_per_room']    = df['floor_area_sqm'] / total_units
    df['floor_area_price_proxy'] = df['floor_area_sqm'] * df['total_dwelling_units']
    df['price_per_sqm_proxy']    = df['floor_area_sqm'] / total_units

    # ── MRT / transport ───────────────────────────────────────────────
    df['mrt_accessibility']  = 1.0 / (df['mrt_nearest_distance'] + 1)
    df['log_mrt_dist']       = np.log1p(df['mrt_nearest_distance'])
    df['mrt_quality_score']  = (df['mrt_interchange'] * 2 + df['bus_interchange']).astype(np.int8)
    df['mrt_walk_cat']       = pd.cut(
        df['mrt_nearest_distance'], bins=[0, 300, 500, 1000, np.inf],
        labels=['<300m', '300-500m', '500m-1km', '>1km']).astype(str)
    df['effective_mrt_dist'] = df['mrt_nearest_distance'] / (1 + df['mrt_interchange'] * 0.5)
    df['transport_score']    = (df['mrt_interchange'] * 3 + df['bus_interchange'] +
                                (df['mrt_nearest_distance'] < 500).astype(int))

    # ── Schools ───────────────────────────────────────────────────────
    df['log_pri_dist']         = np.log1p(df['pri_sch_nearest_distance'])
    df['log_sec_dist']         = np.log1p(df['sec_sch_nearest_dist'])
    df['sec_sch_tier']         = pd.cut(
        df['cutoff_point'], bins=[0, 200, 215, 230, 260],
        labels=['standard', 'good', 'very_good', 'elite']).astype(str)
    df['vacancy_score']        = 1.0 / (df['vacancy'] + 1)
    df['pri_popular']          = (df['vacancy'] < 40).astype(np.int8)
    df['pri_quality_dist']     = df['pri_sch_affiliation'] / (df['pri_sch_nearest_distance'] + 1) * 1000
    df['sec_quality_dist']     = df['cutoff_point'] / (df['sec_sch_nearest_dist'] + 1) * 1000
    df['pri_within_1km']       = (df['pri_sch_nearest_distance'] < 1000).astype(np.int8)
    df['pri_within_1km_aff']   = (df['pri_within_1km'] & df['pri_sch_affiliation']).astype(np.int8)
    df['sec_within_1km']       = (df['sec_sch_nearest_dist'] < 1000).astype(np.int8)
    df['sec_elite_within_2km'] = ((df['sec_sch_nearest_dist'] < 2000) &
                                  (df['cutoff_point'] >= 230)).astype(np.int8)

    # ── Amenities ─────────────────────────────────────────────────────
    df['log_mall_dist']   = np.log1p(df['Mall_Nearest_Distance'])
    df['log_hawker_dist'] = np.log1p(df['Hawker_Nearest_Distance'])
    df['hawker_quality']  = df['hawker_food_stalls'] / (df['Hawker_Nearest_Distance'] + 1)
    df['hawker_density']  = df['Hawker_Within_500m'] * 3 + df['Hawker_Within_1km']
    df['mall_density']    = df['Mall_Within_500m'] * 3 + df['Mall_Within_1km'] * 2 + df['Mall_Within_2km']
    df['amenity_score']   = df['mall_density'] + df['hawker_density']

    # ── Unit mix ──────────────────────────────────────────────────────
    sold_cols = ['1room_sold', '2room_sold', '3room_sold', '4room_sold',
                 '5room_sold', 'exec_sold', 'multigen_sold', 'studio_apartment_sold']
    df['total_units_sold'] = df[[c for c in sold_cols if c in df.columns]].sum(axis=1)
    df['exec_ratio']       = df['exec_sold'] / (df['total_units_sold'] + 1)
    df['pct_3room']        = df['3room_sold']  / total_units
    df['pct_4room']        = df['4room_sold']  / total_units
    df['pct_5room']        = df['5room_sold']  / total_units
    df['pct_exec']         = df['exec_sold']   / total_units
    df['pct_rental']       = (df['1room_rental'] + df['2room_rental'] +
                              df['3room_rental'] + df['other_room_rental']) / total_units
    df['pct_premium']      = (df['5room_sold'] + df['exec_sold'] + df['multigen_sold']) / total_units

    # ── Interactions ──────────────────────────────────────────────────
    df['floor_x_storey']    = df['floor_area_sqm'] * df['mid_storey']
    df['dist_cbd_x_storey'] = df['dist_to_cbd'] * df['storey_ratio']
    df['area_x_age']        = df['floor_area_sqm'] * df['age_at_tranc']
    df['mrt_x_interchange'] = df['mrt_nearest_distance'] * (1 - df['mrt_interchange'] * 0.4)
    df['lease_x_storey']    = df['lease_at_tranc'] * df['storey_rel']

    # ── New leak-free features ─────────────────────────────────────────
    df['log_floor_area']    = np.log1p(df['floor_area_sqm'])
    df['storey_cbd_ratio']  = df['mid_storey'] / (df['dist_to_cbd'] + 0.1)   # high floor near CBD
    df['lease_x_area']      = df['lease_at_tranc'] * df['floor_area_sqm']    # lease-weighted area

# ── Geo clustering ────────────────────────────────────────────────────
_geo_km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
train['geo_cluster'] = _geo_km.fit_predict(train[['Latitude', 'Longitude']])
test['geo_cluster']  = _geo_km.predict(test[['Latitude', 'Longitude']])

# ── Drop raw columns no longer needed as features ─────────────────────
DROP_COLS = [
    'id', 'residential', 'floor_area_sqft', 'lower', 'upper', 'mid',
    'address', 'Tranc_YearMonth', 'bus_stop_name', 'block', 'postal',
]
train.drop(columns=DROP_COLS, inplace=True, errors='ignore')
test.drop(columns=DROP_COLS,  inplace=True, errors='ignore')

# ── Target & feature split ────────────────────────────────────────────
y_raw        = train['resale_price'].values.astype(float)
y            = np.log1p(y_raw)
feature_cols = [c for c in train.columns if c != 'resale_price']

CAT_PRESENT = [c for c in CAT_COLS if c in feature_cols]
NUM_PRESENT  = [c for c in feature_cols if c not in CAT_PRESENT]

print(f'Features: {len(feature_cols)} ({len(NUM_PRESENT)} numeric + {len(CAT_PRESENT)} categorical)')

sample     = pd.read_csv('../data/sample_sub_reg.csv')
sample_ids = set(sample['Id'])

# ── Model-specific feature matrices (prepared once outside seed loop) ──

# LGB — pandas category dtype, CPU (LGB GPU does not support native categoricals)
print('Building LGB matrix (native cats, CPU)...')
X_lgb      = train[feature_cols].copy()
X_lgb_test = test[feature_cols].copy()
for c in CAT_PRESENT:
    X_lgb[c]      = X_lgb[c].astype(str).astype('category')
    X_lgb_test[c] = X_lgb_test[c].astype(str).astype('category')
    unified        = pd.api.types.union_categoricals([X_lgb[c], X_lgb_test[c]]).categories
    X_lgb[c]      = X_lgb[c].cat.set_categories(unified)
    X_lgb_test[c] = X_lgb_test[c].cat.set_categories(unified)
for c in NUM_PRESENT:
    X_lgb[c]      = pd.to_numeric(X_lgb[c], errors='coerce')
    X_lgb_test[c] = pd.to_numeric(X_lgb_test[c], errors='coerce')

# XGB — ordinal-encoded categoricals, median-filled numerics, GPU
print('Building XGB matrix (ordinal enc, GPU)...')
_enc       = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_xgb      = train[feature_cols].copy()
X_xgb_test = test[feature_cols].copy()
for c in NUM_PRESENT:
    X_xgb[c]      = pd.to_numeric(X_xgb[c], errors='coerce')
    X_xgb_test[c] = pd.to_numeric(X_xgb_test[c], errors='coerce')
    med            = X_xgb[c].median()
    X_xgb[c]      = X_xgb[c].fillna(med)
    X_xgb_test[c] = X_xgb_test[c].fillna(med)
X_xgb[CAT_PRESENT]      = _enc.fit_transform(X_xgb[CAT_PRESENT].astype(str))
X_xgb_test[CAT_PRESENT] = _enc.transform(X_xgb_test[CAT_PRESENT].astype(str))

# CB — string categoricals, median-filled numerics, GPU
print('Building CB matrix (string cats, GPU)...')
X_cb      = train[feature_cols].copy()
X_cb_test = test[feature_cols].copy()
for c in CAT_PRESENT:
    X_cb[c]      = X_cb[c].astype(str)
    X_cb_test[c] = X_cb_test[c].astype(str)
for c in NUM_PRESENT:
    X_cb[c]      = pd.to_numeric(X_cb[c], errors='coerce')
    X_cb_test[c] = pd.to_numeric(X_cb_test[c], errors='coerce')
    med           = X_cb[c].median()
    X_cb[c]       = X_cb[c].fillna(med)
    X_cb_test[c]  = X_cb_test[c].fillna(med)

# Meta-learner raw features (extracted once, median-filled)
META_PRESENT = [c for c in META_RAW_COLS if c in feature_cols]

def _get_meta_raw(src_df, ref_df=None):
    cols = []
    for f in META_PRESENT:
        vals = pd.to_numeric(src_df[f], errors='coerce')
        med  = (pd.to_numeric(ref_df[f], errors='coerce') if ref_df is not None else vals).median()
        cols.append(vals.fillna(med).values)
    return np.column_stack(cols) if cols else np.zeros((len(src_df), 1))

meta_raw_tr = _get_meta_raw(train)
meta_raw_te = _get_meta_raw(test, ref_df=train)
print(f'Meta raw features: {META_PRESENT}')

# ── One seed pass ─────────────────────────────────────────────────────
def run_seed(seed):
    m_oof  = {k: np.zeros(len(train)) for k in ('lgb', 'xgb', 'cb')}
    m_test = {k: np.zeros(len(test))  for k in ('lgb', 'xgb', 'cb')}

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for _, (tr_idx, va_idx) in enumerate(kf.split(X_lgb)):
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # LightGBM — CPU, native categoricals (GPU does not support native cats)
        lgb_m = lgb.LGBMRegressor(
            n_estimators=8000, learning_rate=0.03, num_leaves=127,
            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
            min_data_in_leaf=30, lambda_l2=1.0, lambda_l1=0.1,
            max_cat_threshold=64, random_state=seed, verbosity=-1, device='cpu',
        )
        lgb_m.fit(
            X_lgb.iloc[tr_idx], y_tr,
            eval_set=[(X_lgb.iloc[va_idx], y_va)],
            categorical_feature='auto',
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )
        m_oof['lgb'][va_idx] = lgb_m.predict(X_lgb.iloc[va_idx])
        m_test['lgb']       += lgb_m.predict(X_lgb_test) / N_FOLDS

        # XGBoost — GPU, ordinal-encoded
        xgb_m = xgb.XGBRegressor(
            n_estimators=6000, learning_rate=0.01, max_depth=7,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.08, reg_lambda=0.08, min_child_weight=7,
            device='cuda', tree_method='hist',
            early_stopping_rounds=100, eval_metric='rmse',
            random_state=seed, verbosity=0,
        )
        xgb_m.fit(X_xgb.iloc[tr_idx], y_tr,
                  eval_set=[(X_xgb.iloc[va_idx], y_va)], verbose=False)
        m_oof['xgb'][va_idx] = xgb_m.predict(X_xgb.iloc[va_idx])
        m_test['xgb']       += xgb_m.predict(X_xgb_test) / N_FOLDS

        # CatBoost — GPU, string categoricals (native ordered TE internally)
        cb_m = cb.CatBoostRegressor(
            iterations=6000, learning_rate=0.01, depth=10, l2_leaf_reg=1.5,
            subsample=0.85, bootstrap_type='Bernoulli',
            task_type='GPU', random_seed=seed, verbose=False,
            early_stopping_rounds=100,
        )
        cb_m.fit(
            X_cb.iloc[tr_idx], y_tr,
            cat_features=CAT_PRESENT,
            eval_set=(X_cb.iloc[va_idx], y_va),
            use_best_model=True,
        )
        m_oof['cb'][va_idx] = cb_m.predict(X_cb.iloc[va_idx])
        m_test['cb']       += cb_m.predict(X_cb_test) / N_FOLDS

    # RidgeCV meta-learner: OOF preds + disagreement features + raw features
    dis_lx_tr = m_oof['lgb']  - m_oof['xgb'];  dis_lx_te = m_test['lgb']  - m_test['xgb']
    dis_lc_tr = m_oof['lgb']  - m_oof['cb'];   dis_lc_te = m_test['lgb']  - m_test['cb']
    dis_xc_tr = m_oof['xgb']  - m_oof['cb'];   dis_xc_te = m_test['xgb']  - m_test['cb']

    S_tr = np.column_stack([m_oof['lgb'],  m_oof['xgb'],  m_oof['cb'],
                            dis_lx_tr, dis_lc_tr, dis_xc_tr, meta_raw_tr])
    S_te = np.column_stack([m_test['lgb'], m_test['xgb'], m_test['cb'],
                            dis_lx_te, dis_lc_te, dis_xc_te, meta_raw_te])

    scaler  = StandardScaler()
    S_tr_sc = scaler.fit_transform(S_tr)
    S_te_sc = scaler.transform(S_te)

    meta = RidgeCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], cv=5)
    meta.fit(S_tr_sc, y)
    return meta.predict(S_tr_sc), meta.predict(S_te_sc)


# ── Main loop ─────────────────────────────────────────────────────────
oof_sum  = np.zeros(len(train))
test_sum = np.zeros(len(test))

print(f"\n{'='*70}")
print(f"  {N_FOLDS} folds x {N_SEEDS} seeds x 3 models = {N_FOLDS * N_SEEDS * 3:,} fits")
print(f"  Reference: blend 60/40 = 21,313.46  |  1st place = 21,225.31")
print(f"{'='*70}\n")

for i, seed in enumerate(ALL_SEEDS[:N_SEEDS]):
    print(f"  Seed {seed} ({i+1}/{N_SEEDS})")
    s_oof, s_test = run_seed(seed)
    oof_sum  += s_oof
    test_sum += s_test

    if (i + 1) % 5 == 0:
        n    = i + 1
        rmse = np.sqrt(np.mean((np.expm1(oof_sum / n) - y_raw) ** 2))
        print(f"  *** s={n}  OOF RMSE: {rmse:,.2f}")

# ── Final ─────────────────────────────────────────────────────────────
final_oof  = np.expm1(oof_sum  / N_SEEDS)
final_test = np.expm1(test_sum / N_SEEDS)
oof_rmse   = np.sqrt(np.mean((final_oof - y_raw) ** 2))

print(f"\n{'='*70}")
print(f"FINAL OOF RMSE ({N_SEEDS}s x {N_FOLDS}f): {oof_rmse:,.2f}")
print(f"{'='*70}")

final_test = np.clip(final_test, 150000, 1300000)
sub = pd.DataFrame({'Id': test_ids, 'Predicted': final_test.round().astype(int)})
sub = sub[sub['Id'].isin(sample_ids)]
sub.to_csv('../submission_kaggle_final.csv', index=False)
print(f"Saved -> ../submission_kaggle_final.csv  |  rows: {sub.shape[0]}")
print(f"Prediction range: {sub['Predicted'].min():,} - {sub['Predicted'].max():,}")
