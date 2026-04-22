import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import StackingRegressor
import optuna
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA & INITIAL SETUP
# ═══════════════════════════════════════════════════════════════════
print("Loading data...")
train = pd.read_csv('../data/train.csv', low_memory=False)
test  = pd.read_csv('../data/test.csv', low_memory=False)
print(f"Train: {train.shape}  |  Test: {test.shape}")

# ═══════════════════════════════════════════════════════════════════
# 2. LOAD AND PROCESS NEW DATA SOURCES (2014 & 2019)
# ═══════════════════════════════════════════════════════════════════
print("\nLoading new data sources (2014 & 2019)...")

def load_income_data(filepath):
    """Load and normalize income data by planning area"""
    df = pd.read_csv(filepath)
    df_copy = df.copy()
    
    # Identify the first column (area names)
    area_col = df_copy.columns[0]
    income_col = df_copy.columns[1]
    
    # Skip the "Total" row
    df_area = df_copy[df_copy[area_col] != 'Total'].copy()
    df_area[income_col] = pd.to_numeric(df_area[income_col], errors='coerce')
    
    # Extract income distribution columns (income brackets)
    income_brackets = df_copy.columns[3:]
    for col in income_brackets:
        df_area[col] = pd.to_numeric(df_area[col], errors='coerce')
    
    return df_area.set_index(area_col)

def load_transport_data(filepath):
    """Load and normalize transport mode data by planning area"""
    df = pd.read_csv(filepath)
    df_copy = df.copy()
    
    area_col = df_copy.columns[0]
    total_col = df_copy.columns[1]
    
    df_area = df_copy[df_copy[area_col] != 'Total'].copy()
    df_area[total_col] = pd.to_numeric(df_area[total_col], errors='coerce')
    
    transport_cols = df_copy.columns[2:]
    for col in transport_cols:
        df_area[col] = pd.to_numeric(df_area[col], errors='coerce')
    
    return df_area.set_index(area_col)

# Load 2014 data
income_2014 = load_income_data('../geojson/2014/ResidentHouseholdsbyPlanningAreaandMonthlyHouseholdIncomefromWorkGeneralHouseholdSurvey2015.csv')
transport_2014 = load_transport_data('../geojson/2014/ResidentWorkingPersonsAged15YearsandOverbyPlanningAreaandUsualModeofTransporttoWorkGeneralHouseholdSurvey2015.csv')

# Load 2019 data
income_2019 = load_income_data('../geojson/2019/ResidentHouseholdsbyPlanningAreaofResidenceandMonthlyHouseholdIncomefromWorkCensusOfPopulation2020.csv')
employed_2019 = load_transport_data('../geojson/2019/EmployedResidentsAged15YearsandOverbyPlanningAreaofWorkplaceandUsualModeofTransporttoWorkCensusofPopulation2020.csv')

print(f"✓ 2014 Income: {income_2014.shape} | Transport: {transport_2014.shape}")
print(f"✓ 2019 Income: {income_2019.shape} | Employed Transport: {employed_2019.shape}")

def create_demographic_features(df, income_df, transport_df, year_label):
    """Create demographic and commuting features from income and transport data"""
    df = df.copy()
    
    # Merge income data with forward/backward fill for missing areas
    for col in income_df.columns:
        feature_name = f'{year_label}_income_{col.lower()}'
        df[feature_name] = df['planning_area'].map(income_df[col])
        # Fill missing areas with planning_area median
        df[feature_name] = df[feature_name].fillna(df.groupby('planning_area')[feature_name].transform('mean'))
    
    # For completely missing areas, use global mean
    income_cols = [c for c in df.columns if year_label in c and 'income' in c]
    for col in income_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    # Calculate income statistics
    income_brackets = [col for col in income_df.columns if col not in ['Total', 'NoWorkingPerson', 'NoEmployedPerson']]
    if income_brackets and 'Total' in income_df.columns:
        df[f'{year_label}_high_income_pct'] = df['planning_area'].map(
            (income_df[income_brackets[-1]] / income_df['Total'] * 100).fillna(0)
        )
        df[f'{year_label}_high_income_pct'] = df[f'{year_label}_high_income_pct'].fillna(df[f'{year_label}_high_income_pct'].mean())
    
    # Merge transport data
    for col in transport_df.columns:
        feature_name = f'{year_label}_transport_{col.lower()}'
        df[feature_name] = df['planning_area'].map(transport_df[col])
        df[feature_name] = df[feature_name].fillna(df[feature_name].mean())
    
    # Calculate commuting pattern ratios
    if 'Total' in transport_df.columns:
        public_cols = ['PublicBusOnly', 'MRTOnly', 'MRT_LRTOnly']
        public_sum = sum([transport_df[c] if c in transport_df.columns else 0 for c in public_cols])
        df[f'{year_label}_public_transport_pct'] = df['planning_area'].map(
            (public_sum / transport_df['Total'] * 100).fillna(0)
        )
        df[f'{year_label}_public_transport_pct'] = df[f'{year_label}_public_transport_pct'].fillna(df[f'{year_label}_public_transport_pct'].mean())
        
        if 'CarOnly' in transport_df.columns:
            df[f'{year_label}_car_usage_pct'] = df['planning_area'].map(
                (transport_df['CarOnly'] / transport_df['Total'] * 100).fillna(0)
            )
            df[f'{year_label}_car_usage_pct'] = df[f'{year_label}_car_usage_pct'].fillna(df[f'{year_label}_car_usage_pct'].mean())
    
    return df

# Apply demographic features
train = create_demographic_features(train, income_2014, transport_2014, 'y2014')
test = create_demographic_features(test, income_2014, transport_2014, 'y2014')

train = create_demographic_features(train, income_2019, employed_2019, 'y2019')
test = create_demographic_features(test, income_2019, employed_2019, 'y2019')

demographic_features = [col for col in train.columns if 'y2014' in col or 'y2019' in col]
print(f"✓ New demographic features created: {len(demographic_features)}")

# ═══════════════════════════════════════════════════════════════════
# 3. ADVANCED SPATIAL FEATURES
# ═══════════════════════════════════════════════════════════════════
print("\nBuilding advanced spatial features...")

def create_spatial_features(df):
    """Create advanced spatial features from coordinates"""
    df = df.copy()

    CBD_LAT, CBD_LON = 1.2794, 103.8501
    CHANGI_LAT, CHANGI_LON = 1.3644, 103.9915
    JURONG_LAT, JURONG_LON = 1.3329, 103.7436

    df['dist_to_cbd'] = np.sqrt((df['Latitude'] - CBD_LAT)**2 + (df['Longitude'] - CBD_LON)**2)
    df['dist_to_airport'] = np.sqrt((df['Latitude'] - CHANGI_LAT)**2 + (df['Longitude'] - CHANGI_LON)**2)
    df['dist_to_jurong'] = np.sqrt((df['Latitude'] - JURONG_LAT)**2 + (df['Longitude'] - JURONG_LON)**2)

    df['lat_rounded'] = df['Latitude'].round(3)
    df['lon_rounded'] = df['Longitude'].round(3)

    df['lat_bin'] = pd.cut(df['Latitude'], bins=20, labels=False)
    df['lon_bin'] = pd.cut(df['Longitude'], bins=20, labels=False)

    return df

train = create_spatial_features(train)
test  = create_spatial_features(test)

# Rail features - FIXED VERSION
print("\nComputing rail and station features...")

def compute_rail_distances(df_train, df_test, rail_2014_path, rail_2019_path, station_2025_path):
    """Fixed rail distance computation with proper error handling"""
    
    # Initialize columns
    rail_cols = ['dist_to_rail_2014', 'near_rail_2014', 'dist_to_rail_2019', 'near_rail_2019',
                 'dist_to_station_2025', 'near_station_2025', 'station_density_1km_2025']
    
    for col in rail_cols:
        df_train[col] = 0.0
        df_test[col] = 0.0
    
    try:
        # Load rail data
        print("  Loading rail GeoJSON files...")
        rail_2014 = gpd.read_file(rail_2014_path)
        rail_2019 = gpd.read_file(rail_2019_path)
        rail_2025 = gpd.read_file(station_2025_path)
        
        print(f"    2014 Rail: {len(rail_2014)} features")
        print(f"    2019 Rail: {len(rail_2019)} features")
        print(f"    2025 Stations: {len(rail_2025)} features")
        
        # Process 2014 rail for transactions <= 2016
        mask_2014 = df_train['Tranc_Year'] <= 2016
        if mask_2014.sum() > 0:
            print(f"  Processing 2014 rail for {mask_2014.sum()} transactions...")
            train_subset = df_train[mask_2014]
            points = gpd.GeoDataFrame(
                geometry=[Point(lon, lat) for lat, lon in zip(train_subset['Longitude'], train_subset['Latitude'])],
                crs='EPSG:4326',
                index=train_subset.index
            )
            rail_2014_wgs = rail_2014.to_crs('EPSG:4326')
            distances = points.geometry.apply(lambda pt: rail_2014_wgs.geometry.distance(pt).min() * 111)
            df_train.loc[mask_2014, 'dist_to_rail_2014'] = distances
            df_train.loc[mask_2014, 'near_rail_2014'] = (distances <= 1).astype(int)
            print(f"    ✓ 2014 rail: mean={distances.mean():.2f}km")
        
        # Process 2019 rail for transactions 2017-2022
        mask_2019 = (df_train['Tranc_Year'] >= 2017) & (df_train['Tranc_Year'] < 2023)
        if mask_2019.sum() > 0:
            print(f"  Processing 2019 rail for {mask_2019.sum()} transactions...")
            train_subset = df_train[mask_2019]
            points = gpd.GeoDataFrame(
                geometry=[Point(lon, lat) for lat, lon in zip(train_subset['Longitude'], train_subset['Latitude'])],
                crs='EPSG:4326',
                index=train_subset.index
            )
            rail_2019_wgs = rail_2019.to_crs('EPSG:4326')
            distances = points.geometry.apply(lambda pt: rail_2019_wgs.geometry.distance(pt).min() * 111)
            df_train.loc[mask_2019, 'dist_to_rail_2019'] = distances
            df_train.loc[mask_2019, 'near_rail_2019'] = (distances <= 1).astype(int)
            print(f"    ✓ 2019 rail: mean={distances.mean():.2f}km")
        
        # Process 2025 stations for all records in train and test
        print(f"  Processing 2025 rail stations...")
        rail_2025_wgs = rail_2025.to_crs('EPSG:4326')
        station_centroids = rail_2025_wgs.copy()
        station_centroids['geometry'] = station_centroids.geometry.centroid
        
        # Train set
        points_train = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in zip(df_train['Longitude'], df_train['Latitude'])],
            crs='EPSG:4326',
            index=df_train.index
        )
        distances_train = points_train.geometry.apply(lambda pt: station_centroids.geometry.distance(pt).min() * 111)
        df_train['dist_to_station_2025'] = distances_train
        df_train['near_station_2025'] = (distances_train <= 0.5).astype(int)
        
        # Station density
        station_density_train = []
        for pt in points_train.geometry:
            count = (station_centroids.geometry.distance(pt) * 111 <= 1.0).sum()
            station_density_train.append(count)
        df_train['station_density_1km_2025'] = station_density_train
        
        print(f"    ✓ 2025 stations: mean={distances_train.mean():.2f}km")
        
        # Test set
        points_test = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in zip(df_test['Longitude'], df_test['Latitude'])],
            crs='EPSG:4326',
            index=df_test.index
        )
        distances_test = points_test.geometry.apply(lambda pt: station_centroids.geometry.distance(pt).min() * 111)
        df_test['dist_to_station_2025'] = distances_test
        df_test['near_station_2025'] = (distances_test <= 0.5).astype(int)
        
        station_density_test = []
        for pt in points_test.geometry:
            count = (station_centroids.geometry.distance(pt) * 111 <= 1.0).sum()
            station_density_test.append(count)
        df_test['station_density_1km_2025'] = station_density_test
        
        print("✓ All rail features computed successfully")
        return df_train, df_test
        
    except Exception as e:
        print(f"⚠ Rail distance computation error: {e}")
        print("  Using default values (0)")
        return df_train, df_test

train, test = compute_rail_distances(train, test, 
                                      '../geojson/2014/MasterPlan2014RailLine.geojson',
                                      '../geojson/2019/AmendmenttoMasterPlan2019RailLinelayer.geojson',
                                      '../geojson/2025/MasterPlan2025RailStationLayer.geojson')

# ═══════════════════════════════════════════════════════════════════
# 4. ADVANCED FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
print("\nAdvanced feature engineering...")

def advanced_feature_engineering(df):
    df = df.copy()

    df['Tranc_Quarter'] = ((df['Tranc_Month'] - 1) // 3) + 1
    df['is_year_end'] = df['Tranc_Month'].isin([11, 12]).astype(int)
    df['is_q4'] = (df['Tranc_Quarter'] == 4).astype(int)

    conditions = [
        df['Tranc_Year'] <= 2013,
        (df['Tranc_Year'] >= 2014) & (df['Tranc_Year'] <= 2016),
        (df['Tranc_Year'] >= 2017) & (df['Tranc_Year'] <= 2018),
        (df['Tranc_Year'] >= 2019) & (df['Tranc_Year'] <= 2020),
        df['Tranc_Year'] >= 2021
    ]
    choices = ['peak', 'early_decline', 'trough', 'recovery', 'covid_surge']
    df['market_regime'] = np.select(conditions, choices, default='unknown')

    current_year = 2024
    df['years_remaining'] = 99 - (current_year - df['lease_commence_date'])
    df['building_age'] = current_year - df['year_completed']
    df['lease_pct_remaining'] = df['years_remaining'] / 99

    df['area_per_room'] = df['floor_area_sqm'] / (df['total_dwelling_units'] + 1)
    df['storey_ratio'] = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan)
    df['floor_area_x_storey'] = df['floor_area_sqm'] * df['mid_storey']

    df['mall_density_500m'] = df['Mall_Within_500m'] / (np.pi * 0.5**2)
    df['mall_density_1km'] = df['Mall_Within_1km'] / (np.pi * 1**2)
    df['hawker_density_500m'] = df['Hawker_Within_500m'] / (np.pi * 0.5**2)

    df['mrt_accessibility'] = 1 / (df['mrt_nearest_distance'] + 1)

    df['school_quality'] = pd.cut(df['cutoff_point'].fillna(df['cutoff_point'].median()),
                                  bins=[0, 200, 220, 240, 260, 300],
                                  labels=['low', 'below_avg', 'avg', 'above_avg', 'high'])

    if 'postal' in df.columns:
        df['postal_district'] = df['postal'].astype(str).str.zfill(6).str[:2].astype(int)
        df['is_central_district'] = df['postal_district'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(int)

    return df

train = advanced_feature_engineering(train)
test  = advanced_feature_engineering(test)

# ═══════════════════════════════════════════════════════════════════
# 5. OUTLIER DETECTION & HANDLING
# ═══════════════════════════════════════════════════════════════════
print("\nOutlier detection...")

price_q1, price_q3 = train['resale_price'].quantile([0.01, 0.99])
train = train[(train['resale_price'] >= price_q1) & (train['resale_price'] <= price_q3)]

area_q1, area_q3 = train['floor_area_sqm'].quantile([0.01, 0.99])
train = train[(train['floor_area_sqm'] >= area_q1) & (train['floor_area_sqm'] <= area_q3)]

print(f"After outlier removal: {train.shape}")

# ═══════════════════════════════════════════════════════════════════
# 6. MISSING VALUE HANDLING
# ═══════════════════════════════════════════════════════════════════
amenity_cols = [c for c in train.columns if 'Within_' in c or 'Nearest_' in c]
for col in amenity_cols:
    if col in train.columns:
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)

# ═══════════════════════════════════════════════════════════════════
# 7. DROP REDUNDANT COLUMNS
# ═══════════════════════════════════════════════════════════════════
drop_cols = [
    'residential', 'floor_area_sqft', 'storey_range', 'lower', 'upper', 'mid',
    'full_flat_type', 'address', 'Tranc_YearMonth', 'block', 'postal',
    'bus_stop_name', 'street_name', 'lease_commence_date', 'year_completed'
]

train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

# ═══════════════════════════════════════════════════════════════════
# 8. BOOLEAN ENCODING
# ═══════════════════════════════════════════════════════════════════
bool_cols = [c for c in train.columns if train[c].dtype == 'object' and set(train[c].dropna().unique()).issubset({'Y', 'N'})]
for col in bool_cols:
    train[col] = train[col].map({'Y': 1, 'N': 0}).fillna(0)
    test[col] = test[col].map({'Y': 1, 'N': 0}).fillna(0)

# ═══════════════════════════════════════════════════════════════════
# 9. ADVANCED TARGET ENCODING
# ═══════════════════════════════════════════════════════════════════
print("\nAdvanced target encoding...")

def advanced_target_encode(train, test, cat_cols, target, n_splits=5, smoothing=10):
    """Advanced target encoding with smoothing"""
    train_encoded = train.copy()
    test_encoded = test.copy()

    global_mean = train[target].mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for col in cat_cols:
        if col not in train.columns:
            continue

        train_encoded[f'{col}_te'] = 0.0

        for train_idx, val_idx in kf.split(train):
            fold_train = train.iloc[train_idx]
            fold_val = train.iloc[val_idx]

            agg = fold_train.groupby(col)[target].agg(['mean', 'count'])
            agg['smoothed'] = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)

            train_encoded.iloc[val_idx, train_encoded.columns.get_loc(f'{col}_te')] = fold_val[col].map(agg['smoothed']).fillna(global_mean)

        agg_full = train.groupby(col)[target].agg(['mean', 'count'])
        agg_full['smoothed'] = (agg_full['mean'] * agg_full['count'] + global_mean * smoothing) / (agg_full['count'] + smoothing)
        test_encoded[f'{col}_te'] = test[col].map(agg_full['smoothed']).fillna(global_mean)

        train_encoded.drop(columns=[col], inplace=True, errors='ignore')
        test_encoded.drop(columns=[col], inplace=True, errors='ignore')

    return train_encoded, test_encoded

high_card_cols = ['mrt_name', 'pri_sch_name', 'sec_sch_name', 'planning_area', 'school_quality']
high_card_cols = [c for c in high_card_cols if c in train.columns]

y_for_te = train['resale_price'].values
train, test = advanced_target_encode(train, test, high_card_cols, 'resale_price')

# ═══════════════════════════════════════════════════════════════════
# 10. PREPARE DATA FOR MODELING
# ═══════════════════════════════════════════════════════════════════
X = train.drop(columns=['resale_price'])
y = train['resale_price']
y_log = np.log1p(y)

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    test = pd.get_dummies(test, columns=cat_cols, drop_first=True)

    missing_cols = set(X.columns) - set(test.columns)
    for col in missing_cols:
        test[col] = 0
    test = test[X.columns]

print(f"Final data shapes: Train {X.shape}, Test {test.shape}")

# ═══════════════════════════════════════════════════════════════════
# 11. ENSEMBLE MODEL
# ═══════════════════════════════════════════════════════════════════
print("\nBuilding ensemble model...")

# LightGBM
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)

# XGBoost
xgb_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=0
)

# CatBoost
cb_model = cb.CatBoostRegressor(
    iterations=500, learning_rate=0.05, depth=6, random_state=42, verbose=0
)

# Ensemble
ensemble = [('lgb', lgb_model), ('xgb', xgb_model), ('cb', cb_model)]
voting_model = StackingRegressor(estimators=ensemble, final_estimator=Ridge(), cv=3)

print("\nCross-validating ensemble...")
cv_scores = cross_val_score(voting_model, X, y_log, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
cv_rmse = np.sqrt(-cv_scores.mean())
print(f"Cross-validation RMSE (log scale): {cv_rmse:.4f}")

print("\nTraining final ensemble...")
voting_model.fit(X, y_log)

print("\nGenerating predictions...")
y_pred_log = voting_model.predict(test)
y_pred = np.expm1(y_pred_log)

# Save submission
submission = pd.DataFrame({'Id': test.index, 'Predicted': y_pred})
submission.to_csv('../submission_fixed.csv', index=False)

print(f"\nSubmission shape: {submission.shape}")
print(f"Prediction range: {y_pred.min():,.0f} - {y_pred.max():,.0f}")
print(f"✓ Submission saved → ../submission_fixed.csv")
