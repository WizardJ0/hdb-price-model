import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import optuna
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════
print("Loading data...")
train = pd.read_csv('/Users/Josiah/Downloads/NTU_DASI4/Module3/Kaggle regression Challenge/hdb-price-model/data/train.csv')
test  = pd.read_csv('/Users/Josiah/Downloads/NTU_DASI4/Module3/Kaggle regression Challenge/hdb-price-model/data/test.csv')
print(f"Train: {train.shape}  |  Test: {test.shape}")


# ═══════════════════════════════════════════════════════════════════
# 2. GEOJSON SPATIAL JOIN — land use category & gross plot ratio
# ═══════════════════════════════════════════════════════════════════
print("\nRunning GeoJSON spatial join...")

def run_spatial_join(df, gdf):
    """Point-in-polygon join; returns (land_use array, gpr array) aligned to df."""
    lu_col  = next((c for c in gdf.columns if c.upper() == 'LU_DESC'), None)
    gpr_col = next((c for c in gdf.columns if c.upper() == 'GPR'),     None)

    keep = ['geometry']
    if lu_col:  keep.append(lu_col)
    if gpr_col: keep.append(gpr_col)

    points = gpd.GeoDataFrame(
        index=df.index,
        geometry=[Point(lon, lat) for lat, lon in zip(df['Latitude'], df['Longitude'])],
        crs='EPSG:4326'
    )
    gdf_wgs = gdf[keep].to_crs('EPSG:4326')
    joined  = gpd.sjoin(points, gdf_wgs, how='left', predicate='within')
    joined  = joined[~joined.index.duplicated(keep='first')]

    lu  = joined[lu_col].values  if lu_col  else np.full(len(df), 'UNKNOWN')
    gpr = (pd.to_numeric(joined[gpr_col], errors='coerce').values
           if gpr_col else np.full(len(df), np.nan))
    return lu, gpr

try:
    gdf_2014 = gpd.read_file('/Users/Josiah/Downloads/NTU_DASI4/Module3/Kaggle regression Challenge/hdb-price-model/geojson/MasterPlan2014LandUse.geojson')
    gdf_2019 = gpd.read_file('/Users/Josiah/Downloads/NTU_DASI4/Module3/Kaggle regression Challenge/hdb-price-model/geojson/AmendmenttoMasterPlan2019LandUselayer.geojson')
    gdf_2025 = gpd.read_file('/Users/Josiah/Downloads/NTU_DASI4/Module3/Kaggle regression Challenge/hdb-price-model/geojson/MasterPlan2025LandUseLayer.geojson')

    train['land_use'] = 'UNKNOWN'
    train['gpr']      = np.nan

    mask_14 = train['Tranc_Year'] <= 2016
    mask_19 = train['Tranc_Year'] >= 2017

    lu, gpr = run_spatial_join(train[mask_14], gdf_2014)
    train.loc[mask_14, 'land_use'] = lu
    train.loc[mask_14, 'gpr']      = gpr

    lu, gpr = run_spatial_join(train[mask_19], gdf_2019)
    train.loc[mask_19, 'land_use'] = lu
    train.loc[mask_19, 'gpr']      = gpr

    lu, gpr = run_spatial_join(test, gdf_2025)
    test['land_use'] = lu
    test['gpr']      = gpr

    print(f"GeoJSON join complete  |  land_use categories: {train['land_use'].nunique()}")
    HAS_GEOJSON = True

except Exception as e:
    print(f"GeoJSON join skipped ({e})")
    train['land_use'] = 'UNKNOWN'
    train['gpr']      = np.nan
    test['land_use']  = 'UNKNOWN'
    test['gpr']       = np.nan
    HAS_GEOJSON = False


# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════
print("\nEngineering features...")

CBD_LAT, CBD_LON = 1.2794, 103.8501  # Raffles Place MRT

def engineer_features(df):
    df = df.copy()

    # HDB leases are 99 years — remaining lease is a primary price driver
    df['years_remaining'] = 99 - df['hdb_age']

    # Storey ratio: how high the flat sits relative to its block
    df['storey_ratio'] = df['mid_storey'] / df['max_floor_lvl'].replace(0, np.nan)

    # Distance to CBD (Euclidean in degrees — monotone proxy, fine for Singapore)
    df['dist_to_cbd'] = np.sqrt(
        (df['Latitude'] - CBD_LAT) ** 2 + (df['Longitude'] - CBD_LON) ** 2
    )

    # Postal district (first 2 digits) — strong price signal, extracted before postal is dropped
    if 'postal' in df.columns:
        df['postal_district'] = (
            df['postal'].astype(str).str.zfill(6).str[:2]
            .apply(lambda x: int(x) if x.isdigit() else -1)
        )

    # Quarter
    df['Tranc_Quarter'] = ((df['Tranc_Month'] - 1) // 3) + 1

    # Five distinct market phases over the dataset period
    df['market_regime'] = pd.cut(
        df['Tranc_Year'],
        bins=[0, 2013, 2016, 2017, 2019, 2099],
        labels=['peak', 'early_decline', 'trough', 'recovery', 'covid_surge']
    ).astype(str)

    # Bigger flat on a higher floor commands a premium
    df['area_x_storey'] = df['floor_area_sqm'] * df['mid_storey']

    # Total dwelling units sold in the block (density proxy)
    sold_cols = ['1room_sold', '2room_sold', '3room_sold', '4room_sold',
                 '5room_sold', 'exec_sold', 'multigen_sold', 'studio_apartment_sold']
    df['total_units_sold'] = df[[c for c in sold_cols if c in df.columns]].sum(axis=1)

    return df

train = engineer_features(train)
test  = engineer_features(test)


# ═══════════════════════════════════════════════════════════════════
# 4. MISSING VALUE IMPUTATION
# ═══════════════════════════════════════════════════════════════════
# Within-radius counts: missing = no amenity within that radius, true value is 0
amenity_zero_cols = [
    'Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
    'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km',
]
for col in amenity_zero_cols:
    if col in train.columns:
        train[col] = train[col].fillna(0)
        test[col]  = test[col].fillna(0)

# Nearest distance: true distance measurement — use median
mall_median = train['Mall_Nearest_Distance'].median()
train['Mall_Nearest_Distance'] = train['Mall_Nearest_Distance'].fillna(mall_median)
test['Mall_Nearest_Distance']  = test['Mall_Nearest_Distance'].fillna(mall_median)


# ═══════════════════════════════════════════════════════════════════
# 5. DROP REDUNDANT / HIGH-CARDINALITY COLUMNS
# ═══════════════════════════════════════════════════════════════════
drop_cols = [
    'residential',                                      # constant "Y"
    'floor_area_sqft',                                  # duplicate of floor_area_sqm
    'storey_range', 'lower', 'upper', 'mid',            # decomposed into mid_storey
    'full_flat_type',                                   # derived from flat_type + flat_model
    'address',                                          # 9,157 unique values
    'Tranc_YearMonth',                                  # split into Year + Month
    'block', 'postal', 'bus_stop_name', 'street_name',  # high cardinality
]
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols,  inplace=True, errors='ignore')

# Drop gpr if GeoJSON join failed (all-NaN confuses the scaler)
if train['gpr'].isna().all():
    train.drop(columns=['gpr'], inplace=True, errors='ignore')
    test.drop(columns=['gpr'],  inplace=True, errors='ignore')

print(f"After drops — Train: {train.shape}  |  Test: {test.shape}")


# ═══════════════════════════════════════════════════════════════════
# 6. BOOLEAN ENCODING  (Y / N → 1 / 0)
# ═══════════════════════════════════════════════════════════════════
bool_cols = [
    'commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion',
    'bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation',
]
for col in bool_cols:
    if col in train.columns:
        # Only remap if stored as Y/N strings; leave alone if already numeric
        if train[col].dtype == object:
            train[col] = train[col].map({'Y': 1, 'N': 0})
            test[col]  = test[col].map({'Y': 1, 'N': 0})


# ═══════════════════════════════════════════════════════════════════
# 7. TARGET ENCODING  (high-cardinality categoricals)
#    K-fold encoding on train prevents target leakage.
# ═══════════════════════════════════════════════════════════════════
print("\nApplying target encoding...")
HIGH_CARD_COLS = [c for c in ['mrt_name', 'pri_sch_name', 'sec_sch_name']
                  if c in train.columns]

y_for_te = train['resale_price'].values
kf_te    = KFold(n_splits=5, shuffle=True, random_state=42)

for col in HIGH_CARD_COLS:
    global_mean = float(y_for_te.mean())
    te_values   = np.full(len(train), global_mean, dtype=float)

    for tr_idx, val_idx in kf_te.split(train):
        fold_df    = pd.DataFrame({'col': train[col].iloc[tr_idx].values,
                                   'y':   y_for_te[tr_idx]})
        fold_means = fold_df.groupby('col')['y'].mean()
        te_values[val_idx] = (train[col].iloc[val_idx]
                              .map(fold_means).fillna(global_mean).values)

    train[f'{col}_te'] = te_values

    full_df    = pd.DataFrame({'col': train[col].values, 'y': y_for_te})
    full_means = full_df.groupby('col')['y'].mean()
    test[f'{col}_te'] = test[col].map(full_means).fillna(global_mean)

    train.drop(columns=[col], inplace=True)
    test.drop(columns=[col],  inplace=True)

print(f"Target encoding complete — {len(HIGH_CARD_COLS)} columns encoded.")


# ═══════════════════════════════════════════════════════════════════
# 8. PREPARE FEATURES & TARGET
# ═══════════════════════════════════════════════════════════════════
X     = train.drop(columns=['resale_price'])
y     = train['resale_price']
y_log = np.log1p(y)

numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features:     {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)} — {categorical_features}")


# ═══════════════════════════════════════════════════════════════════
# 9. SKLEARN PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════════════
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])


# ═══════════════════════════════════════════════════════════════════
# 10. QUICK EDA
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating EDA plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(y, bins=60, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Resale Price Distribution')
axes[0, 0].set_xlabel('Price (SGD)')

sns.histplot(y_log, bins=60, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Log-Transformed Price Distribution')

yearly_avg = y.groupby(train['Tranc_Year']).mean()
axes[1, 0].plot(yearly_avg.index, yearly_avg.values, marker='o')
axes[1, 0].set_title('Average Resale Price by Year')
axes[1, 0].set_ylabel('Avg Price (SGD)')

town_avg = y.groupby(train['town']).mean().sort_values(ascending=False).head(15)
axes[1, 1].barh(town_avg.index, town_avg.values)
axes[1, 1].set_title('Avg Resale Price by Town (Top 15)')

plt.tight_layout()
plt.savefig('../eda_overview.png', dpi=100)
plt.close()
print("EDA saved → ../eda_overview.png")


# ═══════════════════════════════════════════════════════════════════
# 11. MODEL COMPARISON  (5-fold CV on log target)
# ═══════════════════════════════════════════════════════════════════
def cv_rmse(model, name):
    pipe   = Pipeline([('pre', preprocessor), ('reg', model)])
    scores = cross_val_score(pipe, X, y_log, cv=5,
                             scoring='neg_root_mean_squared_error', n_jobs=-1)
    rmse   = -scores.mean()
    print(f"  {name:<30} RMSE (log): {rmse:.4f}")
    return rmse

print("\n=== Model Comparison (5-fold CV) ===")
results = {}
results['Linear Regression'] = cv_rmse(LinearRegression(),                                          'Linear Regression')
results['Ridge']              = cv_rmse(Ridge(alpha=1.0, random_state=42),                           'Ridge')
results['Random Forest']      = cv_rmse(RandomForestRegressor(n_estimators=100, random_state=42,
                                                               n_jobs=-1),                           'Random Forest')
results['LightGBM']           = cv_rmse(lgb.LGBMRegressor(random_state=42, verbosity=-1),           'LightGBM')
results['Gradient Boosting']  = cv_rmse(GradientBoostingRegressor(n_estimators=100, random_state=42),'Gradient Boosting')
results['XGBoost']            = cv_rmse(xgb.XGBRegressor(random_state=42, verbosity=0, n_jobs=-1),  'XGBoost')

print("\n  --- Rankings (lower is better) ---")
for name, rmse in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name:<30} {rmse:.4f}")


# ═══════════════════════════════════════════════════════════════════
# 12. PREPROCESS ONCE — for fast Optuna tuning
# ═══════════════════════════════════════════════════════════════════
print("\nFitting preprocessor on full training data...")
X_processed = preprocessor.fit_transform(X)

# Hold out 20% for early-stopping signal during hyperparameter search
X_tr, X_val, y_tr, y_val = train_test_split(
    X_processed, y_log, test_size=0.2, random_state=42
)
print(f"Optuna split — Train: {X_tr.shape}  |  Val: {X_val.shape}")


# ═══════════════════════════════════════════════════════════════════
# 13. OPTUNA HYPERPARAMETER TUNING — LightGBM  (50 trials)
# ═══════════════════════════════════════════════════════════════════
print("\n=== LightGBM Hyperparameter Tuning (Optuna, 50 trials) ===")

def lgb_objective(trial):
    params = {
        'n_estimators':      3000,
        'learning_rate':     trial.suggest_float('learning_rate',     0.005, 0.1,  log=True),
        'num_leaves':        trial.suggest_int(  'num_leaves',        31,    255),
        'max_depth':         trial.suggest_int(  'max_depth',         4,     12),
        'subsample':         trial.suggest_float('subsample',         0.5,   1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree',  0.5,   1.0),
        'reg_alpha':         trial.suggest_float('reg_alpha',         1e-9,  10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda',        1e-9,  10.0, log=True),
        'min_child_samples': trial.suggest_int(  'min_child_samples', 5,     100),
        'random_state': 42, 'verbosity': -1, 'n_jobs': -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

lgb_study = optuna.create_study(direction='minimize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
lgb_study.optimize(lgb_objective, n_trials=50, show_progress_bar=True)

print(f"Best LightGBM val RMSE (log): {lgb_study.best_value:.4f}")
print(f"Best params: {lgb_study.best_params}")


# ═══════════════════════════════════════════════════════════════════
# 14. OPTUNA HYPERPARAMETER TUNING — XGBoost  (30 trials)
# ═══════════════════════════════════════════════════════════════════
print("\n=== XGBoost Hyperparameter Tuning (Optuna, 30 trials) ===")

def xgb_objective(trial):
    params = {
        'n_estimators':       3000,
        'early_stopping_rounds': 50,
        'eval_metric':        'rmse',
        'learning_rate':      trial.suggest_float('learning_rate',     0.005, 0.1,  log=True),
        'max_depth':          trial.suggest_int(  'max_depth',         3,     10),
        'subsample':          trial.suggest_float('subsample',         0.5,   1.0),
        'colsample_bytree':   trial.suggest_float('colsample_bytree',  0.5,   1.0),
        'reg_alpha':          trial.suggest_float('reg_alpha',         1e-9,  10.0, log=True),
        'reg_lambda':         trial.suggest_float('reg_lambda',        1e-9,  10.0, log=True),
        'min_child_weight':   trial.suggest_int(  'min_child_weight',  1,     20),
        'gamma':              trial.suggest_float('gamma',             1e-9,  1.0,  log=True),
        'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))

xgb_study = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_objective, n_trials=30, show_progress_bar=True)

print(f"Best XGBoost val RMSE (log): {xgb_study.best_value:.4f}")
print(f"Best params: {xgb_study.best_params}")


# ═══════════════════════════════════════════════════════════════════
# 15. FINAL MODEL TRAINING — retrain on full data
#     Probe on 80/20 split to find best iteration, then train on 100%.
# ═══════════════════════════════════════════════════════════════════
print("\nTraining final models on full training data...")

# — LightGBM —
lgb_probe = lgb.LGBMRegressor(
    **{**lgb_study.best_params,
       'n_estimators': 3000, 'random_state': 42, 'verbosity': -1, 'n_jobs': -1}
)
lgb_probe.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
)
lgb_n_est = int(lgb_probe.best_iteration_ * 1.1)  # 10% buffer for larger training set
print(f"LightGBM: probe stopped at {lgb_probe.best_iteration_}, "
      f"final n_estimators={lgb_n_est}")

final_lgb = lgb.LGBMRegressor(
    **{**lgb_study.best_params,
       'n_estimators': lgb_n_est, 'random_state': 42, 'verbosity': -1, 'n_jobs': -1}
)
final_lgb.fit(X_processed, y_log)

# — XGBoost —
xgb_probe = xgb.XGBRegressor(
    **{**xgb_study.best_params,
       'n_estimators': 3000, 'early_stopping_rounds': 50, 'eval_metric': 'rmse',
       'random_state': 42, 'verbosity': 0, 'n_jobs': -1}
)
xgb_probe.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
xgb_n_est = int(xgb_probe.best_iteration * 1.1)
print(f"XGBoost:  probe stopped at {xgb_probe.best_iteration}, "
      f"final n_estimators={xgb_n_est}")

final_xgb = xgb.XGBRegressor(
    **{**xgb_study.best_params,
       'n_estimators': xgb_n_est, 'random_state': 42, 'verbosity': 0, 'n_jobs': -1}
)
final_xgb.fit(X_processed, y_log)


# ═══════════════════════════════════════════════════════════════════
# 16. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════
print("\n=== Feature Importance (LightGBM) ===")

cat_feat_names = list(
    preprocessor.named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(categorical_features)
)
all_feat_names = numeric_features + cat_feat_names

importance_df = pd.DataFrame({
    'feature':    all_feat_names,
    'importance': final_lgb.feature_importances_,
}).sort_values('importance', ascending=False).reset_index(drop=True)

print("Top 25 features:")
print(importance_df.head(25).to_string(index=False))

plt.figure(figsize=(10, 9))
sns.barplot(x='importance', y='feature', data=importance_df.head(30))
plt.title('Top 30 Feature Importances — LightGBM')
plt.tight_layout()
plt.savefig('../feature_importance.png', dpi=100)
plt.close()
print("Feature importance saved → ../feature_importance.png")


# ═══════════════════════════════════════════════════════════════════
# 17. PREDICT ON TEST SET
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating test predictions...")

# Align test to exactly the same columns as X (in the same order)
test_X = test.drop(columns=['id'], errors='ignore').copy()
for col in X.columns:
    if col not in test_X.columns:
        test_X[col] = 0
test_X = test_X[X.columns]

X_test_proc = preprocessor.transform(test_X)

lgb_pred_log = final_lgb.predict(X_test_proc)
xgb_pred_log = final_xgb.predict(X_test_proc)

# Inverse-RMSE weighted blend: better model gets higher weight
lgb_w   = 1.0 / lgb_study.best_value
xgb_w   = 1.0 / xgb_study.best_value
total_w = lgb_w + xgb_w

blended_log = (lgb_pred_log * lgb_w + xgb_pred_log * xgb_w) / total_w
y_pred      = np.expm1(blended_log)

print(f"Blend — LightGBM: {lgb_w/total_w:.1%}  |  XGBoost: {xgb_w/total_w:.1%}")
print(f"Prediction range: {y_pred.min():.0f} – {y_pred.max():.0f}")
print(f"Mean prediction:  {y_pred.mean():.0f}")


# ═══════════════════════════════════════════════════════════════════
# 18. GENERATE SUBMISSION
# ═══════════════════════════════════════════════════════════════════
sample_sub = pd.read_csv('/Users/Josiah/Downloads/NTU_DASI4/Module3/Kaggle regression Challenge/hdb-price-model/data/sample_sub_reg.csv')

submission = pd.DataFrame({
    'Id':        test['id'],
    'Predicted': y_pred,
})
submission['Predicted'] = submission['Predicted'].round().astype(int)

# Kaggle only scores the 16,735 IDs in the sample submission — filter to those only
submission = submission[submission['Id'].isin(sample_sub.iloc[:, 0])]

submission.to_csv('/Users/Josiah/Downloads/NTU_DASI4/Module3/Kaggle regression Challenge/hdb-price-model/submission.csv', index=False)

assert len(submission) == 16735,                        f"Expected 16735 rows, got {len(submission)}"
assert list(submission.columns) == ['Id', 'Predicted'], f"Wrong columns: {list(submission.columns)}"

print(f"\nSubmission saved → ../submission.csv")
print(f"Shape: {submission.shape}")
print(submission.head(10).to_string(index=False))
print("\nDone.")
