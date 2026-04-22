"""
COMPREHENSIVE FEATURE ANALYSIS
1. Feature Importance Ranking
2. Ablation Study (RMSE impact per feature)
3. ROI Analysis (complexity vs. improvement)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

print("="*70)
print("FEATURE ANALYSIS - KAGGLE HDB PRICE PREDICTION")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# 1. LOAD & PREPARE DATA (same as KaggleFinal.py)
# ═══════════════════════════════════════════════════════════════════
print("\n[1/3] Loading and preprocessing data...")

train = pd.read_csv('../data/train.csv', low_memory=False)
test = pd.read_csv('../data/test.csv', low_memory=False)

drop_cols = ['residential', 'floor_area_sqft', 'storey_range', 'lower', 'upper', 'mid',
             'full_flat_type', 'address', 'Tranc_YearMonth', 'block', 'postal', 
             'bus_stop_name', 'street_name']
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

# Handle missing values
amenity_cols = ['Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
                'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km']
for col in amenity_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)
train['Mall_Nearest_Distance'] = train['Mall_Nearest_Distance'].fillna(train['Mall_Nearest_Distance'].median())
test['Mall_Nearest_Distance'] = test['Mall_Nearest_Distance'].fillna(test['Mall_Nearest_Distance'].median())

# Boolean columns
bool_cols = ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion', 
             'bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']
for col in bool_cols:
    train[col] = train[col].map({'Y': 1, 'N': 0})
    test[col] = test[col].map({'Y': 1, 'N': 0})

# Feature engineering
current_year = 2024
train['market_regime'] = train['Tranc_Year'].apply(lambda x: 0 if x <= 2013 else (1 if x <= 2016 else (2 if x <= 2018 else 3)))
test['market_regime'] = test['Tranc_Year'].apply(lambda x: 0 if x <= 2013 else (1 if x <= 2016 else (2 if x <= 2018 else 3)))
train['building_age'] = current_year - train['year_completed']
test['building_age'] = current_year - test['year_completed']
train['lease_remaining'] = 99 - (current_year - train['lease_commence_date'])
test['lease_remaining'] = 99 - (current_year - test['lease_commence_date'])
train['floor_area_per_room'] = train['floor_area_sqm'] / (train['total_dwelling_units'] + 1)
test['floor_area_per_room'] = test['floor_area_sqm'] / (test['total_dwelling_units'] + 1)
train['Tranc_Quarter'] = ((train['Tranc_Month'] - 1) // 3) + 1
test['Tranc_Quarter'] = ((test['Tranc_Month'] - 1) // 3) + 1

# Interaction features
train['age_lease_interaction'] = train['building_age'] * train['lease_remaining']
test['age_lease_interaction'] = test['building_age'] * test['lease_remaining']
train['floor_area_price_proxy'] = train['floor_area_sqm'] * train['total_dwelling_units']
test['floor_area_price_proxy'] = test['floor_area_sqm'] * test['total_dwelling_units']
train['amenity_score'] = (train['Mall_Within_1km'] + train['Hawker_Within_1km']) / (train['Mall_Nearest_Distance'] + 1)
test['amenity_score'] = (test['Mall_Within_1km'] + test['Hawker_Within_1km']) / (test['Mall_Nearest_Distance'] + 1)
train['price_per_sqm_proxy'] = train['floor_area_sqm'] / (train['total_dwelling_units'] + 1)
test['price_per_sqm_proxy'] = test['floor_area_sqm'] / (test['total_dwelling_units'] + 1)

# Target encoding
def target_encode(train_data, test_data, col, target, smoothing=1.0):
    means = train_data.groupby(col)[target].mean()
    global_mean = train_data[target].mean()
    counts = train_data.groupby(col).size()
    smoothed = (means * counts + global_mean * smoothing) / (counts + smoothing)
    train_data[f'{col}_encoded'] = train_data[col].map(smoothed).fillna(global_mean)
    test_data[f'{col}_encoded'] = test_data[col].map(smoothed).fillna(global_mean)
    return train_data, test_data

y = train['resale_price'].copy()
y_log = np.log1p(y)

for col in ['planning_area', 'mrt_name']:
    if col in train.columns:
        train, test = target_encode(train, test, col, 'resale_price', smoothing=5.0)
        train.drop(columns=[col], inplace=True, errors='ignore')
        test.drop(columns=[col], inplace=True, errors='ignore')

# Prepare X data
X = train.drop(columns=['resale_price', 'id'], errors='ignore')
X_test = test.drop(columns=['id'], errors='ignore')

# One-hot encode remaining categoricals
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
if cat_cols:
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

# Align columns
missing_cols = set(X.columns) - set(X_test.columns)
for col in missing_cols:
    X_test[col] = 0
X_test = X_test[X.columns]

print(f"✓ Data prepared: {X.shape}")
feature_names = list(X.columns)

# ═══════════════════════════════════════════════════════════════════
# 2. TRAIN BASELINE MODEL
# ═══════════════════════════════════════════════════════════════════
print("\n[2/3] Training baseline ensemble model...")

lgb_model = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31, 
                               max_depth=6, subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbosity=-1)

xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                              reg_lambda=1.0, random_state=42, verbosity=0)

cb_model = cb.CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6,
                                l2_leaf_reg=3, random_seed=42, verbose=False)

# Train individual models for importance extraction
lgb_model.fit(X, y_log)
xgb_model.fit(X, y_log)
cb_model.fit(X, y_log)

print("✓ Models trained")

# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 1: FEATURE IMPORTANCE RANKING")
print("="*70)

# LightGBM importance
lgb_importance = pd.DataFrame({
    'feature': feature_names,
    'lgb_gain': lgb_model.booster_.feature_importance(importance_type='gain'),
    'lgb_split': lgb_model.booster_.feature_importance(importance_type='split')
})

# XGBoost importance
xgb_importance = pd.DataFrame({
    'feature': feature_names,
    'xgb_gain': xgb_model.feature_importances_
})

# CatBoost importance
cb_importance = pd.DataFrame({
    'feature': feature_names,
    'cb_importance': cb_model.feature_importances_
})

# Merge all
importance_df = lgb_importance.merge(xgb_importance, on='feature').merge(cb_importance, on='feature')

# Normalize to 0-100 scale
for col in ['lgb_gain', 'xgb_gain', 'cb_importance']:
    importance_df[col] = (importance_df[col] / importance_df[col].max()) * 100

# Calculate ensemble importance (average)
importance_df['ensemble_importance'] = importance_df[['lgb_gain', 'xgb_gain', 'cb_importance']].mean(axis=1)
importance_df = importance_df.sort_values('ensemble_importance', ascending=False)

print("\n🔝 TOP 20 MOST IMPORTANT FEATURES:")
print("-" * 70)
print(importance_df[['feature', 'lgb_gain', 'xgb_gain', 'cb_importance', 'ensemble_importance']].head(20).to_string(index=False))

# Save top features
top_20_features = importance_df.head(20)['feature'].tolist()

# ═══════════════════════════════════════════════════════════════════
# 4. ABLATION STUDY - RMSE IMPACT
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 2: ABLATION STUDY (Feature Removal Impact on RMSE)")
print("="*70)

# Baseline CV score
print("\nCalculating baseline RMSE (all features)...")
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([
    ('lgb', lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=31,
                              max_depth=6, random_state=42, verbosity=-1)),
    ('xgb', xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42, verbosity=0)),
    ('cb', cb.CatBoostRegressor(iterations=100, learning_rate=0.05, depth=6, random_seed=42, verbose=False))
])

cv_scores = cross_val_score(ensemble, X, y_log, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
baseline_rmse = np.sqrt(-cv_scores.mean())

print(f"Baseline RMSE (all {len(feature_names)} features): {baseline_rmse:.4f}")

# Ablation: remove each top 20 feature one at a time
ablation_results = []

for i, feature in enumerate(top_20_features):
    print(f"\rAblating {i+1}/20: {feature:<40}", end='', flush=True)
    
    X_ablated = X.drop(columns=[feature])
    cv_scores_ablated = cross_val_score(ensemble, X_ablated, y_log, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    ablated_rmse = np.sqrt(-cv_scores_ablated.mean())
    
    rmse_increase = ablated_rmse - baseline_rmse
    pct_increase = (rmse_increase / baseline_rmse) * 100
    
    ablation_results.append({
        'feature': feature,
        'baseline_rmse': baseline_rmse,
        'ablated_rmse': ablated_rmse,
        'rmse_delta': rmse_increase,
        'pct_impact': pct_increase
    })

print("\n")

ablation_df = pd.DataFrame(ablation_results).sort_values('rmse_delta', ascending=False)

print("\n🎯 ABLATION STUDY RESULTS (Ranked by RMSE Impact):")
print("-" * 70)
print(ablation_df[['feature', 'baseline_rmse', 'ablated_rmse', 'rmse_delta', 'pct_impact']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# 5. ROI ANALYSIS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ANALYSIS 3: ROI ANALYSIS (Complexity vs. Impact)")
print("="*70)

# Feature complexity scoring
roi_df = ablation_df.copy()

# Define feature categories and their complexity
def get_complexity_score(feature_name):
    """Score complexity from 1-10"""
    name_lower = feature_name.lower()
    
    # Engineered features (high complexity)
    if any(x in name_lower for x in ['interaction', 'proxy', 'score', '_encoded', 'age_lease']):
        return 8
    # Time-based features (medium-high)
    elif any(x in name_lower for x in ['tranc', 'quarter', 'market_regime']):
        return 6
    # Spatial features (medium-high)
    elif any(x in name_lower for x in ['distance', 'nearest', 'lat', 'lon']):
        return 7
    # Amenity features (medium)
    elif any(x in name_lower for x in ['mall', 'hawker']):
        return 5
    # Simple/raw features (low complexity)
    else:
        return 3

roi_df['complexity'] = roi_df['feature'].apply(get_complexity_score)
roi_df['roi_score'] = roi_df['rmse_delta'] / roi_df['complexity']
roi_df = roi_df.sort_values('roi_score', ascending=False)

print("\n💰 ROI RANKINGS (Impact per Complexity Unit):")
print("-" * 70)
print(roi_df[['feature', 'rmse_delta', 'complexity', 'roi_score']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════
# 6. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("Generating visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Feature Importance
ax = axes[0, 0]
importance_df.head(15).sort_values('ensemble_importance').plot(
    x='feature', y='ensemble_importance', kind='barh', ax=ax, color='steelblue', legend=False
)
ax.set_xlabel('Ensemble Importance Score')
ax.set_title('Top 15 Features by Importance', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Plot 2: Ablation Study Results
ax = axes[0, 1]
ablation_df.head(15).sort_values('rmse_delta').plot(
    x='feature', y='rmse_delta', kind='barh', ax=ax, color='coral', legend=False
)
ax.set_xlabel('RMSE Increase When Removed')
ax.set_title('Top 15 Features by RMSE Impact', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Plot 3: ROI Analysis
ax = axes[1, 0]
scatter = ax.scatter(roi_df.head(15)['complexity'], roi_df.head(15)['rmse_delta'],
                     s=100, alpha=0.6, c=roi_df.head(15)['roi_score'], cmap='viridis')
for i, row in roi_df.head(15).iterrows():
    ax.annotate(row['feature'][:15], (row['complexity'], row['rmse_delta']),
               fontsize=8, alpha=0.7)
ax.set_xlabel('Feature Complexity (1-10)')
ax.set_ylabel('RMSE Impact')
ax.set_title('ROI: Impact vs. Complexity (Top 15 Features)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax, label='ROI Score')
ax.grid(alpha=0.3)

# Plot 4: Diminishing Returns Analysis
ax = axes[1, 1]
cumsum_importance = importance_df.head(20).sort_values('ensemble_importance', ascending=False)
cumulative_importance = np.cumsum(cumsum_importance['ensemble_importance'].values)
ax.plot(range(1, 21), cumulative_importance, marker='o', linewidth=2, markersize=6, color='darkgreen')
ax.axhline(y=80, color='red', linestyle='--', label='80% Threshold')
ax.set_xlabel('Number of Features')
ax.set_ylabel('Cumulative Importance')
ax.set_title('Diminishing Returns: Cumulative Feature Importance', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../feature_analysis_report.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization → ../feature_analysis_report.png")

# ═══════════════════════════════════════════════════════════════════
# 7. SUMMARY & CONCLUSIONS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY & CONCLUSIONS")
print("="*70)

print(f"\n📊 Dataset: {X.shape[0]} records, {X.shape[1]} features")
print(f"📈 Baseline RMSE: {baseline_rmse:.4f}")

# Find 80% threshold
cumsum = np.cumsum(sorted(importance_df['ensemble_importance'].values, reverse=True))
features_for_80pct = np.argmax(cumsum >= (cumsum[-1] * 0.8)) + 1
print(f"\n⚡ Diminishing Returns:")
print(f"   • Top {features_for_80pct} features achieve 80% of total importance")
print(f"   • Remaining {len(feature_names) - features_for_80pct} features contribute only 20%")

# High-value features
print(f"\n🔝 High-Value Features (Top ROI):")
for idx, row in roi_df.head(5).iterrows():
    print(f"   • {row['feature']:<35} — ROI: {row['roi_score']:.3f} (RMSE impact: +{row['rmse_delta']:.4f})")

# Low-value features
print(f"\n❌ Low-Value Features (High complexity, low impact):")
low_roi = roi_df.sort_values('roi_score', ascending=True).head(5)
for idx, row in low_roi.iterrows():
    print(f"   • {row['feature']:<35} — ROI: {row['roi_score']:.3f} (complexity: {row['complexity']}/10)")

print(f"\n💡 RECOMMENDATIONS:")
print(f"   1. Keep top {features_for_80pct} features for faster inference")
print(f"   2. Consider removing low-ROI features: {', '.join(low_roi['feature'].head(3).tolist())}")
print(f"   3. Focus engineering efforts on top-ROI features")
print(f"   4. Current model complexity is justified by {baseline_rmse:.4f} RMSE")

print("\n" + "="*70)
print("Analysis complete! Check ../feature_analysis_report.png for visualizations")
print("="*70)
