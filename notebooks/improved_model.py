import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import spatial join functionality
from spatial_join import load_geojson_files, create_spatial_join

# Load data
print("Loading data...")
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Load GeoJSON files for spatial features
print("\nLoading spatial data...")
gdf_2014, gdf_2019, gdf_2025 = load_geojson_files()

# Add spatial features
print("Adding spatial features to train...")
train = create_spatial_join(train, gdf_2014, gdf_2019, gdf_2025)

print("Adding spatial features to test...")
test = create_spatial_join(test, gdf_2014, gdf_2019, gdf_2025)

print(f"Train shape after spatial join: {train.shape}")
print(f"Test shape after spatial join: {test.shape}")

# Data preprocessing
drop_cols = [
    'residential', 'floor_area_sqft', 'storey_range', 'lower', 'upper', 'mid',
    'full_flat_type', 'address', 'Tranc_YearMonth', 'block', 'postal', 
    'bus_stop_name', 'street_name'
]

train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

# Handle missing values
amenity_cols = [
    'Mall_Within_500m', 'Mall_Within_1km', 'Mall_Within_2km',
    'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km'
]

for col in amenity_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

train['Mall_Nearest_Distance'] = train['Mall_Nearest_Distance'].fillna(train['Mall_Nearest_Distance'].median())
test['Mall_Nearest_Distance'] = test['Mall_Nearest_Distance'].fillna(test['Mall_Nearest_Distance'].median())

# Convert boolean columns
bool_cols = ['commercial', 'market_hawker', 'multistorey_carpark', 'precinct_pavilion', 
             'bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']

for col in bool_cols:
    train[col] = train[col].map({'Y': 1, 'N': 0})
    test[col] = test[col].map({'Y': 1, 'N': 0})

# Additional feature engineering
train['Tranc_Quarter'] = ((train['Tranc_Month'] - 1) // 3) + 1
test['Tranc_Quarter'] = ((test['Tranc_Month'] - 1) // 3) + 1

# Market regime feature
def get_market_regime(year):
    if year <= 2013:
        return 'early_decline'
    elif year <= 2016:
        return 'continued_decline' 
    elif year <= 2018:
        return 'recovery'
    else:
        return 'high_demand'

train['market_regime'] = train['Tranc_Year'].apply(get_market_regime)
test['market_regime'] = test['Tranc_Year'].apply(get_market_regime)

# Age-based features
current_year = 2024
train['building_age'] = current_year - train['year_completed']
test['building_age'] = current_year - test['year_completed']

# Lease remaining
train['lease_remaining'] = 99 - (current_year - train['lease_commence_date'])
test['lease_remaining'] = 99 - (current_year - test['lease_commence_date'])

# Interaction features
train['floor_area_per_room'] = train['floor_area_sqm'] / (train['total_dwelling_units'] + 1)
test['floor_area_per_room'] = test['floor_area_sqm'] / (test['total_dwelling_units'] + 1)

print(f"Final train shape: {train.shape}")
print(f"Final test shape: {test.shape}")

# Target encoding for high-cardinality features
from sklearn.model_selection import KFold

def target_encode(train, test, column, target, n_splits=5):
    """Target encode a categorical column"""
    train_encoded = train.copy()
    test_encoded = test.copy()
    
    train_encoded[f'{column}_encoded'] = 0
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(train):
        train_fold = train.iloc[train_idx]
        val_fold = train.iloc[val_idx]
        
        # Calculate mean target for each category in training fold
        means = train_fold.groupby(column)[target].mean()
        
        # Apply to validation fold
        train_encoded.loc[val_idx, f'{column}_encoded'] = val_fold[column].map(means)
    
    # For test set, use overall means from train
    overall_means = train.groupby(column)[target].mean()
    test_encoded[f'{column}_encoded'] = test[column].map(overall_means)
    
    # Fill NaN with global mean
    global_mean = train[target].mean()
    train_encoded[f'{column}_encoded'] = train_encoded[f'{column}_encoded'].fillna(global_mean)
    test_encoded[f'{column}_encoded'] = test_encoded[f'{column}_encoded'].fillna(global_mean)
    
    return train_encoded, test_encoded

# Prepare target
y = train['resale_price']
y_log = np.log1p(y)
X = train.drop('resale_price', axis=1)

# Target encode high-cardinality features
high_cardinality_cols = ['mrt_name', 'pri_sch_name', 'sec_sch_name', 'planning_area']

for col in high_cardinality_cols:
    if col in X.columns:
        print(f"Target encoding {col}...")
        X, test_encoded = target_encode(X.assign(resale_price=y_log), 
                                       test.assign(resale_price=y_log.mean()), 
                                       col, 'resale_price')
        test = test_encoded.drop('resale_price', axis=1)

# Drop original high-cardinality columns
X = X.drop(columns=high_cardinality_cols, errors='ignore')
test = test.drop(columns=high_cardinality_cols, errors='ignore')

print(f"After target encoding - Train shape: {X.shape}")
print(f"After target encoding - Test shape: {test.shape}")

# Prepare final datasets
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# One-hot encode remaining categorical features
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
test_encoded = pd.get_dummies(test, columns=categorical_features, drop_first=True)

# Ensure test has same columns as train
missing_cols = set(X_encoded.columns) - set(test_encoded.columns)
for col in missing_cols:
    test_encoded[col] = 0

extra_cols = set(test_encoded.columns) - set(X_encoded.columns) 
test_encoded = test_encoded.drop(columns=extra_cols, errors='ignore')

# Align column order
test_encoded = test_encoded[X_encoded.columns]

print(f"Final encoded shapes - Train: {X_encoded.shape}, Test: {test_encoded.shape}")

# Train improved model
import lightgbm as lgb
from sklearn.model_selection import cross_val_score

print("\nTraining improved LightGBM model...")

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    verbosity=-1
)

# Cross-validation
cv_scores = cross_val_score(model, X_encoded, y_log, cv=5, 
                           scoring='neg_root_mean_squared_error', verbose=1)
cv_rmse = -cv_scores.mean()
print(f"Cross-validation RMSE (log scale): {cv_rmse:.4f}")

# Train final model
model.fit(X_encoded, y_log)

# Feature importance
importance_df = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Feature Importances:")
print(importance_df.head(15))

# Make predictions
print("\nGenerating predictions...")
y_pred_log = model.predict(test_encoded)
y_pred = np.expm1(y_pred_log)

# Create submission (filtered to match sample)
sample = pd.read_csv('../data/sample_sub_reg.csv')
sample_ids = set(sample['Id'])

submission = pd.DataFrame({
    'Id': test['id'],
    'Predicted': y_pred.round().astype(int)
})

# Filter to only include IDs in sample
submission_filtered = submission[submission['Id'].isin(sample_ids)]

print(f"Original predictions: {len(submission)} rows")
print(f"Filtered submission: {len(submission_filtered)} rows")
print(f"Prediction range: {submission_filtered['Predicted'].min()} - {submission_filtered['Predicted'].max()}")

# Save submission
submission_filtered.to_csv('../submission_improved.csv', index=False)
print("Improved submission saved to ../submission_improved.csv")
