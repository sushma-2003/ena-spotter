
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


DATA_DIR = Path(".")
TRAIN_PATH = DATA_DIR / "train-test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
DECEMBER_PATH = DATA_DIR / "december-chart-inputs.csv"
TEMPLATE_PATH = DATA_DIR / "validation-predictions-template.csv"

OUTPUT_VALIDATION = DATA_DIR / "validation_predictions.csv"
OUTPUT_DECEMBER = DATA_DIR / "december_chart_inputs_filled.csv"


def load_data():
    """Load all datasets and compute lat/lon averages from training data."""
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    december = pd.read_csv(DECEMBER_PATH)
    template = pd.read_csv(TEMPLATE_PATH)

    # Parse dates
    train['date'] = pd.to_datetime(train['date'])
    validation['date'] = pd.to_datetime(validation['date'])
    december['date'] = pd.to_datetime(december['date'])

    # Compute average lat/lon for each city from training data
    # We'll compute for both pickup and delivery
    pickup_lat_mean = train.groupby('pickup')['pickup_lat'].mean()
    pickup_lon_mean = train.groupby('pickup')['pickup_lon'].mean()
    delivery_lat_mean = train.groupby('delivery')['delivery_lat'].mean()
    delivery_lon_mean = train.groupby('delivery')['delivery_lon'].mean()

    lat_lon_maps = {
        'pickup_lat': pickup_lat_mean,
        'pickup_lon': pickup_lon_mean,
        'delivery_lat': delivery_lat_mean,
        'delivery_lon': delivery_lon_mean
    }

    return train, validation, december, template, lat_lon_maps



def impute_missing(df, train_stats=None, is_train=False):
    """Impute missing values for weight and market_index.

    For training: compute and store stats.
    For validation/test: use stored training stats.
    """
    df = df.copy()

    if is_train:
        # Compute imputation values from training data
        weight_median_by_equip = df.groupby('equipment')['weight'].median()
        market_index_median = df['market_index'].median()

        train_stats = {
            'weight_median_by_equip': weight_median_by_equip,
            'market_index_median': market_index_median,
            'weight_global_median': df['weight'].median()
        }
    else:
        weight_median_by_equip = train_stats['weight_median_by_equip']
        market_index_median = train_stats['market_index_median']
        weight_global_median = train_stats['weight_global_median']

    # Impute weight: by equipment median, fallback to global median
    missing_weight = df['weight'].isna()
    if missing_weight.any():
        df.loc[missing_weight, 'weight'] = df.loc[missing_weight, 'equipment'].map(weight_median_by_equip)
        # Fallback for any equipment not seen in training
        still_missing = df['weight'].isna()
        if still_missing.any():
            df.loc[still_missing, 'weight'] = weight_global_median

    # Impute market_index: global median
    df['market_index'] = df['market_index'].fillna(market_index_median)

    return df, train_stats


def create_features(df, lat_lon_maps=None):
    """Create engineered features.
    If lat/lon columns are missing, impute them using lat_lon_maps (from training data).
    """
    df = df.copy()

    # If lat/lon columns are missing, create them from city names using maps
    if 'pickup_lat' not in df.columns:
        df['pickup_lat'] = df['pickup'].map(lat_lon_maps['pickup_lat'])
        df['pickup_lon'] = df['pickup'].map(lat_lon_maps['pickup_lon'])
        df['delivery_lat'] = df['delivery'].map(lat_lon_maps['delivery_lat'])
        df['delivery_lon'] = df['delivery'].map(lat_lon_maps['delivery_lon'])

        # Fill any missing (city not seen in training) with global average
        df['pickup_lat'] = df['pickup_lat'].fillna(lat_lon_maps['pickup_lat'].mean())
        df['pickup_lon'] = df['pickup_lon'].fillna(lat_lon_maps['pickup_lon'].mean())
        df['delivery_lat'] = df['delivery_lat'].fillna(lat_lon_maps['delivery_lat'].mean())
        df['delivery_lon'] = df['delivery_lon'].fillna(lat_lon_maps['delivery_lon'].mean())

    # Time features
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['day'] = df['date'].dt.day
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Cyclical encoding for month and dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Quote signal (already present)
    df['quote_signal'] = df['quote_signal'].astype(float)

    # Equipment encoding (will be one-hot encoded in pipeline)
    df['equipment'] = df['equipment'].astype(str)

    # Distance features
    df['log_distance'] = np.log1p(df['distance'])
    df['distance_weight_ratio'] = df['distance'] / (df['weight'] + 1)

    # Geographic features
    df['lat_diff'] = df['delivery_lat'] - df['pickup_lat']
    df['lon_diff'] = df['delivery_lon'] - df['pickup_lon']
    df['abs_lat_diff'] = np.abs(df['lat_diff'])
    df['abs_lon_diff'] = np.abs(df['lon_diff'])

    # Route encoding (pickup-delivery pair)
    df['route'] = df['pickup'] + '_' + df['delivery']

    return df


def get_feature_columns():
    """Return list of feature columns for modeling."""
    return [
        'distance', 'weight', 'market_index', 'quote_signal',
        'month', 'dayofweek', 'day', 'dayofyear', 'quarter', 'is_weekend',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'log_distance', 'distance_weight_ratio',
        'lat_diff', 'lon_diff', 'abs_lat_diff', 'abs_lon_diff',
        'equipment'
    ]



def build_model():
    """Build the modeling pipeline."""
    numeric_features = [
        'distance', 'weight', 'market_index', 'quote_signal',
        'month', 'dayofweek', 'day', 'dayofyear', 'quarter', 'is_weekend',
        'month_sin', 'month_cos', 'dow_sin', 'dow_cos',
        'log_distance', 'distance_weight_ratio',
        'lat_diff', 'lon_diff', 'abs_lat_diff', 'abs_lon_diff'
    ]
    categorical_features = ['equipment']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )

    # Use HistGradientBoostingRegressor - handles missing values, fast, good performance
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            random_state=42,
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1.0
        ))
    ])

    return model


def time_based_split(df, test_months=2):
    """Split data temporally: hold out last N months for validation."""
    df = df.sort_values('date')
    unique_dates = df['date'].unique()
    # approximate cutoff: remove last test_months * 30 days
    cutoff_idx = max(0, len(unique_dates) - test_months * 30)
    cutoff_date = unique_dates[cutoff_idx] if cutoff_idx < len(unique_dates) else unique_dates[0]

    train_mask = df['date'] <= cutoff_date
    val_mask = df['date'] > cutoff_date

    return train_mask, val_mask


def train_and_evaluate(train_df):
    """Train model with time-based validation."""
    print("Preparing features...")
    X = train_df[get_feature_columns()]
    y = train_df['posted_rate']

    # Time-based split (hold out ~last 2 months = ~60 days)
    train_mask, val_mask = time_based_split(train_df, test_months=2)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    print(f"Train size: {len(X_train)}, Val size: {len(X_val)}")
    print(f"Train date range: {train_df.loc[train_mask, 'date'].min()} to {train_df.loc[train_mask, 'date'].max()}")
    print(f"Val date range: {train_df.loc[val_mask, 'date'].min()} to {train_df.loc[val_mask, 'date'].max()}")

    # Build and train model
    model = build_model()
    print("Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    train_mae = mean_absolute_error(y_train, train_pred)
    val_mae = mean_absolute_error(y_val, val_pred)

    print(f"\nTrain RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
    print(f"Val RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}")

    # Also compute MAPE
    val_mape = np.mean(np.abs((y_val - val_pred) / y_val)) * 100
    print(f"Val MAPE: {val_mape:.2f}%")

    # Feature importance (from the regressor)
    regressor = model.named_steps['regressor']
    if hasattr(regressor, 'feature_importances_'):
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out(['equipment'])
        feature_names = (
            get_feature_columns()[:-1] +  # all except equipment
            list(cat_features)
        )
        importances = regressor.feature_importances_
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feat_imp = feat_imp.sort_values('importance', ascending=False)
        print("\nTop 20 Feature Importances:")
        print(feat_imp.head(20).to_string(index=False))

    return model



def predict_validation(model, validation_df, template_df):
    """Generate predictions for validation set."""
    print("Generating validation predictions...")
    X_val = validation_df[get_feature_columns()]
    preds = model.predict(X_val)

    # Ensure positive predictions
    preds = np.maximum(preds, 1.0)

    # Fill template
    result = template_df.copy()
    result['predicted_rate'] = preds
    result.to_csv(OUTPUT_VALIDATION, index=False)
    print(f"Saved validation predictions to {OUTPUT_VALIDATION}")
    print(f"Prediction stats: mean={preds.mean():.2f}, min={preds.min():.2f}, max={preds.max():.2f}")

    return result


def predict_december(model, december_df):
    """Generate predictions for December chart."""
    print("Generating December predictions...")
    X_dec = december_df[get_feature_columns()]
    preds = model.predict(X_dec)

    # Ensure positive predictions
    preds = np.maximum(preds, 1.0)
    required_columns = ['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date']
    result = december_df[required_columns].copy()
    result['predicted_rate'] = preds

    # Ensure column order matches scorer expectation
    result = result[['pickup', 'delivery', 'distance', 'equipment', 'weight', 'date', 'predicted_rate']]

    result.to_csv(OUTPUT_DECEMBER, index=False)
    print(f"Saved December predictions to {OUTPUT_DECEMBER}")
    print(f"Prediction stats: mean={preds.mean():.2f}, min={preds.min():.2f}, max={preds.max():.2f}")

    return result



def main():
    print("=" * 60)
    print("Freight Rate Prediction Pipeline")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    train, validation, december, template, lat_lon_maps = load_data()
    print(f"   Train: {train.shape}, Validation: {validation.shape}, December: {december.shape}")

    # Impute missing values on training data
    print("\n2. Imputing missing values...")
    train, train_stats = impute_missing(train, is_train=True)
    validation, _ = impute_missing(validation, train_stats=train_stats, is_train=False)
    december, _ = impute_missing(december, train_stats=train_stats, is_train=False)

    print(f"   Train missing after impute: {train.isnull().sum().sum()}")
    print(f"   Validation missing after impute: {validation.isnull().sum().sum()}")
    print(f"   December missing after impute: {december.isnull().sum().sum()}")

    # Feature engineering
    print("\n3. Feature engineering...")
    train = create_features(train, lat_lon_maps=lat_lon_maps)
    validation = create_features(validation, lat_lon_maps=lat_lon_maps)
    december = create_features(december, lat_lon_maps=lat_lon_maps)

    # Train model
    print("\n4. Training model with time-based validation...")
    model = train_and_evaluate(train)

    # Retrain on full training data for final predictions
    print("\n5. Retraining on full training data...")
    X_full = train[get_feature_columns()]
    y_full = train['posted_rate']
    model.fit(X_full, y_full)
    print("   Full model trained.")

    # Predict validation
    print("\n6. Predicting validation set...")
    predict_validation(model, validation, template)

    # Predict December
    print("\n7. Predicting December chart...")
    predict_december(model, december)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
