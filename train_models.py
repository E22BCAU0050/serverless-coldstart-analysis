#!/usr/bin/env python3
"""
Cold Start Prediction — Hybrid ML + DL Framework
==================================================
ML Part:  Random Forest + XGBoost to predict cold start latency
DL Part:  LSTM to predict invocation traffic (time-series)
Output:   Recommendation engine for optimal Lambda config

Usage:
    pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn
    python train_models.py --dataset cold_start_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost not installed. Skipping XGB model.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠ TensorFlow not installed. Skipping LSTM model.")

# ── Args ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cold_start_dataset.csv')
parser.add_argument('--output_dir', default='model_outputs')
args = parser.parse_args()

import os
os.makedirs(args.output_dir, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────

def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} records")

    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df['function_type'] = df['function_type'].fillna('unknown')
    df['api_path']      = df['api_path'].fillna('/unknown')
    df['api_method']    = df['api_method'].fillna('GET')

    # Encode categoricals
    le = LabelEncoder()
    df['function_type_enc'] = le.fit_transform(df['function_type'])
    df['api_method_enc']    = le.fit_transform(df['api_method'])

    # Parse timestamps
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df['hour_of_day']  = df['timestamp'].dt.hour.fillna(0).astype(int)
        df['day_of_week']  = df['timestamp'].dt.dayofweek.fillna(0).astype(int)

    return df

# ─────────────────────────────────────────────────────────────
# ML MODELS: Predict cold start init_duration_ms
# ─────────────────────────────────────────────────────────────

ML_FEATURES = [
    'memory_size_mb', 'vpc_flag', 'provisioned_flag', 'container_flag',
    'hour_of_day', 'day_of_week', 'function_type_enc', 'api_method_enc',
]

def train_ml_models(df: pd.DataFrame):
    print("\n" + "="*60)
    print("ML MODELS — Cold Start Latency Prediction")
    print("="*60)

    # Filter to cold starts only (init_duration > 0)
    cold_df = df[df['init_duration_ms'] > 0].copy()
    if len(cold_df) < 10:
        print(f"⚠ Only {len(cold_df)} cold start records. Generating synthetic data for demo...")
        cold_df = generate_synthetic_data()

    features = [f for f in ML_FEATURES if f in cold_df.columns]
    X = cold_df[features].values
    y = cold_df['init_duration_ms'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
        'Ridge Regression': Ridge(alpha=1.0),
    }

    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        )

    results = {}
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)
        cv   = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mae')

        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'CV_MAE': -cv.mean()}
        print(f"\n{name}:")
        print(f"  MAE:   {mae:.2f} ms")
        print(f"  RMSE:  {rmse:.2f} ms")
        print(f"  R²:    {r2:.4f}")
        print(f"  CV MAE:{-cv.mean():.2f} ms (±{cv.std():.2f})")

    # Feature importance (Random Forest)
    rf_model = models['Random Forest']
    fi = pd.DataFrame({
        'feature':   features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nFeature Importances (Random Forest):")
    print(fi.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(fi['feature'], fi['importance'], color='steelblue')
    axes[0].set_title('Feature Importance — Cold Start Prediction')
    axes[0].set_xlabel('Importance')

    results_df = pd.DataFrame(results).T
    results_df[['MAE', 'RMSE']].plot(kind='bar', ax=axes[1], color=['coral', 'steelblue'])
    axes[1].set_title('Model Comparison (lower is better)')
    axes[1].set_ylabel('Latency (ms)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/ml_model_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: {args.output_dir}/ml_model_comparison.png")

    return models['Random Forest'], scaler, features, results

# ─────────────────────────────────────────────────────────────
# DL MODEL: LSTM for time-series invocation prediction
# ─────────────────────────────────────────────────────────────

def prepare_lstm_sequences(df: pd.DataFrame, lookback: int = 10):
    """Create sliding window sequences for LSTM."""
    if 'timestamp' not in df.columns or df['timestamp'].isna().all():
        return None, None

    ts_df = df.set_index('timestamp').sort_index()
    # Resample to per-minute invocation count
    hourly = ts_df.resample('1min')['duration_ms'].count().fillna(0)
    series = hourly.values.astype(float)

    if len(series) < lookback + 1:
        return None, None

    scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    X, y = [], []
    for i in range(lookback, len(series_scaled)):
        X.append(series_scaled[i-lookback:i])
        y.append(series_scaled[i])

    return np.array(X).reshape(-1, lookback, 1), np.array(y), scaler

def build_lstm_model(lookback: int = 10) -> 'tf.keras.Model':
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_lstm(df: pd.DataFrame, lookback: int = 10):
    if not HAS_TF:
        print("\n⚠ TensorFlow not available — skipping LSTM training")
        return None

    print("\n" + "="*60)
    print("DL MODEL — LSTM Traffic Prediction")
    print("="*60)

    result = prepare_lstm_sequences(df, lookback)
    if result[0] is None:
        print("⚠ Not enough time-series data. Need more invocations spread over time.")
        return None

    X, y, ts_scaler = result
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model(lookback)
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(patience=5, factor=0.5),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100, batch_size=32,
        callbacks=callbacks, verbose=0
    )

    preds = model.predict(X_test).flatten()
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"LSTM Test MAE:  {mae:.4f}")
    print(f"LSTM Test RMSE: {rmse:.4f}")

    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('LSTM Training History')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.tight_layout()
    plt.savefig(f'{args.output_dir}/lstm_training.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {args.output_dir}/lstm_training.png")

    return model

# ─────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────

def recommend_config(rf_model, scaler, features, df: pd.DataFrame):
    print("\n" + "="*60)
    print("RECOMMENDATION ENGINE")
    print("="*60)

    configs = [
        {'name': 'Non-VPC 128MB',    'memory_size_mb': 128,  'vpc_flag': 0, 'provisioned_flag': 0, 'container_flag': 0},
        {'name': 'Non-VPC 512MB',    'memory_size_mb': 512,  'vpc_flag': 0, 'provisioned_flag': 0, 'container_flag': 0},
        {'name': 'Non-VPC 1024MB',   'memory_size_mb': 1024, 'vpc_flag': 0, 'provisioned_flag': 0, 'container_flag': 0},
        {'name': 'VPC 512MB',        'memory_size_mb': 512,  'vpc_flag': 1, 'provisioned_flag': 0, 'container_flag': 0},
        {'name': 'VPC 1024MB',       'memory_size_mb': 1024, 'vpc_flag': 1, 'provisioned_flag': 0, 'container_flag': 0},
        {'name': 'Container 512MB',  'memory_size_mb': 512,  'vpc_flag': 0, 'provisioned_flag': 0, 'container_flag': 1},
        {'name': 'Provisioned 512MB','memory_size_mb': 512,  'vpc_flag': 0, 'provisioned_flag': 1, 'container_flag': 0},
    ]

    recommendations = []
    for cfg in configs:
        row = {f: cfg.get(f, 0) for f in features}
        row['hour_of_day']       = 12
        row['day_of_week']       = 1
        row['function_type_enc'] = 0
        row['api_method_enc']    = 0

        X_pred = scaler.transform([[row[f] for f in features]])
        pred_cold_start = rf_model.predict(X_pred)[0]

        # Cost estimate (Lambda pricing: $0.0000000083/ms/GB)
        gb_seconds = (cfg['memory_size_mb'] / 1024) * (pred_cold_start / 1000)
        cost_per_10k = gb_seconds * 0.0000166667 * 10000

        recommendations.append({
            **cfg,
            'predicted_cold_start_ms': round(pred_cold_start, 1),
            'est_cost_per_10k_invocations_usd': round(cost_per_10k, 6),
        })

    rec_df = pd.DataFrame(recommendations)[['name', 'predicted_cold_start_ms', 'est_cost_per_10k_invocations_usd']]
    rec_df = rec_df.sort_values('predicted_cold_start_ms')

    print("\nConfig Recommendations (sorted by predicted cold start):")
    print(rec_df.to_string(index=False))

    best = rec_df.iloc[0]
    print(f"\n🏆 RECOMMENDED CONFIG: {best['name']}")
    print(f"   Predicted Cold Start: {best['predicted_cold_start_ms']} ms")
    print(f"   Cost per 10k invocations: ${best['est_cost_per_10k_invocations_usd']}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['green' if i == 0 else 'steelblue' for i in range(len(rec_df))]
    ax.barh(rec_df['name'], rec_df['predicted_cold_start_ms'], color=colors)
    ax.set_title('Predicted Cold Start Latency by Lambda Configuration')
    ax.set_xlabel('Predicted Cold Start (ms)')
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/config_recommendations.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {args.output_dir}/config_recommendations.png")

    rec_df.to_csv(f'{args.output_dir}/recommendations.csv', index=False)
    return rec_df

# ─────────────────────────────────────────────────────────────
# Synthetic data generator (for testing before real data)
# ─────────────────────────────────────────────────────────────

def generate_synthetic_data(n: int = 500) -> pd.DataFrame:
    """Generate synthetic cold start data that mirrors real AWS Lambda behavior."""
    np.random.seed(42)
    records = []
    configs = [
        {'function_type': 'non-vpc',    'vpc_flag': 0, 'provisioned_flag': 0, 'base_init': 180, 'std': 60},
        {'function_type': 'vpc',        'vpc_flag': 1, 'provisioned_flag': 0, 'base_init': 820, 'std': 200},
        {'function_type': 'provisioned','vpc_flag': 0, 'provisioned_flag': 1, 'base_init': 10,  'std': 5},
    ]
    memory_options = [128, 256, 512, 1024, 2048]

    for _ in range(n):
        cfg = configs[np.random.choice(len(configs))]
        mem = np.random.choice(memory_options)
        # Higher memory = slightly lower cold start
        mem_factor = 1.0 - (np.log2(mem) - np.log2(128)) * 0.05
        init = max(5, np.random.normal(cfg['base_init'] * mem_factor, cfg['std']))

        records.append({
            'function_type':    cfg['function_type'],
            'memory_size_mb':   mem,
            'vpc_flag':         cfg['vpc_flag'],
            'provisioned_flag': cfg['provisioned_flag'],
            'container_flag':   np.random.choice([0, 1], p=[0.8, 0.2]),
            'init_duration_ms': round(init, 2),
            'duration_ms':      round(np.random.exponential(80) + 20, 2),
            'hour_of_day':      np.random.randint(0, 24),
            'day_of_week':      np.random.randint(0, 7),
            'cold_start_flag':  1,
            'function_type_enc': cfg['vpc_flag'] + cfg['provisioned_flag'] * 2,
            'api_method_enc':   0,
        })
    return pd.DataFrame(records)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    try:
        df = load_and_preprocess(args.dataset)
    except FileNotFoundError:
        print(f"⚠ Dataset file not found: {args.dataset}")
        print("  Generating synthetic data for demonstration...")
        df = generate_synthetic_data(1000)

    rf_model, scaler, features, results = train_ml_models(df)
    train_lstm(df)
    recommend_config(rf_model, scaler, features, df)

    # Save results summary
    summary = {
        'dataset_records':  len(df),
        'cold_start_count': int(df.get('cold_start_flag', pd.Series([0])).sum()),
        'model_results':    {k: {m: round(v, 2) for m, v in v.items()} for k, v in results.items()},
    }
    import json
    with open(f'{args.output_dir}/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Summary saved: {args.output_dir}/training_summary.json")

if __name__ == '__main__':
    main()
