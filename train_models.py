#!/usr/bin/env python3
"""
Cold Start Prediction — Hybrid ML + DL Framework
=================================================
Account : 873166938412 | Region: us-east-1

ML:   Random Forest + XGBoost → predict init_duration_ms (cold start latency)
DL:   LSTM                    → predict invocation traffic patterns
Out:  Recommendation engine for optimal Lambda config + cost estimate

Usage:
  pip install pandas numpy scikit-learn xgboost matplotlib seaborn
  python train_models.py [--dataset cold_start_dataset.csv] [--output_dir model_outputs]
"""

import os
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                  # headless — no display required
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model      import Ridge
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.inspection        import permutation_importance

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ pip install xgboost  for XGBoost model")

try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    from tensorflow.keras.models    import Sequential
    from tensorflow.keras.layers    import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    HAS_TF = True
    print(f"✓ TensorFlow {tf.__version__}")
except ImportError:
    HAS_TF = False
    print("⚠ pip install tensorflow  for LSTM model")

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',    default='cold_start_dataset.csv')
parser.add_argument('--output_dir', default='model_outputs')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Features used for ML ──────────────────────────────────────────────────────
ML_FEATURES = [
    'memory_size_mb',
    'vpc_flag',
    'provisioned_flag',
    'container_flag',
    'hour_of_day',
    'day_of_week',
    'function_type_enc',
    'api_method_enc',
]
TARGET = 'init_duration_ms'

# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"✓ Loaded {len(df)} records from {path}")
    except FileNotFoundError:
        print(f"⚠ File not found: {path}")
        print("  → Generating synthetic data (realistic AWS Lambda distributions)")
        df = generate_synthetic()

    # Encode categoricals
    le = LabelEncoder()
    df['function_type_enc'] = le.fit_transform(df['function_type'].fillna('unknown'))
    df['api_method_enc']    = le.fit_transform(df.get('api_method', pd.Series(['GET']*len(df))).fillna('GET'))

    # Ensure numeric cols
    for col in ML_FEATURES + [TARGET, 'duration_ms', 'cold_start_flag']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

# =============================================================================
# SYNTHETIC DATA (realistic Lambda cold start distributions)
# =============================================================================
def generate_synthetic(n: int = 800) -> pd.DataFrame:
    """
    Based on published AWS Lambda benchmarks:
      Non-VPC Python 128MB : ~180-400ms cold start
      VPC     Python 128MB : ~700-1500ms cold start (ENI setup)
      Provisioned           : ~5-20ms  (warm handler only)
    """
    np.random.seed(42)
    configs = [
        {'function_type':'non-vpc',    'vpc_flag':0,'provisioned_flag':0,'base':280,'std':90},
        {'function_type':'vpc',        'vpc_flag':1,'provisioned_flag':0,'base':950,'std':250},
        {'function_type':'provisioned','vpc_flag':0,'provisioned_flag':1,'base':12, 'std':4},
    ]
    rows = []
    for _ in range(n):
        cfg = configs[np.random.choice(len(configs))]
        init = max(2.0, np.random.normal(cfg['base'], cfg['std']))
        rows.append({
            'function_type':    cfg['function_type'],
            'memory_size_mb':   128,
            'vpc_flag':         cfg['vpc_flag'],
            'provisioned_flag': cfg['provisioned_flag'],
            'container_flag':   0,
            'hour_of_day':      np.random.randint(0, 24),
            'day_of_week':      np.random.randint(0, 7),
            'api_method':       np.random.choice(['GET','POST','PUT','DELETE'], p=[0.5,0.3,0.1,0.1]),
            'init_duration_ms': round(init, 2),
            'duration_ms':      round(np.random.exponential(60) + 15, 2),
            'cold_start_flag':  1,
        })
    return pd.DataFrame(rows)

# =============================================================================
# ML MODELS
# =============================================================================
def train_ml(df: pd.DataFrame):
    print("\n" + "="*60)
    print("ML MODELS — Predicting Cold Start init_duration_ms")
    print("="*60)

    cold_df = df[df[TARGET] > 0].copy()
    if len(cold_df) < 20:
        print(f"  Only {len(cold_df)} cold start records → using synthetic data")
        cold_df = generate_synthetic(600)
        cold_df['function_type_enc'] = LabelEncoder().fit_transform(cold_df['function_type'])
        cold_df['api_method_enc']    = 0

    feats = [f for f in ML_FEATURES if f in cold_df.columns]
    X = cold_df[feats].values
    y = cold_df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    print(f"\n  Train: {len(Xtr)}  Test: {len(Xte)}  Features: {feats}")

    models = {
        'Random Forest':     RandomForestRegressor(n_estimators=300, max_depth=12,
                                                    min_samples_split=4, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                        max_depth=5, random_state=42),
        'Ridge Regression':  Ridge(alpha=1.0),
    }
    if HAS_XGB:
        models['XGBoost'] = xgb.XGBRegressor(n_estimators=400, learning_rate=0.04,
                                               max_depth=6, subsample=0.8,
                                               colsample_bytree=0.8, random_state=42,
                                               verbosity=0)

    results = {}
    best_mae  = float('inf')
    best_model = None

    for name, model in models.items():
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        mae   = mean_absolute_error(y_test, preds)
        rmse  = np.sqrt(mean_squared_error(y_test, preds))
        r2    = r2_score(y_test, preds)
        cv    = cross_val_score(model, Xtr, y_train, cv=5, scoring='neg_mean_absolute_error')
        results[name] = {'MAE_ms': round(mae,2), 'RMSE_ms': round(rmse,2),
                         'R2': round(r2,4), 'CV_MAE_ms': round(-cv.mean(),2)}
        print(f"\n  {name}:")
        print(f"    MAE={mae:.2f}ms  RMSE={rmse:.2f}ms  R²={r2:.4f}  CV_MAE={-cv.mean():.2f}ms")
        if mae < best_mae:
            best_mae   = mae
            best_model = (name, model)

    print(f"\n  🏆 Best model: {best_model[0]}  (MAE={best_mae:.2f}ms)")

    # Feature importance
    rf = models['Random Forest']
    fi = pd.DataFrame({'feature': feats, 'importance': rf.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print("\n  Feature Importances (Random Forest):")
    print(fi.to_string(index=False))

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Cold Start Prediction — ML Models (Account: 873166938412, us-east-1)', fontsize=12)

    # 1. Feature importance
    axes[0].barh(fi['feature'], fi['importance'], color='steelblue')
    axes[0].set_title('Feature Importance (RF)')
    axes[0].set_xlabel('Importance')
    axes[0].invert_yaxis()

    # 2. Model comparison
    res_df = pd.DataFrame(results).T[['MAE_ms','RMSE_ms']]
    res_df.plot(kind='bar', ax=axes[1], color=['coral','steelblue'], rot=30)
    axes[1].set_title('Model Comparison — Error (ms)')
    axes[1].set_ylabel('ms (lower = better)')

    # 3. Actual vs Predicted (best model)
    _, bm = best_model
    preds = bm.predict(Xte)
    axes[2].scatter(y_test, preds, alpha=0.5, color='steelblue', s=15)
    mn, mx = y_test.min(), y_test.max()
    axes[2].plot([mn,mx],[mn,mx], 'r--', linewidth=1)
    axes[2].set_title(f'Actual vs Predicted ({best_model[0]})')
    axes[2].set_xlabel('Actual init_duration_ms')
    axes[2].set_ylabel('Predicted init_duration_ms')

    plt.tight_layout()
    out = f"{args.output_dir}/ml_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved: {out}")

    return best_model[1], scaler, feats, results

# =============================================================================
# LSTM — Invocation Traffic Prediction
# =============================================================================
def train_lstm(df: pd.DataFrame, lookback: int = 12):
    if not HAS_TF:
        return None

    print("\n" + "="*60)
    print("DL MODEL — LSTM Invocation Traffic Prediction")
    print("="*60)

    if 'timestamp' not in df.columns or df['timestamp'].isna().all():
        print("  ⚠ No valid timestamps. Generating synthetic time-series.")
        n = 200
        t = np.arange(n)
        series = 50 + 30*np.sin(2*np.pi*t/24) + np.random.normal(0, 5, n)
    else:
        ts = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df2 = df.copy(); df2.index = ts
        resampled = df2['duration_ms'].resample('5min').count().fillna(0)
        series = resampled.values.astype(float)

    if len(series) < lookback + 10:
        print(f"  ⚠ Only {len(series)} time steps. Need more data. Generating synthetic.")
        n = 300
        t = np.arange(n)
        series = 50 + 30*np.sin(2*np.pi*t/24) + 15*np.sin(2*np.pi*t/168) + np.random.normal(0,5,n)

    scaler = StandardScaler()
    s = scaler.fit_transform(series.reshape(-1,1)).flatten()

    X, y = [], []
    for i in range(lookback, len(s)):
        X.append(s[i-lookback:i])
        y.append(s[i])
    X, y = np.array(X).reshape(-1, lookback, 1), np.array(y)

    split = int(len(X) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback,1)),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    cb = [EarlyStopping(patience=15, restore_best_weights=True),
          ReduceLROnPlateau(patience=7, factor=0.5, verbose=0)]
    hist = model.fit(Xtr, ytr, validation_data=(Xte,yte),
                     epochs=150, batch_size=16, callbacks=cb, verbose=0)

    preds = model.predict(Xte, verbose=0).flatten()
    mae   = mean_absolute_error(yte, preds)
    rmse  = np.sqrt(mean_squared_error(yte, preds))
    print(f"  LSTM  MAE={mae:.4f}  RMSE={rmse:.4f}  (scaled units)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(hist.history['loss'],    label='Train Loss')
    axes[0].plot(hist.history['val_loss'],label='Val Loss')
    axes[0].set_title('LSTM Training Loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE'); axes[0].legend()

    axes[1].plot(yte,  label='Actual',    alpha=0.7)
    axes[1].plot(preds,label='Predicted', alpha=0.7, linestyle='--')
    axes[1].set_title('LSTM: Actual vs Predicted Invocations')
    axes[1].set_xlabel('Time Step'); axes[1].legend()

    plt.tight_layout()
    out = f"{args.output_dir}/lstm_results.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved: {out}")
    return model

# =============================================================================
# RECOMMENDATION ENGINE
# =============================================================================
def recommend(model, scaler, feats: list, df: pd.DataFrame):
    print("\n" + "="*60)
    print("RECOMMENDATION ENGINE")
    print("="*60)
    print("  Account: 873166938412 | Memory tested: 128MB | Region: us-east-1")

    configs = [
        {'name':'Non-VPC 128MB (ZIP)',      'memory_size_mb':128,  'vpc_flag':0,'provisioned_flag':0,'container_flag':0},
        {'name':'VPC 128MB (ZIP)',           'memory_size_mb':128,  'vpc_flag':1,'provisioned_flag':0,'container_flag':0},
        {'name':'Provisioned 128MB',         'memory_size_mb':128,  'vpc_flag':0,'provisioned_flag':1,'container_flag':0},
        {'name':'Non-VPC 128MB (Container)', 'memory_size_mb':128,  'vpc_flag':0,'provisioned_flag':0,'container_flag':1},
        {'name':'Non-VPC 512MB (ZIP)',       'memory_size_mb':512,  'vpc_flag':0,'provisioned_flag':0,'container_flag':0},
        {'name':'VPC 512MB (ZIP)',           'memory_size_mb':512,  'vpc_flag':1,'provisioned_flag':0,'container_flag':0},
        {'name':'Non-VPC 1024MB (ZIP)',      'memory_size_mb':1024, 'vpc_flag':0,'provisioned_flag':0,'container_flag':0},
    ]

    rows = []
    for cfg in configs:
        row = {f: cfg.get(f, 0) for f in feats}
        row.setdefault('function_type_enc', 0)
        row.setdefault('api_method_enc', 0)
        row.setdefault('hour_of_day', 12)
        row.setdefault('day_of_week', 1)
        X_pred = scaler.transform([[row[f] for f in feats]])
        pred   = max(0, model.predict(X_pred)[0])

        # AWS Lambda pricing us-east-1
        # $0.0000166667 per GB-second + $0.20 per 1M requests
        mem_gb = cfg['memory_size_mb'] / 1024
        gb_sec = mem_gb * (pred / 1000)
        cost   = gb_sec * 0.0000166667 * 10000 + 0.20 * (10000/1e6)

        # Extra: Provisioned CC cost ($0.0000041667/GB-second × 2 instances × 3600s/hr)
        prov_cost_hr = (2 * mem_gb * 3600 * 0.0000041667) if cfg['provisioned_flag'] else 0

        rows.append({
            'Configuration':           cfg['name'],
            'Predicted Cold Start ms': round(pred, 1),
            'Cost per 10k calls ($)':  round(cost, 5),
            'ProvCC overhead ($/hr)':  round(prov_cost_hr, 4),
        })

    rec = pd.DataFrame(rows).sort_values('Predicted Cold Start ms')
    print("\n  Configurations sorted by predicted cold start:\n")
    print(rec.to_string(index=False))

    best = rec.iloc[0]
    print(f"\n  🏆 LOWEST COLD START: {best['Configuration']}")
    print(f"     Predicted: {best['Predicted Cold Start ms']} ms")
    print(f"     Cost per 10k calls: ${best['Cost per 10k calls ($)']}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ['gold' if i==0 else ('lightcoral' if 'VPC' in r else 'steelblue')
              for i, r in enumerate(rec['Configuration'])]
    ax.barh(rec['Configuration'], rec['Predicted Cold Start ms'], color=colors)
    ax.set_title('Predicted Cold Start by Config — 873166938412 / us-east-1 / 128MB')
    ax.set_xlabel('Predicted init_duration_ms (lower = better)')
    ax.invert_yaxis()
    for i, v in enumerate(rec['Predicted Cold Start ms']):
        ax.text(v + 5, i, f'{v}ms', va='center', fontsize=9)
    plt.tight_layout()
    out = f"{args.output_dir}/recommendations.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Saved: {out}")

    rec.to_csv(f"{args.output_dir}/recommendations.csv", index=False)
    return rec

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("COLD START ML/DL TRAINING PIPELINE")
    print("Account : 873166938412 | Region: us-east-1")
    print("=" * 60)

    df = load_data(args.dataset)

    # ML
    rf_model, scaler, feats, ml_results = train_ml(df)

    # DL
    lstm_model = train_lstm(df)

    # Recommend
    rec = recommend(rf_model, scaler, feats, df)

    # Save summary
    summary = {
        'account':            '873166938412',
        'region':             'us-east-1',
        'memory_mb':          128,
        'dataset_records':    len(df),
        'cold_start_records': int(df['cold_start_flag'].sum()) if 'cold_start_flag' in df.columns else 'N/A',
        'ml_model_results':   ml_results,
        'top_recommendation': rec.iloc[0].to_dict() if len(rec) > 0 else {},
        'output_files': [
            f"{args.output_dir}/ml_results.png",
            f"{args.output_dir}/lstm_results.png",
            f"{args.output_dir}/recommendations.png",
            f"{args.output_dir}/recommendations.csv",
        ]
    }
    summary_path = f"{args.output_dir}/training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ All done. Outputs in: {args.output_dir}/")
    print(f"   training_summary.json")
    print(f"   ml_results.png")
    print(f"   lstm_results.png")
    print(f"   recommendations.png  ← use in your research paper")
    print(f"   recommendations.csv")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
