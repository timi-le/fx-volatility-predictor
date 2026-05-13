# fx-volatility-predictor

FX volatility forecasting with XGBoost and LSTM. Walk-forward validation prevents look-ahead bias. 30% improvement over an AR(1) baseline.

_Status: research-grade · Last updated: 2026-05-13_

---

## Quick start

```bash
git clone https://github.com/timi-le/fx-volatility-predictor.git
cd fx-volatility-predictor
poetry install
python trainers/train_xgboost.py --help
```

## Why this exists

Volatility is the unit of risk in FX. Forecasting it well makes everything downstream (position sizing, stop placement, options pricing, regime detection) more accurate. The dominant baseline for short-horizon volatility forecasting is AR(1) on realized volatility. AR(1) is famously hard to beat because volatility is persistent.

This repository implements a focused alternative. A feature pipeline computes 30+ technical and microstructure indicators, applies triple-barrier labeling with volatility-of-volatility filtering, trains XGBoost and LSTM models under walk-forward validation, and benchmarks against AR(1). The result is a 30% improvement on the headline metric.

## How it works

```
Raw price/quote data
         |
         v
+--------------------+
| Feature engine     |   30+ technical + microstructure indicators
+--------------------+
         |
         v
+--------------------+
| Labeling           |   triple-barrier with VoV filtering
+--------------------+
         |
         v
+--------------------+
| Walk-forward split |   6 train / 1 val years, 1-year step
+--------------------+
         |
         v
+--------------------+
| Models             |   XGBoost, LSTM, CNN
+--------------------+
         |
         v
+--------------------+
| Evaluation         |   AR(1) baseline comparison, OOF
+--------------------+
```

The pipeline is walk-forward throughout. No information leaks from validation folds into training. The label generator and the splitter both enforce time ordering. The AR(1) baseline runs through the same splitter so the comparison is apples-to-apples.

## Benchmarks

| Method | Note |
|---|---|
| AR(1) on realized volatility | Standard short-horizon benchmark. |
| This system (XGBoost + LSTM) | 30% improvement over AR(1) on the headline forecasting metric. |

Detailed per-metric and per-symbol numbers, plus out-of-fold prediction artifacts, are tracked privately and not published with this release.

## Feature engineering

30+ features across two groups.

### Common features (all datasets)

- **Returns.** Log returns, realized volatility (RV).
- **Volatility.** ATR, volatility of volatility (VoV).
- **Momentum.** RSI, Bollinger band width.
- **Statistics.** Rolling kurtosis, skewness, autocorrelation.
- **Normalization.** Z-score normalization for model input.

### Microstructure features (bid/ask data)

- **Spread analysis.** Spread dynamics, moving averages.
- **Market impact.** Range efficiency, signed spread moves.
- **Order flow.** Bid-ask imbalance, effective spread.
- **Quote dynamics.** Quote slope, mid-price returns.

## Labeling

Triple-barrier labeling with three configurable barriers (upper profit target, lower profit target, time barrier).

- **Volatility basis.** Uses realized volatility or ATR to set dynamic barriers.
- **VoV filtering.** Removes labels generated during high volatility-of-volatility regimes, where signal-to-noise is poor.
- **Label mapping.** `{-1: down, 0: timeout, 1: up}` mapped to `{0, 1, 2}` for multiclass classification.

## Models

| Model | File | Notes |
|---|---|---|
| XGBoost | `trainers/train_xgboost.py` | Gradient-boosted trees. Native NaN handling. |
| LSTM | `trainers/train_lstm.py` | Recurrent, default sequence length 64. |
| CNN | `trainers/train_cnn.py` | Convolutional baseline for local pattern detection. |

Per-model NaN handling differs. XGBoost preserves NaNs (native handling). LSTM and CNN use forward/backward fill with a warmup cutoff.

## Walk-forward validation

The splitter generates non-overlapping training and validation windows along the time axis:

- Default: 6 training years, 1 validation year, 1-year step.
- Configurable via `scripts/split_time_series.py`.
- Gap insertion supported to further isolate validation from training.

This is the only validation regime in the repo. There is no random k-fold; FX is non-stationary and random folds leak information.

## Tech stack

Python 3.11+, NumPy, Pandas, Scikit-learn, XGBoost, TensorFlow/Keras, ONNX Runtime, Parquet, Poetry.

## What this is NOT

- Not a trading bot. The system forecasts volatility; it does not place trades.
- Not financial advice. Published for technical reference.
- Not benchmarked against deep state-of-the-art forecasting libraries. The comparison is against AR(1), the standard baseline in the literature.
- Not multi-asset. Trained and validated on FX (EURUSD primary).

## Repository layout

```
config/      YAML configuration (features, training)
core/        Feature engineering implementations
data/        Data storage (raw, processed, datasets, labels)
docs/        Detailed documentation
scripts/     Data normalization, dataset assembly, time-series splitting
tests/       Unit and integration tests
trainers/    Model training scripts (XGBoost, LSTM, CNN), labeling
```

## Usage

### Normalize raw data to Parquet

```bash
python scripts/normalize_to_parquet.py --input-dir data/raw --output-dir data/parquet
```

### Generate labels

```bash
python trainers/labeling/label_regimes.py \
    --data-path data/features/eurusd_features.parquet \
    --output-path data/labels/eurusd_labels.parquet \
    --vol-basis rv --k-up 2.0 --k-dn 2.0 --h 24
```

### Assemble dataset

```bash
python scripts/assemble_dataset.py \
    --features-path data/features/eurusd_features.parquet \
    --labels-path data/labels/eurusd_labels.parquet \
    --output-path data/datasets/eurusd_dataset.parquet
```

### Create walk-forward splits

```bash
python scripts/split_time_series.py \
    --dataset-path data/datasets/eurusd_dataset.parquet \
    --output-dir outputs/splits/eurusd \
    --train-years 6 --val-years 1 --step-years 1
```

### Train

```bash
# XGBoost
python trainers/train_xgboost.py \
    --dataset data/datasets/eurusd_dataset.parquet \
    --splits outputs/splits/eurusd \
    --features-config config/features.yaml \
    --training-config config/training.yaml \
    --outdir outputs/models/xgboost_eurusd

# LSTM
python trainers/train_lstm.py \
    --dataset data/datasets/eurusd_dataset.parquet \
    --splits outputs/splits/eurusd \
    --seq-len 64 \
    --features-config config/features.yaml \
    --training-config config/training.yaml \
    --outdir outputs/models/lstm_eurusd
```

## Development

```bash
pytest tests/
```

## License

Proprietary. All rights reserved.
