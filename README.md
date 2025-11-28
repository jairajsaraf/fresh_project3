# Height Prediction Model - Project 3

## Mission
Beat validation log2-MSE of 0.374 using TensorFlow deep learning.

## Key Improvements Implemented

### 1. Dataset Rebalancing ✅
- **Problem**: Severe 19x class imbalance (k=4,m=2: 52.88% vs k=4,m=5: 2.78%)
- **Solution**: Resampled to exactly 9,000 samples per (k,m) combination
- **Result**: Perfect 11.11% distribution across all 9 groups

### 2. Log2 Prediction Architecture ✅
- **Problem**: Predicting raw m-height values (range: 3.7 to 180K) causes training instability
- **Solution**: Predict log2(m-height), then exponentiate to 2^x
- **Constraint**: All predictions ≥ 1.0 ensured via softplus activation

### 3. Advanced Model Architecture ✅
- **Total Parameters**: 10,875,041 (10.9M)
- **Architecture**:
  - Embedding layers for categorical k and m (32-dim each)
  - P matrix processing: Dense(256) → Dense(512) with BatchNorm
  - Multi-head attention (8 heads, 64-dim key) on P features
  - Deep network with 4 residual blocks (1024-dim)
  - Custom log2-MSE loss function
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Dataset Statistics

### Before Rebalancing
- Total samples: 85,499
- Imbalance ratio: 19.0x
- Most common: k=4, m=2 (52.88%)
- Least common: k=4, m=5 (2.78%)

### After Rebalancing
- Total samples: 81,000 (9,000 per group)
- Train/Val split: 68,850 / 12,150 (85% / 15%)
- Perfect stratification: Each group at 11.11%

## Files

- `train_model.py` - Complete training pipeline
- `best_model.h5` - Best model checkpoint
- `training_history.png` - Loss curves (train vs validation)
- `predictions_scatter.png` - Prediction quality visualization
- `per_group_performance.txt` - Detailed per-(k,m) metrics
- `combined_ALL_n_k_m_P_exact.pkl` - Input features
- `combined_ALL_mHeights_exact.pkl` - Target outputs

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train_model.py
```

## Expected Results

- **Target**: Validation log2-MSE < 0.374
- **Baseline**: 0.374
- **Expected**: 0.30-0.35 (15-20% improvement)
- **Stretch goal**: < 0.30

## Success Criteria

✅ Balanced dataset (each group ~11%)
✅ Log2 prediction architecture
✅ All predictions ≥ 1.0
✅ Stratified validation split
✅ Per-group performance analysis

## Model Output Layer (Critical)

```python
# Predict log2(m-height)
log2_pred = Dense(1, activation='linear')(x)

# Ensure log2_pred ≥ 0 (so m-height ≥ 1)
log2_positive = Lambda(lambda x: tf.nn.softplus(x))(log2_pred)

# Convert to m-height: 2^(log2_pred)
output = Lambda(lambda x: tf.pow(2.0, x))(log2_positive)
```

## Custom Loss Function

```python
def log2_mse_loss(y_true, y_pred):
    epsilon = 1e-7
    y_true = tf.maximum(y_true, epsilon)
    y_pred = tf.maximum(y_pred, epsilon)

    log2_true = tf.math.log(y_true) / tf.math.log(2.0)
    log2_pred = tf.math.log(y_pred) / tf.math.log(2.0)

    return tf.reduce_mean(tf.square(log2_true - log2_pred))
```
