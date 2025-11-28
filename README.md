# Height Prediction Model - Project 3

## Mission
Beat validation log2-MSE of 0.374 using TensorFlow deep learning.

## Key Improvements Implemented

### 1. Advanced Data Augmentation ✅
- **Problem**: Original combined dataset had 19x imbalance and missing DS-3
- **Solution**:
  - Properly merged DS-1, DS-2, and **DS-3 (NEW!)**
  - Applied domain-aware augmentation techniques
  - Target: 12,000 samples per (k,m) combination
- **Result**:
  - **108,000 perfectly balanced samples** (1.00x imbalance ratio)
  - Perfect 11.11% distribution across all 9 (k,m) groups
  - 132,851 raw samples → 108,000 optimally balanced samples

### 2. Log2 Prediction Architecture ✅
- **Problem**: Predicting raw m-height values (range: 3.7 to 180K) causes training instability
- **Solution**: Predict log2(m-height), then exponentiate to 2^x
- **Constraint**: All predictions ≥ 1.0 ensured via softplus activation

### 2. Advanced Model Architecture ✅
- **Total Parameters**: 10,875,041 (10.9M)
- **Architecture**:
  - Embedding layers for categorical k and m (32-dim each)
  - P matrix processing: Dense(256) → Dense(512) with BatchNorm
  - Multi-head attention (8 heads, 64-dim key) on P features
  - Deep network with 4 residual blocks (1024-dim)
  - Custom log2-MSE loss function
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

## Data Augmentation Strategy

### Augmentation Techniques
1. **Gaussian Noise Injection**: Preserves statistical properties
2. **Controlled Perturbations**: Domain-aware magnitude scaling
3. **SMOTE-like Interpolation**: Generate samples between similar instances
4. **Mixed Strategies**: Combine multiple techniques for diversity

### Source Datasets
- **DS-1**: 32,087 samples (24.2%)
- **DS-2**: 44,399 samples (33.4%)
- **DS-3**: 56,365 samples (42.4%) - **NEW!**
- **Total**: 132,851 raw samples

### Augmented Dataset
- Total samples: **108,000** (perfectly balanced)
- Samples per (k,m): **12,000** (exactly)
- Imbalance ratio: **1.00x** (perfect balance)
- Train/Val split: 91,800 / 16,200 (85% / 15%)
- Perfect stratification: Each group at 11.11%

## Files

### Scripts
- `augment_dataset.py` - **Data augmentation script (CPU-intensive, run first)**
- `train_model.py` - Training pipeline (GPU-intensive)
- `analyze_datasets.py` - Dataset analysis tool
- `inspect_data.py` - Data structure inspection utility

### Datasets
- `DS-1-samples_*.pkl` - Original dataset 1 (32K samples)
- `DS-2-Train_*.pkl` - Original dataset 2 (44K samples)
- `DS-3-Train_*.pkl` - **NEW dataset 3 (56K samples)**
- `augmented_*.pkl` - **Balanced augmented dataset (108K samples)**
- `combined_ALL_*.pkl` - Old combined dataset (deprecated)

### Outputs
- `best_model.h5` - Best model checkpoint
- `training_history.png` - Loss curves
- `predictions_scatter.png` - Prediction quality plots
- `per_group_performance.txt` - Per-(k,m) metrics

### Documentation
- `README.md` - This file
- `DATA_AUGMENTATION_PLAN.md` - **Detailed augmentation strategy**
- `requirements.txt` - Python dependencies

## Usage

### Step 1: Data Augmentation (Run Once)
```bash
# Install dependencies
pip install -r requirements.txt

# Run data augmentation (CPU-intensive, ~1-2 minutes)
python augment_dataset.py

# Output: augmented_n_k_m_P.pkl, augmented_mHeights.pkl
```

### Step 2: Train Model
```bash
# Train model (GPU-intensive, uses augmented data)
python train_model.py
```

### Optional: Analyze Datasets
```bash
# Analyze all datasets and see statistics
python analyze_datasets.py
```

## Expected Results

- **Target**: Validation log2-MSE < 0.374
- **Baseline (old combined)**: 0.374
- **Expected (with augmentation)**: 0.30-0.33 (12-20% improvement)
- **Stretch goal**: < 0.30

### Why Better Performance Expected?

1. **26% more training data**: 108K vs 85K samples
2. **Perfect balance**: 1.00x vs 19x imbalance ratio
3. **DS-3 inclusion**: 56K new samples (42.4% of total)
4. **Domain-aware augmentation**: Better than naive resampling
5. **Improved minority class coverage**: (4,5) and (5,4) now well-represented

## Success Criteria

✅ Perfectly balanced dataset (each group exactly 11.11%)
✅ All 3 source datasets included (DS-1, DS-2, DS-3)
✅ Domain-aware augmentation applied
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
