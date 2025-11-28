# Run 1 Training Analysis and Recommendations

**Date**: 2025-11-28
**Model Performance**: Val Log2-MSE = 0.495 (Target: 0.374)
**Performance Gap**: 32.4% worse than target

---

## Executive Summary

The model achieved **0.495 validation log2-MSE**, failing to beat the 0.374 target by 32.4%. While the model performs excellently on common (k,m) combinations (k=4,m=2: 0.103), it **catastrophically fails on high-complexity groups** (k=4,m=5: 1.386 - a 13.5x performance gap).

**Critical Finding**: The model exhibits severe overfitting (train loss 0.2 vs val loss 0.495) and extreme per-group performance variance, indicating fundamental architectural and training issues despite using the properly augmented and balanced dataset.

---

## 1. Detailed Performance Breakdown

### 1.1 Per-Group Validation Log2-MSE

| k,m Group | Log2-MSE | vs Target | Relative Performance |
|-----------|----------|-----------|---------------------|
| k=4, m=2  | **0.103** | -72.5% ✓  | **Excellent** (baseline) |
| k=4, m=3  | **0.106** | -71.7% ✓  | **Excellent** |
| k=5, m=2  | **0.107** | -71.4% ✓  | **Excellent** |
| k=4, m=4  | 0.215    | -42.5% ✓  | Good |
| k=5, m=3  | 0.224    | -40.1% ✓  | Good |
| k=6, m=2  | 0.361    | -3.5% ✓   | Acceptable |
| k=5, m=4  | **0.976** | +161% ✗   | **Poor** (9.5x vs k=4,m=2) |
| k=6, m=3  | **0.981** | +162% ✗   | **Poor** (9.5x vs k=4,m=2) |
| k=4, m=5  | **1.386** | +270% ✗   | **Failed** (13.5x vs k=4,m=2) |

**Pattern Identified**:
- Performance degrades **exponentially** as `m` increases (especially m ≥ 4)
- Performance degrades **linearly** as `k` increases
- The combination of high k AND high m is catastrophic (k=6,m=3 and k=4,m=5)

### 1.2 Training Dynamics

**From training_history.png analysis**:
- **Initial phase (epochs 0-40)**: Rapid improvement, both train and val loss decrease together
- **Mid phase (epochs 40-100)**: Train loss continues decreasing, val loss plateaus at ~0.6-0.7
- **Late phase (epochs 100-200)**: Severe overfitting evident
  - Train loss: 0.59 → 0.20 (67% improvement)
  - Val loss: 0.60 → 0.50 (17% improvement, high variance)
  - **Generalization gap**: 2.5x difference by epoch 200

**From predictions_scatter.png analysis**:
- Log-scale plot shows reasonable overall correlation
- Significant scatter for m-height > 10^5 (corresponds to high k,m groups)
- Predictions tend to **cluster** around certain values rather than spanning the full range
- Some predictions severely underestimate true values (visible outliers below diagonal)

---

## 2. Root Cause Analysis

### 2.1 ✓ NOT the Problem: Dataset Imbalance

**Evidence**: The model used `augmented_n_k_m_P.pkl` (108,000 perfectly balanced samples with 12,000 per group). The original 19x imbalance has been eliminated.

**Conclusion**: Dataset balance is not the issue.

### 2.2 ✗ CRITICAL ISSUE: Augmentation Quality for Complex Groups

**Hypothesis**: The augmentation techniques (Gaussian noise, perturbations, SMOTE interpolation) may generate **unrealistic P matrices** for high-complexity groups (large k,m values).

**Evidence**:
1. Original DS-1/DS-2/DS-3 had very few samples for k=4,m=5 (minority class)
2. To reach 12,000 samples, we needed ~95% augmented data for these groups
3. SMOTE interpolation between two P matrices with different structures may violate domain constraints
4. Gaussian noise on sparse/structured matrices may destroy critical patterns

**Why this matters**:
- P matrices for k=4,m=5 have dimensions based on n (population size)
- The relationship between P matrix structure and m-height is complex
- Random interpolation/perturbation likely breaks these relationships

### 2.3 ✗ CRITICAL ISSUE: Model Architecture Limitations

**Problem**: The model uses a **single shared architecture** for all (k,m) combinations, but the complexity varies dramatically:

| Group | State Space Size | Computation Required |
|-------|-----------------|---------------------|
| k=4,m=2 | ~O(n²) | Low |
| k=4,m=5 | ~O(n^5) | **1000x higher** |
| k=6,m=3 | ~O(n^3) with 6 alleles | **Very high** |

**Current architecture**:
- Same 8-head attention mechanism for all groups
- Same 4 residual blocks (1024 dims) for all groups
- Only k,m embeddings differentiate between groups (2 × 32 = 64 dims out of 10.9M params)

**Why this fails**: High-complexity groups need:
- More attention heads to capture complex interactions
- Deeper/wider residual blocks for feature extraction
- Specialized processing pathways

### 2.4 ✗ MAJOR ISSUE: Severe Overfitting

**Evidence**:
- Train loss: 0.20 (excellent)
- Val loss: 0.495 (poor)
- Gap: 2.5x difference

**Root causes**:
1. **Model capacity mismatch**: 10.9M parameters may be too large for effective data (even with augmentation, high-k,m groups have mostly synthetic data)
2. **Insufficient regularization**: Current setup:
   - Dropout: 0.3 (moderate)
   - Weight decay: 1e-4 (weak)
   - No data augmentation during training (only pre-training)
3. **Training dynamics**: Learning rate 1e-3 may be too aggressive

### 2.5 ✗ MODERATE ISSUE: Loss Function Limitation

**Current**: Uniform log2-MSE across all (k,m) groups

**Problem**: The loss treats all groups equally, so the model optimizes for the majority of samples. With equal group sizes, the model still learns patterns that work well for "easy" groups and ignores "hard" groups.

**Why**: The intrinsic difficulty varies dramatically:
- k=4,m=2: Relatively simple relationship, achieves 0.103 easily
- k=4,m=5: Extremely complex relationship, current loss doesn't force the model to work harder

---

## 3. Recommendations

### 3.1 PRIORITY 1: Implement Group-Specific Loss Weighting

**Objective**: Force the model to focus more on difficult (k,m) groups

**Implementation**:

```python
# In train_model.py, modify the loss function

def adaptive_log2_mse_loss(y_true, y_pred):
    """
    Weighted log2-MSE loss that penalizes high-complexity groups more
    """
    epsilon = 1e-7
    y_true = tf.maximum(y_true, epsilon)
    y_pred = tf.maximum(y_pred, epsilon)

    log2_true = tf.math.log(y_true) / tf.math.log(2.0)
    log2_pred = tf.math.log(y_pred) / tf.math.log(2.0)

    squared_error = tf.square(log2_true - log2_pred)

    # Define per-group weights based on complexity
    # Higher k and m -> higher weight
    group_weights = {
        (4, 2): 1.0,   # baseline
        (4, 3): 1.0,
        (5, 2): 1.0,
        (4, 4): 1.5,   # increase focus
        (5, 3): 1.5,
        (6, 2): 2.0,   # significant focus
        (5, 4): 3.0,   # high focus
        (6, 3): 3.0,
        (4, 5): 5.0,   # maximum focus on worst performer
    }

    # Apply weights (requires sample_weight parameter in fit())
    return tf.reduce_mean(squared_error)  # Weights applied via fit()
```

**Training modification**:
```python
# Compute sample weights based on (k,m) groups
sample_weights = np.array([
    group_weights.get((k, m), 1.0)
    for k, m in zip(k_values_train, m_values_train)
])

# Pass to fit()
history = model.fit(
    [n_train, k_train, m_train, P_train],
    mheight_train,
    sample_weight=sample_weights,  # Add this
    ...
)
```

**Expected Impact**: 15-25% improvement on worst groups, 5-10% degradation on best groups (acceptable trade-off)

### 3.2 PRIORITY 2: Implement Complexity-Aware Architecture

**Objective**: Allow the model to allocate different capacity to different (k,m) groups

**Implementation**:

```python
def build_complexity_aware_model(n_max, max_P_size, k_values, m_values):
    """
    Model with dynamic capacity based on k,m complexity
    """
    # Inputs (same as before)
    n_input = Input(shape=(1,), name='n_input')
    k_input = Input(shape=(1,), name='k_input')
    m_input = Input(shape=(1,), name='m_input')
    P_input = Input(shape=(max_P_size,), name='P_input')

    # Embeddings (same as before)
    k_embed = Embedding(len(k_values), 32, name='k_embedding')(k_input)
    k_embed = Flatten()(k_embed)
    m_embed = Embedding(len(m_values), 32, name='m_embedding')(m_input)
    m_embed = Flatten()(m_embed)

    # Compute complexity score: k * log2(m)
    # This is a simplified proxy for computational complexity
    complexity_input = Concatenate()([k_embed, m_embed])
    complexity_score = Dense(1, activation='sigmoid', name='complexity_gate')(complexity_input)

    # Process P matrix with standard pathway
    P_reshaped = Reshape((max_P_size // 64, 64))(P_input)

    # STANDARD PATHWAY (for all groups)
    attn_standard = MultiHeadAttention(
        num_heads=8, key_dim=64, name='standard_attention'
    )(P_reshaped, P_reshaped)
    attn_standard = GlobalAveragePooling1D()(attn_standard)

    # HIGH-COMPLEXITY PATHWAY (activated for difficult groups)
    attn_complex = MultiHeadAttention(
        num_heads=16, key_dim=128, name='complex_attention'  # More capacity
    )(P_reshaped, P_reshaped)
    attn_complex = GlobalAveragePooling1D()(attn_complex)

    # Gated combination based on complexity score
    # complexity_score close to 0 -> use standard pathway
    # complexity_score close to 1 -> use complex pathway
    attn_output = complexity_score * attn_complex + (1 - complexity_score) * attn_standard

    # Combine all features
    combined = Concatenate()([n_input, k_embed, m_embed, attn_output])

    # Shared residual blocks (same as before)
    x = Dense(1024, activation='relu')(combined)
    x = Dropout(0.4)(x)  # Increased from 0.3

    for i in range(4):
        residual = x
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Add()([x, residual])

    # Output (same as before)
    log2_pred = Dense(1, activation='linear', name='log2_prediction')(x)
    log2_positive = Lambda(lambda x: tf.nn.softplus(x), name='softplus_activation')(log2_pred)
    output = Lambda(lambda x: tf.pow(2.0, x), name='m_height_output')(log2_positive)

    model = Model(
        inputs=[n_input, k_input, m_input, P_input],
        outputs=output,
        name='complexity_aware_model'
    )

    return model
```

**Expected Impact**: 20-35% improvement on high-complexity groups

### 3.3 PRIORITY 3: Improve Augmentation Quality

**Objective**: Generate more realistic augmented samples for minority classes

**Implementation** (modify `augment_dataset.py`):

```python
def domain_aware_interpolate(sample1, sample2, alpha=None):
    """
    Improved interpolation that preserves P matrix properties
    """
    if alpha is None:
        alpha = np.random.uniform(0.3, 0.7)

    P1 = sample1[3]
    P2 = sample2[3]

    # Method 1: Geometric mean (better for positive matrices)
    P_new = np.power(np.power(P1, alpha) * np.power(P2, 1-alpha), 1.0)

    # Method 2: Preserve row/column structure
    # (Only if P matrices have similar structure)
    if P1.shape == P2.shape:
        # Interpolate row sums separately to preserve structure
        row_sums_1 = P1.sum(axis=1, keepdims=True)
        row_sums_2 = P2.sum(axis=1, keepdims=True)

        P1_normalized = P1 / (row_sums_1 + 1e-10)
        P2_normalized = P2 / (row_sums_2 + 1e-10)

        P_new_normalized = alpha * P1_normalized + (1-alpha) * P2_normalized
        row_sums_new = alpha * row_sums_1 + (1-alpha) * row_sums_2

        P_new = P_new_normalized * row_sums_new

    return [sample1[0], sample1[1], sample1[2], P_new]

def validate_augmented_sample(sample, original_samples):
    """
    Ensure augmented sample is realistic by comparing to originals
    """
    P_aug = sample[3]
    k, m = sample[1], sample[2]

    # Find original samples from same (k,m) group
    similar_samples = [s for s in original_samples if s[1] == k and s[2] == m]

    if len(similar_samples) == 0:
        return True  # No validation possible

    # Check if statistics match original distribution
    P_aug_mean = np.mean(P_aug)
    P_aug_std = np.std(P_aug)

    orig_means = [np.mean(s[3]) for s in similar_samples]
    orig_stds = [np.std(s[3]) for s in similar_samples]

    mean_range = (np.min(orig_means) * 0.8, np.max(orig_means) * 1.2)
    std_range = (np.min(orig_stds) * 0.8, np.max(orig_stds) * 1.2)

    if not (mean_range[0] <= P_aug_mean <= mean_range[1]):
        return False
    if not (std_range[0] <= P_aug_std <= std_range[1]):
        return False

    return True
```

**Expected Impact**: 10-20% improvement on minority classes

### 3.4 PRIORITY 4: Strengthen Regularization

**Objective**: Reduce overfitting (train 0.2 vs val 0.495)

**Implementation**:

```python
# 1. Increase dropout
Dropout(0.4)  # up from 0.3

# 2. Increase weight decay
optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-3)  # up from 1e-4

# 3. Add L2 regularization to dense layers
from tensorflow.keras.regularizers import l2
Dense(1024, activation='relu', kernel_regularizer=l2(1e-4))

# 4. Reduce learning rate
optimizer = AdamW(learning_rate=5e-4, weight_decay=1e-3)  # down from 1e-3

# 5. More aggressive early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=30,  # down from 50
    restore_best_weights=True,
    verbose=1
)

# 6. Data augmentation during training (on-the-fly)
def augment_batch(n, k, m, P):
    """Apply light augmentation during training"""
    noise = tf.random.normal(P.shape, mean=0.0, stddev=0.02)
    P_augmented = P + noise * tf.math.reduce_std(P)
    return n, k, m, P_augmented
```

**Expected Impact**: 15-25% reduction in generalization gap

### 3.5 PRIORITY 5: Curriculum Learning

**Objective**: Train on easy examples first, gradually introduce difficult ones

**Implementation**:

```python
def curriculum_training(model, X_train, y_train, X_val, y_val, k_train, m_train):
    """
    Train with curriculum: start with easy (k,m) groups, gradually add harder ones
    """
    # Define difficulty tiers
    tier_1 = [(4,2), (4,3), (5,2)]  # Easy groups
    tier_2 = [(4,4), (5,3), (6,2)]  # Medium groups
    tier_3 = [(5,4), (6,3), (4,5)]  # Hard groups

    # Phase 1: Train on easy groups only (50 epochs)
    print("Phase 1: Training on easy groups...")
    mask_1 = [any((k==k_t and m==m_t) for k_t, m_t in tier_1)
              for k, m in zip(k_train, m_train)]
    X_train_tier1 = [x[mask_1] for x in X_train]
    y_train_tier1 = y_train[mask_1]

    model.fit(X_train_tier1, y_train_tier1, epochs=50, ...)

    # Phase 2: Add medium groups (50 epochs)
    print("Phase 2: Adding medium groups...")
    mask_2 = [any((k==k_t and m==m_t) for k_t, m_t in tier_1+tier_2)
              for k, m in zip(k_train, m_train)]
    X_train_tier2 = [x[mask_2] for x in X_train]
    y_train_tier2 = y_train[mask_2]

    model.fit(X_train_tier2, y_train_tier2, epochs=50, ...)

    # Phase 3: All groups (100 epochs)
    print("Phase 3: Training on all groups...")
    model.fit(X_train, y_train, epochs=100, ...)

    return model
```

**Expected Impact**: 10-15% improvement on hard groups

---

## 4. Recommended Action Plan

### Immediate Actions (Run 2)

**Focus**: Quick wins with minimal code changes

1. **Implement group-weighted loss** (Priority 1)
   - Estimated time: 30 minutes
   - Expected improvement: 15-20%

2. **Strengthen regularization** (Priority 4)
   - Estimated time: 15 minutes
   - Expected improvement: 10-15%

3. **Reduce learning rate**
   - Change: 1e-3 → 5e-4
   - Estimated time: 2 minutes

**Expected Run 2 result**: Val log2-MSE ~0.38-0.40 (within or near target)

### Medium-term Actions (Run 3)

**Focus**: Architectural improvements

1. **Implement complexity-aware architecture** (Priority 2)
   - Estimated time: 2-3 hours
   - Expected improvement: 20-30%

2. **Improve augmentation quality** (Priority 3)
   - Estimated time: 1-2 hours
   - Expected improvement: 10-15%

**Expected Run 3 result**: Val log2-MSE ~0.30-0.35 (significantly better than target)

### Advanced Actions (Run 4+)

**Focus**: Sophisticated techniques

1. **Implement curriculum learning** (Priority 5)
2. **Ensemble models** (train 3-5 models with different random seeds, average predictions)
3. **Group-specific models** (separate models for low/medium/high complexity groups)

**Expected Run 4+ result**: Val log2-MSE ~0.25-0.30 (exceptional performance)

---

## 5. Expected Performance Improvements

| Intervention | Impact on Worst Groups | Impact on Overall | Impact on Best Groups |
|--------------|------------------------|-------------------|----------------------|
| Group-weighted loss | +25-35% | +15-20% | -5-10% (acceptable) |
| Complexity-aware arch | +30-40% | +20-25% | +0-5% |
| Better augmentation | +15-25% | +10-15% | +0-5% |
| Stronger regularization | +10-15% | +12-18% | +5-10% |
| Curriculum learning | +15-20% | +10-12% | +0-5% |

**Cumulative expected improvement** (Run 2 + Run 3):
- Worst groups (k=4,m=5): 1.386 → 0.55-0.65 (60-65% improvement)
- Overall validation: 0.495 → 0.32-0.36 (14-35% below target ✓)
- Best groups (k=4,m=2): 0.103 → 0.10-0.12 (stable or slight improvement)

---

## 6. Risk Assessment

### High Risk
- **Over-regularization**: Too much dropout/weight decay may hurt best-performing groups
  - Mitigation: Monitor per-group performance during training, back off if k=4,m=2 degrades significantly

### Medium Risk
- **Complexity-aware architecture complexity**: Gating mechanism may not learn properly
  - Mitigation: Start with fixed gates based on k,m values, then move to learned gates

- **Augmentation improvements may be insufficient**: Domain constraints are unknown
  - Mitigation: Consider collecting more real data for minority classes if possible

### Low Risk
- **Group-weighted loss**: Well-established technique, unlikely to cause issues
- **Learning rate reduction**: Conservative change, very safe

---

## 7. Monitoring Plan for Run 2

Track these metrics to validate improvements:

1. **Overall validation log2-MSE**: Target < 0.374
2. **Per-group validation log2-MSE**: All groups < 0.60
3. **Worst group performance**: k=4,m=5 target < 0.80 (currently 1.386)
4. **Train-val gap**: Target < 1.5x (currently 2.5x)
5. **Training stability**: Val loss should decrease smoothly without high variance

**Success criteria for Run 2**:
- Overall val log2-MSE < 0.40 (7% margin from target)
- k=4,m=5 log2-MSE < 0.90 (35% improvement)
- Train-val gap < 2.0x (20% improvement)
- No group worse than 1.0 log2-MSE

If Run 2 doesn't meet these criteria, proceed directly to Run 3 with architectural changes.

---

## 8. Code Modifications Summary

### File: `train_model.py`

**Changes needed**:
1. Line ~180: Change learning rate from 1e-3 to 5e-4
2. Line ~190: Change weight decay from 1e-4 to 1e-3
3. Line ~150-160: Change all Dropout(0.3) to Dropout(0.4)
4. Line ~200: Change early stopping patience from 50 to 30
5. Line ~120-140: Add sample weight computation based on (k,m) groups
6. Line ~210: Add sample_weight parameter to model.fit()

**Testing**:
- Run training for 10 epochs to verify no errors
- Check that sample weights are being applied correctly (loss should be higher initially)
- Verify per-group loss computation in custom callback

---

## Conclusion

The current model's failure to meet the 0.374 target is primarily due to:

1. **Lack of focus on difficult groups** (uniform loss weighting)
2. **Insufficient regularization** (severe overfitting)
3. **Single-pathway architecture** (can't handle varying complexity)
4. **Potentially low-quality augmentation** for minority classes

The recommended two-phase approach (Run 2: quick wins, Run 3: architectural improvements) should reliably achieve the target with high confidence. The analysis provides clear, actionable steps with expected outcomes.

**Recommended immediate next step**: Implement Run 2 changes (group-weighted loss + stronger regularization) and retrain. This should get us to ~0.38-0.40 val log2-MSE with minimal risk.
