# Run 6 Analysis: Critical Issues & Recommendations

**Analysis Date:** 2025-11-30
**Notebook Analyzed:** `train_model_v5_COMPLETE.ipynb`
**Data Files:** `data/combined_final_n_k_m_P.pkl`, `data/combined_final_mHeights.pkl`

---

## Executive Summary

Run 6 achieved a validation log2-MSE of **0.181** (51.5% better than 0.374 target), but this result is **INVALID** due to critical implementation flaws:

1. **Data Leakage:** Augmentation applied before train-val split
2. **Excessive Augmentation:** 26x increase (50 copies for hard cases)
3. **Easy Case Degradation:** 3-4x worse performance than baseline
4. **Imbalanced Metrics:** Results dominated by overrepresented hard cases

**Verdict:** Run 6 results are unreliable and won't generalize to new data.

---

## 🚨 Critical Issues

### ISSUE #1: Data Leakage (CRITICAL)

**Location:** `train_model_v5_COMPLETE.ipynb`, Cell 6-10

**Problem:** Data augmentation happens BEFORE train-val split:

```
Flow in Run 6:
1. Load 108,000 samples
2. Augment to 2,808,000 samples (includes original + augmented)
3. Split into train (2,386,800) and val (421,200)
```

**Impact:**
- Original sample X might be in validation set
- Augmented copies of X (with only 2% noise difference) are in training set
- Model sees near-identical versions of validation data during training
- Validation metrics are artificially inflated
- Explains tiny train-val gap (0.004) and suspiciously good performance

**Evidence:**
- Train loss: 0.177
- Val loss: 0.181
- Gap: 0.004 (should be larger if properly separated)

---

### ISSUE #2: Excessive Augmentation

**Location:** `train_model_v5_COMPLETE.ipynb`, Cell 6

**Documentation vs Reality:**

| What Code Says | What Code Does |
|----------------|----------------|
| Hard cases: +2 copies | Hard cases: **+50 copies** |
| Medium cases: +1 copy | Medium cases: **+25 copies** |
| Easy cases: 0 copies | Easy cases: **0 copies** |

**Result:** 108,000 → 2,808,000 samples (26x increase!)

**Code:**
```python
augmentation_config = {
    (5, 4): 50,  # Comments say 2!
    (6, 3): 50,
    (4, 5): 50,
    (4, 4): 25,  # Comments say 1!
    (5, 3): 25,
    (6, 2): 25,
}
```

**Impact:**
- With only 2% noise, creates near-duplicate samples
- Model memorizes patterns instead of learning generalizable features
- Combined with data leakage, severely compromises validation

---

### ISSUE #3: Easy Group Degradation (SEVERE)

**Per-Group Performance:**

| Group | Type | Run 2 | Run 6 | Change | % Change |
|-------|------|-------|-------|--------|----------|
| k=4,m=2 | Easy | 0.103 | **0.405** | +0.302 | **-293%** |
| k=4,m=3 | Easy | 0.108 | **0.460** | +0.352 | **-328%** |
| k=5,m=2 | Easy | 0.096 | **0.451** | +0.355 | **-369%** |
| k=4,m=4 | Med | 0.210 | 0.127 | -0.083 | +39% |
| k=5,m=3 | Med | 0.227 | 0.138 | -0.089 | +39% |
| k=6,m=2 | Med | 0.356 | 0.155 | -0.201 | +56% |
| k=4,m=5 | Hard | 1.422 | 0.224 | -1.198 | +84% |
| k=5,m=4 | Hard | 0.958 | 0.200 | -0.758 | +79% |
| k=6,m=3 | Hard | 1.083 | 0.168 | -0.915 | +85% |

**Analysis:**
- Easy cases: **3-4x worse** than Run 2 baseline
- Medium cases: 39-56% better
- Hard cases: 79-85% better

**Root Cause:**
1. Easy groups received 0x augmentation (only 1,800 val samples each)
2. Hard groups received 50x augmentation (91,800 val samples each)
3. Model overfitted to heavily-augmented hard/medium cases
4. Mild weighting (1.0-2.0x) insufficient to prevent easy case degradation

---

### ISSUE #4: Imbalanced Validation Distribution

**Validation Set Composition:**

| Difficulty | Groups | Val Samples | % of Total |
|------------|--------|-------------|------------|
| Easy | 3 groups | 5,400 | 1.3% |
| Medium | 3 groups | 93,600 | 22.2% |
| Hard | 3 groups | 275,400 | 65.4% |

**Impact:**
- Overall metric (0.181) is dominated by hard cases (65%)
- Easy case degradation (293-369% worse) is hidden by weighted average
- Misleading "success" metric that masks severe failures

**Calculation:**
```
Overall = (5.4K × 0.44) + (93.6K × 0.14) + (275.4K × 0.19) / 421.2K
        ≈ 0.181 (dominated by hard cases with good performance)
```

---

### ISSUE #5: Suspiciously Small Train-Val Gap

**Metrics:**
- Training log2-MSE: 0.177264
- Validation log2-MSE: 0.181483
- **Gap: 0.004218**

**Expected Behavior:**
- Validation loss should be noticeably higher than training loss
- Gap of 0.05-0.15 is typical for well-regularized models
- Very small gaps often indicate data leakage or insufficient model capacity

**Why It's Suspicious:**
- Combined with data leakage (Issue #1)
- Both losses converged quickly and plateaued
- Suggests model is performing artificially well on validation data

---

## 📊 What Actually Worked

Despite the issues, some architectural choices were sound:

✅ **Model Architecture:**
- Simplified residual blocks (removed attention)
- LayerNorm throughout
- Progressive width reduction (1024→512→256→128)
- Progressive dropout (0.3→0.2→0.1)

✅ **Training Setup:**
- AdamW optimizer with gradient clipping
- Exponential LR decay
- Reasonable batch size (256)

✅ **Output Constraints:**
- All predictions ≥ 1.0 (via softplus)
- Predictions in reasonable range [3.83, 14.8M]

---

## 🔧 Recommendations for Run 7

### PRIORITY 1: Fix Data Leakage (CRITICAL)

**Change the order:**

```python
# WRONG (Run 6):
1. Load data
2. Augment FULL dataset
3. Split train/val

# CORRECT (Run 7):
1. Load data
2. Split train/val (on ORIGINAL data only)
3. Augment ONLY training data
4. Validation remains unchanged
```

**Implementation:**

```python
# Step 1: Load raw data
with open('data/combined_final_n_k_m_P.pkl', 'rb') as f:
    inputs_raw = pickle.load(f)
with open('data/combined_final_mHeights.pkl', 'rb') as f:
    outputs_raw = pickle.load(f)

# Step 2: Split FIRST (on original data)
stratify_labels = [k*10 + m for n, k, m, P in inputs_raw]

(inputs_train_orig, inputs_val,
 outputs_train_orig, outputs_val) = train_test_split(
    inputs_raw,
    outputs_raw,
    test_size=0.15,
    random_state=42,
    stratify=stratify_labels
)

# Step 3: Augment ONLY training data
inputs_train_aug = []
outputs_train_aug = []

for sample, target in zip(inputs_train_orig, outputs_train_orig):
    # Keep original
    inputs_train_aug.append(sample)
    outputs_train_aug.append(target)

    # Add augmented copies
    n, k, m, P = sample
    num_copies = augmentation_config.get((k, m), 0)

    for _ in range(num_copies):
        aug_sample, aug_target = augment_sample(n, k, m, P, target)
        inputs_train_aug.append(aug_sample)
        outputs_train_aug.append(aug_target)

# Step 4: Validation set is UNCHANGED
# inputs_val and outputs_val remain as original samples only
```

---

### PRIORITY 2: Reduce Augmentation

**Run 6 (Excessive):**
```python
augmentation_config = {
    (5, 4): 50,  # Hard
    (6, 3): 50,
    (4, 5): 50,
    (4, 4): 25,  # Medium
    (5, 3): 25,
    (6, 2): 25,
}
# Result: 26x augmentation
```

**Run 7 (Recommended):**
```python
augmentation_config = {
    # Hard cases: Moderate augmentation
    (5, 4): 3,   # +3 copies (not 50!)
    (6, 3): 3,
    (4, 5): 3,

    # Medium cases: Light augmentation
    (4, 4): 2,   # +2 copies (not 25!)
    (5, 3): 2,
    (6, 2): 2,

    # Easy cases: Minimal augmentation to prevent degradation
    (4, 2): 1,   # NEW: +1 copy (was 0)
    (4, 3): 1,   # NEW: +1 copy (was 0)
    (5, 2): 1,   # NEW: +1 copy (was 0)
}
# Expected result: 3-5x augmentation
```

**Rationale:**
- Prevents memorization of pseudo-duplicates
- Gives easy cases some augmentation to prevent degradation
- More balanced representation across difficulty tiers

---

### PRIORITY 3: Increase Noise Scale

**Run 6:**
```python
noise_scale = 0.02  # 2% noise
```

**Run 7:**
```python
noise_scale = 0.05  # 5% noise

def augment_sample(n, k, m, P_matrix, m_height, noise_scale=0.05):
    # Add noise to n (±5%)
    n_aug = n * (1 + np.random.uniform(-noise_scale, noise_scale))
    n_aug = max(1.0, n_aug)

    # Add noise to P matrix (±5%)
    P_aug = P_matrix.copy()
    if P_aug.size > 0:
        noise = np.random.normal(0, noise_scale, P_aug.shape)
        P_aug = P_aug + noise
        P_aug = np.clip(P_aug, 0, 1)

        # Renormalize
        if len(P_aug.shape) == 2:
            row_sums = P_aug.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-7)
            P_aug = P_aug / row_sums

    return (n_aug, k, m, P_aug), m_height
```

**Rationale:**
- Creates more diverse augmented samples
- Reduces memorization of near-duplicates
- Better generalization to new data

---

### PRIORITY 4: Add Easy Case Protection

**Option A: Early Stopping on Easy Cases**

```python
from tensorflow.keras.callbacks import Callback

class EasyCaseMonitor(Callback):
    def __init__(self, val_data, easy_groups, threshold=0.15):
        super().__init__()
        self.val_data = val_data
        self.easy_groups = easy_groups
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        # Compute loss on easy cases only
        easy_mask = self.get_easy_mask()
        easy_predictions = self.model.predict(self.val_data[easy_mask])
        easy_targets = self.val_targets[easy_mask]

        easy_loss = compute_log2_mse(easy_targets, easy_predictions)

        print(f"  Easy case loss: {easy_loss:.4f}")

        if easy_loss > self.threshold:
            print(f"\nWARNING: Easy cases degrading ({easy_loss:.4f} > {self.threshold})")
            print("Consider stopping or reducing hard case weighting")

# Add to callbacks
callbacks = [
    EarlyStopping(...),
    ReduceLROnPlateau(...),
    ModelCheckpoint(...),
    lr_scheduler,
    EasyCaseMonitor(val_data, EASY_GROUPS, threshold=0.15)  # NEW
]
```

**Option B: Balanced Loss Weighting**

```python
# Increase loss weight for easy cases to prevent degradation
group_loss_weights = {
    # Easy groups: Higher loss weight (prevent degradation)
    (4, 2): 3.0,  # 3x weight in loss
    (4, 3): 3.0,
    (5, 2): 3.0,

    # Medium groups: Moderate weight
    (4, 4): 1.5,
    (5, 3): 1.5,
    (6, 2): 1.5,

    # Hard groups: Normal weight
    (5, 4): 1.0,
    (6, 3): 1.0,
    (4, 5): 1.0,
}

# Apply BOTH sample weights (for balancing) AND loss weights (for protection)
```

---

## 📋 Run 7 Implementation Checklist

### Must Fix (Critical):
- [ ] Move augmentation AFTER train-val split
- [ ] Reduce augmentation: 3/2/1 copies for hard/medium/easy
- [ ] Increase noise scale to 5%
- [ ] Add easy case monitoring/protection

### Should Fix (Important):
- [ ] Track per-group metrics separately during training
- [ ] Report balanced metric (equal weight to easy/medium/hard tiers)
- [ ] Add validation on original data only (no augmentation)

### Nice to Have:
- [ ] Implement balanced loss weighting
- [ ] Create separate test set for final evaluation
- [ ] Add data augmentation ablation study

---

## 🎯 Expected Run 7 Results

### Realistic Targets:

| Metric | Run 2 (Baseline) | Run 6 (Invalid) | Run 7 (Target) |
|--------|------------------|-----------------|----------------|
| Overall | ~0.48 | 0.181* | 0.28-0.35 |
| Easy cases | 0.096-0.108 | 0.405-0.460* | **< 0.15** |
| Medium cases | 0.210-0.356 | 0.127-0.155* | 0.20-0.30 |
| Hard cases | 0.958-1.422 | 0.168-0.224* | 0.70-1.20 |

*Invalid due to data leakage

### Success Criteria:
1. ✅ Overall metric < 0.374 (beat target)
2. ✅ Easy cases < 0.15 (no degradation from Run 2)
3. ✅ Medium cases < 0.30
4. ✅ Hard cases < 1.30
5. ✅ No data leakage (proper augmentation order)
6. ✅ Reasonable train-val gap (0.02-0.08)

---

## 📝 Additional Notes

### Why Run 6 Appeared to Succeed

The overall metric of 0.181 looked great, but:
1. **Data leakage** artificially inflated validation performance
2. **Sample imbalance** (65% hard cases) masked easy case failures
3. **Excessive augmentation** created pseudo-duplicates the model memorized

### Real-World Impact

If deployed:
- Would perform poorly on truly new easy cases (4-5x worse than baseline)
- Hard case performance gains wouldn't generalize beyond training distribution
- 2% noise augmentation too small to capture real-world variation

### Lessons Learned

1. **Always split before augmentation** - This is ML 101
2. **Monitor per-group metrics** - Overall metrics can be misleading
3. **Beware of excessive augmentation** - More isn't always better
4. **Document code accurately** - Comments said "+2 copies", code did "+50"
5. **Validate assumptions** - Small train-val gap should raise red flags

---

## References

**Data Files:**
- `data/combined_final_n_k_m_P.pkl` (108,000 samples)
- `data/combined_final_mHeights.pkl` (108,000 targets)

**Notebooks:**
- `train_model_v5.ipynb` (local version)
- `train_model_v5_COMPLETE.ipynb` (Run 6 - master branch)

**Comparison Baselines:**
- Run 2: Val log2-MSE ~0.48 (balanced baseline)
- Run 4: Val log2-MSE 0.538 (aggressive weighting)
- Run 6: Val log2-MSE 0.181 (invalid - data leakage)

---

**Analysis Completed:** 2025-11-30
**Analyst:** Claude Code AI Assistant
**Status:** CRITICAL ISSUES FOUND - RUN 7 REQUIRED
