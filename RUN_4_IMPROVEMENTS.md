# Run 4 Improvements - Based on Run 2 Analysis

## Executive Summary

After analyzing Run 2 results (training history, per-group performance, and prediction scatter plots), I've identified key issues and created `train_model_v4.py` with targeted improvements to address:

1. **Overfitting** (train-val loss gap in last 80% of training)
2. **Poor performance on hard groups** (k=4,m=5: 1.422, k=6,m=3: 1.083, k=5,m=4: 0.958)
3. **Validation loss plateau** (~100-120 epochs)

---

## Run 2 Analysis Summary

### 1. Training History Analysis (training_history.png)
**Observations:**
- Train loss decreases from ~9 to ~0.4
- Val loss decreases from ~7 to ~0.5 but plateaus around epoch 100-120
- **Critical issue**: Persistent gap between train (~0.4) and val (~0.5) in last 80%
- **Diagnosis**: Model is overfitting despite existing regularization

### 2. Per-Group Performance Analysis (per_group_performance.txt)
**Observations:**
```
Best performers (Log2-MSE):
  k=5,m=2: 0.096 ✅
  k=4,m=2: 0.103 ✅
  k=4,m=3: 0.108 ✅

Good performers:
  k=4,m=4: 0.210 ✓
  k=5,m=3: 0.227 ✓
  k=6,m=2: 0.356 ✓

Problematic performers:
  k=5,m=4: 0.958 ❌
  k=6,m=3: 1.083 ❌
  k=4,m=5: 1.422 ❌ (WORST)
```
**Diagnosis**: Model struggles with higher complexity groups (larger m values)

### 3. Prediction Scatter Analysis (predictions_scatter.png)
**Observations:**
- Log scale plot shows excellent alignment across 10^0 to 10^6
- Linear scale shows more scatter at higher values
- No systematic bias detected
- **Conclusion**: Model architecture is sound, needs better regularization

---

## Run 4 Improvements

### 1. **Enhanced Regularization** (Combat Overfitting)

#### Dropout Rate Increases:
```python
Component              | Run 3    | Run 4    | Change
-----------------------|----------|----------|--------
P matrix dropout       | 0.20     | 0.25     | +25%
Expert dropout         | 0.30     | 0.35-0.40| +17-33%
Attention dropout      | 0.10     | 0.15     | +50%
Main network dropout   | 0.35     | 0.45     | +29%
Residual blocks        | 0.30     | 0.40     | +33%
Final layer dropout    | 0.20     | 0.30     | +50%
Added final Dense dropout | 0.00  | 0.20     | NEW
```

#### Weight Decay Increase:
```python
Run 3: weight_decay = 5e-4
Run 4: weight_decay = 1e-3  # Doubled (2x increase)
```

**Expected Impact**: Reduce train-val gap from ~0.1 to ~0.05

---

### 2. **More Aggressive Early Stopping**

#### Changes:
```python
Component              | Run 3    | Run 4    | Rationale
-----------------------|----------|----------|---------------------------
EarlyStopping patience | 50       | 25       | Val loss plateaus @100-120
Max epochs             | 250      | 200      | Reduced training time
ReduceLROnPlateau      | None     | Added    | Patience=10, factor=0.5
```

**Expected Impact**: Stop training earlier when val loss plateaus, preventing overfitting

---

### 3. **Enhanced Curriculum Learning**

#### Adjusted Timeline (More Aggressive):
```python
Phase               | Run 3         | Run 4         | Change
--------------------|---------------|---------------|------------------
Phase 1 (Easy)      | Epochs 0-40   | Epochs 0-30   | Start earlier
Phase 2 (Medium)    | Epochs 41-80  | Epochs 31-60  | Start earlier
Phase 3 (Hard)      | Epochs 81+    | Epochs 61+    | Start earlier
```

#### Group-Specific Weighting (Phase 3):
Based on Run 2 performance, targeting worst performers:

```python
Group    | Run 2 Loss | Run 4 Weight | Multiplier
---------|------------|--------------|------------
k=4,m=5  | 1.422      | up to 6.5x   | 1.3x hard_weight
k=6,m=3  | 1.083      | up to 5.5x   | 1.1x hard_weight
k=5,m=4  | 0.958      | up to 5.0x   | 1.0x hard_weight
Medium   | 0.2-0.4    | 1.5x         | baseline
Easy     | 0.1        | 1.0x         | baseline
```

**Weight Growth Formula**:
```python
hard_weight = min(5.0, 2.0 + (epoch - 60) * 0.08)
# Grows from 2.0 at epoch 60 to 5.0 at epoch 97+
# More aggressive than Run 3: min(3.0, 1.0 + (epoch - 80) * 0.05)
```

**Expected Impact**: Significantly improve k=4,m=5, k=6,m=3, k=5,m=4 performance

---

### 4. **Adjusted Learning Rate Schedule**

#### Cosine Annealing Changes:
```python
Component          | Run 3    | Run 4    | Rationale
-------------------|----------|----------|---------------------------
Restart period     | 40       | 30       | Shorter cycles
Initial LR         | 7e-4     | 7e-4     | Unchanged
Min LR             | 1e-6     | 1e-6     | Unchanged
```

**Expected Impact**: More frequent LR restarts for better exploration

---

## Expected Outcomes

### Primary Goals:
1. **Reduce overfitting**: Train-val gap from ~0.1 → ~0.05
2. **Improve hard groups**:
   - k=4,m=5: 1.422 → <1.0 (target: -30% improvement)
   - k=6,m=3: 1.083 → <0.8 (target: -26% improvement)
   - k=5,m=4: 0.958 → <0.7 (target: -27% improvement)
3. **Maintain easy/medium performance**: Keep <0.3 for easy, <0.4 for medium

### Secondary Goals:
1. Faster convergence (stop around epoch 100-120 instead of 200+)
2. Better generalization (smaller train-val gap)
3. Overall validation log2-MSE: <0.45 (vs Run 2: 0.507)

---

## How to Run

```bash
python train_model_v4.py
```

**Expected Training Time**:
- ~100-120 epochs (vs 200+ in Run 3)
- With early stopping at patience=25

**Outputs** (saved to `run_4/` folder):
1. `best_model.h5` - Best model weights
2. `run_4/training_history.png` - Training curves
3. `run_4/predictions_scatter.png` - Prediction quality
4. `run_4/per_group_performance.txt` - Per-(k,m) metrics with Run 2 comparison

---

## Key Differences from Run 3

| Aspect | Run 3 | Run 4 |
|--------|-------|-------|
| **Focus** | Mixture-of-Experts + Curriculum | Enhanced regularization + Aggressive hard-group weighting |
| **Regularization** | Moderate (dropout 0.2-0.35) | Strong (dropout 0.25-0.45) |
| **Weight Decay** | 5e-4 | 1e-3 (2x) |
| **Curriculum** | 3 phases (0-40, 41-80, 81+) | 3 phases (0-30, 31-60, 61+) - earlier transitions |
| **Hard Group Weight** | Up to 3x | Up to 6.5x for k=4,m=5 |
| **Early Stopping** | Patience 50 | Patience 25 + ReduceLROnPlateau |
| **Max Epochs** | 250 | 200 |
| **LR Restart Period** | 40 | 30 |

---

## Monitoring During Training

Watch for these indicators of success:

### 1. Reduced Overfitting:
- Train loss should stay closer to val loss (gap <0.05)
- Val loss should not plateau as early

### 2. Improved Hard Groups:
- After epoch 60, watch for improvements in k=4,m=5, k=6,m=3, k=5,m=4

### 3. Better Generalization:
- Final val log2-MSE should be <0.45
- Per-group performance should be more balanced

---

## Success Criteria

**Minimum Acceptable Performance**:
- Overall val log2-MSE: <0.45 (vs Run 2: 0.507)
- k=4,m=5: <1.0 (vs Run 2: 1.422)
- k=6,m=3: <0.8 (vs Run 2: 1.083)
- k=5,m=4: <0.7 (vs Run 2: 0.958)
- Easy groups: Maintain <0.15
- Medium groups: Maintain <0.4

**Target Performance** (Stretch Goal):
- Overall val log2-MSE: <0.374 (beat the benchmark!)
- All groups: <0.5
- Hard groups: <0.8

---

## Next Steps After Run 4

If Run 4 achieves the goals:
1. Analyze which improvements had the most impact
2. Fine-tune hyperparameters further
3. Consider ensemble methods

If Run 4 doesn't achieve the goals:
1. Analyze which groups are still problematic
2. Consider architecture changes (e.g., group-specific heads)
3. Explore alternative loss functions (e.g., Huber loss for hard groups)
4. Investigate data augmentation for hard groups
