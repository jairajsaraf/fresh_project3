# Run 7: Fixed Data Leakage & Improved Generalization

## Overview

Run 7 addresses all critical issues found in Run 6 and implements a robust training pipeline with proper data handling and balanced optimization.

## Critical Fixes from Run 6

### 1. ✅ Data Leakage Fixed

**Run 6 Problem:**
```python
# WRONG: Augment THEN split
augment(full_dataset)  # Creates 2.8M samples
train_test_split()     # Augmented copies leak into validation
```

**Run 7 Solution:**
```python
# CORRECT: Split THEN augment
train_test_split()            # Split 108K original samples
augment(training_set_only)    # Only training set gets augmented
# Validation set: ONLY original samples (no augmentation)
```

### 2. ✅ Reduced Augmentation

| Difficulty | Run 6 | Run 7 | Improvement |
|------------|-------|-------|-------------|
| Hard cases | +50 copies | +3 copies | 94% reduction |
| Medium cases | +25 copies | +2 copies | 92% reduction |
| Easy cases | 0 copies | +1 copy | NEW: prevents degradation |

**Result:** 26x → ~3-4x augmentation ratio

### 3. ✅ Increased Noise Scale

- **Run 6:** 2% noise (too small, creates pseudo-duplicates)
- **Run 7:** 5% noise (better diversity, improved generalization)

### 4. ✅ Easy Case Protection

**New: TierMetricsCallback**
- Monitors Easy/Medium/Hard performance separately each epoch
- Stops training if easy cases degrade beyond 0.15 threshold
- Prevents Run 6 issue where easy cases got 3-4x worse

### 5. ✅ Balanced Loss Function

**Group loss weights:**
```python
Easy groups:   3.0x weight  # Protect from degradation
Medium groups: 1.5x weight  # Moderate priority
Hard groups:   1.0x weight  # Normal priority
```

### 6. ✅ Tier Metrics Tracking

Reports performance separately by difficulty tier:
- Easy average (k=4,m=2 | k=4,m=3 | k=5,m=2)
- Medium average (k=4,m=4 | k=5,m=3 | k=6,m=2)
- Hard average (k=5,m=4 | k=6,m=3 | k=4,m=5)
- Balanced metric (equal weight to each tier)

### 7. ✅ L2 Regularization

- L2 regularization (1e-4) on all Dense layers
- Reduces overfitting
- Improves generalization

## Files

- `train_model_v7.py` - Python script version
- `train_model_v7.ipynb` - Jupyter notebook version
- `best_model_run7.h5` - Trained model weights (after training)
- `run_7/` - Output directory
  - `training_history_run7.png` - Loss curves
  - `predictions_scatter_run7.png` - Prediction quality plots
  - `detailed_comparison.txt` - Per-group metrics

## Expected Performance

Based on Run 6 analysis and fixes:

| Metric | Target | Reasoning |
|--------|--------|-----------|
| Easy cases | < 0.15 | Match Run 2 baseline (~0.10), no degradation |
| Medium cases | 0.20-0.30 | Improve from Run 2 (~0.26) |
| Hard cases | 0.70-1.20 | Improve from Run 2 (~1.14) |
| Overall | 0.28-0.35 | Beat 0.374 target with VALID metrics |

## How to Run

### Option 1: Python Script

```bash
cd /home/user/fresh_project3
python train_model_v7.py
```

### Option 2: Jupyter Notebook

```bash
cd /home/user/fresh_project3
jupyter notebook train_model_v7.ipynb
```

## Training Details

**Data:**
- Original samples: 108,000
- Training (after aug): ~270,000-350,000
- Validation (original): 16,200

**Model:**
- Architecture: 2 residual blocks + progressive reduction
- Parameters: ~5.6M
- Regularization: L2 (1e-4) + Dropout (0.3→0.2→0.1)

**Training:**
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Batch size: 256
- Max epochs: 250
- Early stopping: patience=40
- LR schedule: Exponential decay (0.95 per epoch)

**Callbacks:**
- EarlyStopping (val_loss, patience=40)
- ReduceLROnPlateau (factor=0.7, patience=15)
- ModelCheckpoint (best_model_run7.h5)
- LR Scheduler (exponential decay)
- **TierMetricsCallback (easy case monitoring)**

## Key Differences from Previous Runs

### vs Run 2 (Baseline)
- **Added:** Data augmentation (3/2/1 for hard/medium/easy)
- **Added:** Balanced loss weighting
- **Added:** L2 regularization
- **Expected:** Better hard/medium cases, same or better easy cases

### vs Run 6 (Invalid)
- **Fixed:** Data leakage (split before augment)
- **Fixed:** Excessive augmentation (3-4x vs 26x)
- **Fixed:** Noise scale (5% vs 2%)
- **Fixed:** Easy case degradation (monitoring + weighting)
- **Expected:** Valid metrics, no easy case degradation

## Validation

After training, verify:

1. **No Data Leakage:**
   - Validation set size should be ~16,200 (original samples only)
   - Check that augmentation happened after split

2. **Easy Cases Not Degraded:**
   - Easy case average < 0.15
   - Compare to Run 2 baseline (~0.10)

3. **Reasonable Train-Val Gap:**
   - Gap should be 0.02-0.08
   - Too small (<0.01) might indicate issues
   - Too large (>0.15) indicates overfitting

4. **Overall Performance:**
   - Validation log2-MSE < 0.374 (target)
   - All tiers meet targets
   - Balanced metric reasonable

## Troubleshooting

**If easy cases degrade:**
- Increase easy case weight (currently 3.0x, try 4.0x)
- Lower easy case threshold (currently 0.15, try 0.12)
- Reduce hard case augmentation (currently +3, try +2)

**If overall metric doesn't beat target:**
- Check per-tier performance separately
- May need more augmentation for hard cases
- May need to adjust loss weights

**If train-val gap is too large:**
- Increase dropout rates
- Increase L2 regularization
- Reduce model capacity

**If training is unstable:**
- Reduce learning rate
- Increase gradient clipping
- Check for NaN values in predictions

## Success Criteria

Run 7 is successful if:

1. ✅ Validation log2-MSE < 0.374
2. ✅ Easy cases < 0.15 (no degradation)
3. ✅ Medium cases < 0.30
4. ✅ Hard cases < 1.30
5. ✅ Train-val gap 0.02-0.08 (healthy range)
6. ✅ No data leakage (verified by sample counts)

## Next Steps After Run 7

If Run 7 succeeds:
1. Save the model for production use
2. Generate predictions on test set
3. Document final architecture and hyperparameters
4. Create deployment pipeline

If Run 7 needs tuning:
1. Analyze per-group metrics
2. Adjust loss weights if specific groups underperform
3. Fine-tune augmentation ratios
4. Consider ensemble methods

## Contact

For questions or issues with Run 7 implementation, refer to:
- `RUN_6_ANALYSIS.md` - Detailed analysis of issues
- `train_model_v7.py` - Full implementation with comments
- GitHub issues for this repository
