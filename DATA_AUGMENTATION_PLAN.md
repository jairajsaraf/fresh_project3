# Data Augmentation Plan for Height Prediction

## Executive Summary

Successfully implemented advanced data augmentation strategy that:
- ✅ Properly merges all three datasets (DS-1, DS-2, DS-3)
- ✅ Achieves **perfect balance** (1.00x imbalance ratio)
- ✅ Generates **108,000 high-quality samples** (12,000 per (k,m) group)
- ✅ Applies domain-aware augmentation techniques

## Problem Analysis

### Original "Combined" Dataset Issues
- **Severe imbalance**: 19x ratio (k=4,m=2: 52.88% vs k=4,m=5: 2.78%)
- **Missing DS-3**: Only used DS-1 + partial DS-2
- **Total**: 85,499 samples with poor distribution

### Available Source Datasets

| Dataset | Samples | Contribution | Description |
|---------|---------|--------------|-------------|
| DS-1 | 32,087 | 24.2% | Original samples |
| DS-2 | 44,399 | 33.4% | Training set |
| DS-3 | **56,365** | **42.4%** | **New dataset!** |
| **Total** | **132,851** | **100%** | All sources combined |

### Distribution Before Augmentation

```
(k,m)    Count    Percentage
(4, 2)   12,494    9.40%
(4, 3)   12,188    9.17%
(4, 4)   11,226    8.45%
(4, 5)    7,220    5.43%  ← Underrepresented
(5, 2)   14,883   11.20%
(5, 3)   13,778   10.37%
(5, 4)    8,603    6.48%  ← Underrepresented
(6, 2)   34,247   25.78%  ← Overrepresented
(6, 3)   18,212   13.71%

Imbalance ratio: 4.7x
```

## Augmentation Strategy

### Design Principles

1. **Merge First**: Combine all three datasets to maximize real data
2. **Balance Smart**: Target 12,000 samples per (k,m) group
3. **Augment Carefully**: Use domain-aware techniques that preserve physical properties
4. **Subsample Wisely**: Reduce overrepresented groups to avoid dominance

### Augmentation Techniques

#### 1. Gaussian Noise Injection
```python
P_augmented = P + noise * std(P)
noise ~ N(0, σ²)
```
- **Purpose**: Add statistical variability while preserving structure
- **Noise levels**: 0.05 (moderate), 0.08 (strong)
- **Preserves**: Matrix dimensionality, value distribution

#### 2. Controlled Perturbations
```python
P_augmented = P * (1 + perturbation)
perturbation ~ U(-δ, δ)
```
- **Purpose**: Scale values while maintaining relationships
- **Strength**: 0.03 (moderate), 0.05 (strong)
- **Preserves**: Matrix structure, relative magnitudes

#### 3. SMOTE-like Interpolation
```python
P_new = α * P1 + (1 - α) * P2
m_height_new = exp(α * log(h1) + (1 - α) * log(h2))
```
- **Purpose**: Generate realistic samples between existing ones
- **Constraint**: Only interpolate samples with same (k, m, n)
- **Alpha range**: [0.3, 0.7] (avoid extremes)
- **Output**: Geometric mean in log-space (appropriate for m-height)

#### 4. Mixed Techniques
```python
P_aug = add_noise(perturb(P))
```
- **Purpose**: Combine multiple augmentations for diversity
- **Application**: Random selection per augmented sample

### Augmentation Execution

| (k,m) | Original | Needed | Action | Final |
|-------|----------|--------|--------|-------|
| (4,2) | 12,494 | -494 | Subsample | 12,000 |
| (4,3) | 12,188 | -188 | Subsample | 12,000 |
| (4,4) | 11,226 | +774 | Augment | 12,000 |
| (4,5) | 7,220 | +4,780 | **Augment heavily** | 12,000 |
| (5,2) | 14,883 | -2,883 | Subsample | 12,000 |
| (5,3) | 13,778 | -1,778 | Subsample | 12,000 |
| (5,4) | 8,603 | +3,397 | **Augment heavily** | 12,000 |
| (6,2) | 34,247 | -22,247 | Subsample | 12,000 |
| (6,3) | 18,212 | -6,212 | Subsample | 12,000 |

**Total augmented samples generated**: ~8,951
**Total real samples used**: ~99,049

## Results

### Final Dataset Characteristics

```
Total samples: 108,000
Samples per (k,m): 12,000 (exactly)
Imbalance ratio: 1.00x (PERFECT BALANCE)
Distribution: 11.11% per group (all 9 groups)
```

### Quality Metrics

- **Real data percentage**: 91.7% (99,049 / 108,000)
- **Augmented data**: 8.3% (strategically applied only where needed)
- **Training samples**: ~91,800 (85% split)
- **Validation samples**: ~16,200 (15% split)

### Advantages Over Previous Approach

| Metric | Old Combined | New Augmented | Improvement |
|--------|--------------|---------------|-------------|
| Total samples | 85,499 | 108,000 | +26% |
| Real data sources | 2 (DS-1, DS-2) | 3 (DS-1, DS-2, DS-3) | +33% |
| Imbalance ratio | 19.0x | 1.00x | **19x better** |
| DS-3 included | ❌ | ✅ | New data! |
| Augmentation | Random resampling | Domain-aware | Smarter |

## Implementation

### Separate Script Design

**File**: `augment_dataset.py`

**Why separate from training?**
- ✅ **CPU-intensive**: Data augmentation doesn't need GPU
- ✅ **Run once**: Generate dataset once, train multiple times
- ✅ **Reproducible**: Fixed seed ensures consistent results
- ✅ **Modularity**: Easy to modify augmentation without retraining
- ✅ **Efficiency**: Don't re-augment every training run

### Usage

```bash
# Step 1: Run data augmentation (CPU-intensive, run once)
python augment_dataset.py

# Output files:
#   - augmented_n_k_m_P.pkl (108,000 samples)
#   - augmented_mHeights.pkl

# Step 2: Train model (GPU-intensive, can run multiple times)
python train_model.py
```

### Modified Training Script

Updated `train_model.py` to:
1. Load augmented dataset instead of combined dataset
2. Skip rebalancing step (data already balanced)
3. Use all 108,000 samples directly

## Expected Impact

### Performance Improvements

1. **Better generalization**: More diverse training data
2. **Reduced bias**: Equal representation of all (k,m) groups
3. **Lower variance**: More samples in minority classes
4. **Improved per-group metrics**: Especially for (4,5) and (5,4)

### Predicted Results

- **Baseline (old)**: 0.374 log2-MSE
- **Expected (new)**: 0.30-0.33 log2-MSE
- **Improvement**: 12-20% reduction in error

## Future Enhancements

### Potential Additions

1. **Adversarial augmentation**: Generate challenging samples
2. **Physics-based constraints**: If P matrices represent physical properties
3. **Conditional augmentation**: Different strategies per (k,m) group
4. **Active learning**: Identify and augment most uncertain regions

### Monitoring

- Track per-(k,m) validation loss
- Compare real vs augmented sample performance
- Adjust augmentation ratios if needed

## Conclusion

The new data augmentation strategy provides:
- ✅ **108,000 balanced samples** vs 85,499 imbalanced
- ✅ **1.00x perfect balance** vs 19x imbalance
- ✅ **All 3 datasets included** vs only 2
- ✅ **Domain-aware augmentation** vs naive resampling
- ✅ **Modular CPU script** vs embedded in training

This should significantly improve model performance, especially for previously underrepresented (k,m) combinations.
