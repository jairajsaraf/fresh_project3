# Run 5: Simplified Residual Architecture - Fix Run 4 Overfitting

## 🚨 Critical Problem Identified in Run 4

**Run 4 Results: WORSE than baseline!**
- Average validation log2-MSE: **0.538** (target: 0.374)
- Easy cases got **60-70% WORSE** compared to Run 2
- Hard cases barely improved (0.5-3%)

**Root Cause:** Aggressive group weighting (5.0x) caused severe overfitting to hard cases at the expense of easy cases.

## 📊 Key Insights from Professor's Top Submissions

### What Works (Top 30% of Submissions):

1. **Advanced Techniques (Avg Score: 77.8) - BEST**
   - Residual connections with skip paths
   - Multi-head attention (selective use)
   - LayerNorm + BatchNorm combinations

2. **Dense Networks (Avg Score: 73.2) - SIMPLE & EFFECTIVE**
   - Standard MLPs with progressive width reduction
   - BatchNorm across layers
   - Fast training, interpretable

3. **Log-Space Prediction - CRUCIAL**
   - Predicts log2(m-height) instead of raw values
   - Uses MSE(log2(y_true), log2(y_pred))
   - Output: Dense(1, custom_relu) where custom_relu(x) = ReLU(x) + 1.0

4. **Residual MLP + LayerNorm - MODERN BEST PRACTICE**
   - Skip connections prevent vanishing gradients
   - LayerNorm ensures stable activations
   - Dense(1, Softplus) to keep output ≥ 1

5. **Progressive Dropout (45% of top submissions)**
   - Rates decrease across layers: 0.3 → 0.2 → 0.1

6. **AdamW (30% of top submissions)**
   - Better generalization via weight decay

## 🎯 Run 5 Strategy

### Architecture Changes - REMOVED:
- ❌ Multi-head attention (too complex, not worth it)
- ❌ Aggressive group weighting (1.0-5.0x range)
- ❌ Complex MoE architecture

### Architecture Changes - ADDED:
- ✅ Residual connections (proven in top submissions)
- ✅ LayerNorm (better than pure BatchNorm)
- ✅ Simpler architecture (like Example 3 and 5 from PDF)
- ✅ Log-space prediction throughout
- ✅ Progressive width reduction (1024→512→256→128)
- ✅ Progressive dropout pattern (0.3→0.2→0.1)

### Training Changes - REMOVED:
- ❌ Complex sample weighting scheme (5.0x)
- ❌ Overly aggressive regularization on hard cases
- ❌ Curriculum learning (complex)

### Training Changes - ADDED:
- ✅ **MILD class weighting (1.0-2.0x max, not 5.0x)**
- ✅ Better learning rate schedule (exponential decay)
- ✅ More epochs with better early stopping (250 epochs, patience=40)
- ✅ Gradient clipping (clipnorm=1.0)

## 🏗️ Architecture Details

### Simplified Residual Model

```python
# Inputs: n, k, m, P_matrix

# Embeddings
k_embed = Embedding(7, 32)(k)  # 32-dim embedding
m_embed = Embedding(6, 32)(m)  # 32-dim embedding

# P matrix processing
P → Dense(256, gelu) → LayerNorm → Dropout(0.3)
  → Dense(512, gelu) → LayerNorm → Dropout(0.2)

# Combine features
combined = Concat([n, k_embed, m_embed, P_processed])

# Main network
x = Dense(1024, gelu) → LayerNorm → Dropout(0.3)

# Residual Block 1
res1 = x
x → Dense(1024, gelu) → LayerNorm → Dropout(0.2)
  → Dense(1024, gelu) → LayerNorm
  → Add(x, res1)  # Skip connection

# Residual Block 2
res2 = x
x → Dense(1024, gelu) → LayerNorm → Dropout(0.2)
  → Dense(1024, gelu) → LayerNorm
  → Add(x, res2)  # Skip connection

# Progressive width reduction
x → Dense(512, gelu) → LayerNorm → Dropout(0.2)
  → Dense(256, gelu) → LayerNorm → Dropout(0.1)
  → Dense(128, gelu) → LayerNorm → Dropout(0.1)

# Output (log2 prediction with Softplus)
x → Dense(1, linear)
  → Softplus  # Ensures log2 ≥ 0
  → 2^x       # Convert to m-height
```

**Total Parameters:** ~3-4M (compared to ~8M in Run 4)

## 🔧 Training Configuration

### 1. MILD Group Weights (CRITICAL FIX!)

```python
group_weights = {
    (4, 2): 1.0,   # Easy
    (4, 3): 1.0,   # Easy
    (4, 4): 1.0,   # Medium
    (4, 5): 2.0,   # Hard (was 5.0+ in Run 4!)
    (5, 2): 1.0,   # Easy
    (5, 3): 1.0,   # Medium
    (5, 4): 1.5,   # Hard (was 3.0+ in Run 4)
    (6, 2): 1.2,   # Medium
    (6, 3): 1.5,   # Hard (was 3.0+ in Run 4)
}
```

**Max weight: 2.0x** (was 5.0x in Run 4)

### 2. Optimizer - AdamW with Gradient Clipping

```python
optimizer = AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4,
    clipnorm=1.0  # Gradient clipping for stability
)
```

### 3. Learning Rate Schedule

```python
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr  # Warmup
    else:
        return lr * 0.95  # Exponential decay
```

### 4. Callbacks

- **EarlyStopping:** patience=40 (increased from 25)
- **ReduceLROnPlateau:** patience=15, factor=0.7
- **ModelCheckpoint:** Save best model as `best_model_run5.h5`
- **LearningRateScheduler:** Exponential decay

### 5. Training Parameters

- **Epochs:** 250 (increased from 200)
- **Batch size:** 256
- **Sample weights:** MILD weighting (1.0-2.0x)
- **Validation split:** 15% stratified by (k,m)

## 📊 Expected Performance

### Conservative Estimates:

- **Easy cases** (k=4,m=2, k=5,m=2): **< 0.15 log2-MSE** (match or beat Run 2)
- **Medium cases** (k=4,m=4, k=5,m=3, k=6,m=2): **0.20-0.30 log2-MSE**
- **Hard cases** (k=4,m=5, k=5,m=4, k=6,m=3): **0.80-1.20 log2-MSE** (match Run 4)
- **Overall average:** **0.30-0.35** (beat 0.374 target!)

### Key Success Factors:

✅ Don't sacrifice easy cases for hard cases (balanced approach)
✅ Simpler architecture = less overfitting
✅ Better regularization strategy
✅ Log-space prediction for stability

## 🚀 How to Run

```bash
python train_model_v5.py
```

The script will:
1. Load augmented dataset
2. Create stratified train/val split
3. Build simplified residual model
4. Train with MILD weighting
5. Generate comprehensive comparison report

## 📁 Outputs

All results saved to `run_5/` folder:

1. **best_model_run5.h5** - Trained model weights
2. **training_history.png** - Loss curves
3. **predictions_scatter.png** - Prediction quality plots
4. **detailed_comparison.txt** - Complete comparison vs Run 2 and Run 4

## 🔍 What to Monitor

The script prints detailed per-group comparison:

```
k,m      Run 2      Run 4      Run 5      vs R2       vs R4       n
----------------------------------------------------------------------
k=4,m=2  0.103043   0.167000   ???        ???         ???         n=XXX
k=5,m=2  0.096024   0.152000   ???        ???         ???         n=XXX
...
```

**Key metrics to check:**
- Easy cases should NOT degrade from Run 2 (< 0.15)
- Hard cases should match Run 4 performance (0.80-1.30)
- Overall should beat 0.374 target

## 💡 Run 5 Improvements Summary

| Aspect | Run 4 | Run 5 | Impact |
|--------|-------|-------|--------|
| Architecture | MoE + Attention | Simple Residual | -50% parameters |
| Normalization | BatchNorm | LayerNorm | Better gradients |
| Residual Blocks | 3 | 2 | Less overfitting |
| Group Weighting | 1.0-5.0x | 1.0-2.0x | **CRITICAL FIX** |
| Dropout | 0.4-0.45 | 0.3-0.1 (progressive) | Better regularization |
| LR Schedule | Cosine restarts | Exponential decay | Smoother convergence |
| Gradient Clipping | None | 1.0 | More stable |
| Early Stopping | patience=25 | patience=40 | Better convergence |

## 📈 Expected Improvements Over Run 4

1. **Easy cases:** 60-70% improvement (fix degradation)
2. **Medium cases:** 10-20% improvement
3. **Hard cases:** Maintain Run 4 performance
4. **Overall:** 30-40% improvement (0.538 → 0.30-0.35)

---

**Ready to train!** Run the script and compare results against baselines. 🚀
