"""
Height Prediction Model Training Script - Run 5
Objective: Fix Run 4 overfitting issues with simplified residual architecture

Run 5 Strategy (Based on Professor's Top Submissions):
1. REMOVE multi-head attention (too complex, not worth it)
2. REMOVE aggressive group weighting (caused 60-70% degradation in easy cases)
3. ADD residual connections with LayerNorm (proven in top submissions)
4. ADD progressive width reduction (1024→512→256→128)
5. ADD progressive dropout pattern (0.3→0.2→0.1)
6. USE mild class weighting (1.0-2.0x max, not 5.0x)
7. IMPROVE learning rate schedule with exponential decay
8. EXTEND training with better early stopping (patience=40, max_epochs=250)

Expected Performance:
- Easy cases: < 0.15 log2-MSE (match Run 2, don't degrade!)
- Medium cases: 0.20-0.30 log2-MSE
- Hard cases: 0.80-1.30 log2-MSE (match Run 4)
- Overall: 0.30-0.35 (beat 0.374 target!)
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Flatten, Concatenate,
    Dropout, BatchNormalization, Add, LayerNormalization, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    Callback, LearningRateScheduler
)
from tensorflow.keras.optimizers import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("="*70)
print("HEIGHT PREDICTION MODEL - RUN 5 (SIMPLIFIED RESIDUAL)")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: Loading Data")
print("-"*70)

with open('augmented_n_k_m_P.pkl', 'rb') as f:
    inputs_raw = pickle.load(f)

with open('augmented_mHeights.pkl', 'rb') as f:
    outputs_raw = pickle.load(f)

print(f"Raw input samples: {len(inputs_raw)}")
print(f"Raw output samples: {len(outputs_raw)}")
if len(inputs_raw) > 0:
    sample = inputs_raw[0]
    print(f"Sample structure: [n={sample[0]}, k={sample[1]}, m={sample[2]}, P_matrix shape={sample[3].shape}]")
print(f"Output range: [{np.min(outputs_raw):.2f}, {np.max(outputs_raw):.2f}]")
print()

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================
print("STEP 2: Preparing Data for Training")
print("-"*70)

inputs_rebalanced = inputs_raw
outputs_rebalanced = outputs_raw

n_values = []
k_values = []
m_values = []
P_matrices_flattened = []

for sample in inputs_rebalanced:
    n_values.append(sample[0])
    k_values.append(sample[1])
    m_values.append(sample[2])
    P_matrices_flattened.append(sample[3].flatten())

n_values = np.array(n_values, dtype=np.float32).reshape(-1, 1)
k_values = np.array(k_values, dtype=np.int32).reshape(-1, 1)
m_values = np.array(m_values, dtype=np.int32).reshape(-1, 1)
outputs_array = np.array(outputs_rebalanced, dtype=np.float32)

# Pad P matrices
max_p_size = max(len(p) for p in P_matrices_flattened)
P_matrices_padded = []

for p in P_matrices_flattened:
    if len(p) < max_p_size:
        padded = np.zeros(max_p_size, dtype=np.float32)
        padded[:len(p)] = p
        P_matrices_padded.append(padded)
    else:
        P_matrices_padded.append(p)

P_matrices = np.array(P_matrices_padded, dtype=np.float32)

print(f"n_values shape: {n_values.shape}")
print(f"k_values shape: {k_values.shape}, range: [{k_values.min()}, {k_values.max()}]")
print(f"m_values shape: {m_values.shape}, range: [{m_values.min()}, {m_values.max()}]")
print(f"P_matrices shape: {P_matrices.shape}")

# Normalize P matrices
scaler = StandardScaler()
P_matrices_normalized = scaler.fit_transform(P_matrices)

print(f"P matrices normalized: mean={P_matrices_normalized.mean():.4f}, std={P_matrices_normalized.std():.4f}")

outputs_array = np.maximum(outputs_array, 1.0)
print(f"Output (m-height) range: [{outputs_array.min():.2f}, {outputs_array.max():.2f}]")
print()

# ============================================================================
# STEP 3: STRATIFIED TRAIN-VAL SPLIT
# ============================================================================
print("STEP 3: Creating Stratified Train-Validation Split")
print("-"*70)

stratify_labels = k_values.flatten() * 10 + m_values.flatten()

(n_train, n_val,
 k_train, k_val,
 m_train, m_val,
 P_train, P_val,
 y_train, y_val,
 strat_train, strat_val) = train_test_split(
    n_values, k_values, m_values, P_matrices_normalized, outputs_array,
    stratify_labels,
    test_size=0.15,
    random_state=42,
    stratify=stratify_labels
)

print(f"Training samples: {len(y_train)}")
print(f"Validation samples: {len(y_val)}")
print()

# ============================================================================
# STEP 4: DEFINE COMPLEXITY GROUPS
# ============================================================================
print("STEP 4: Defining Complexity Groups")
print("-"*70)

# Based on Run 2 per-group performance:
# Best: k=5,m=2 (0.096), k=4,m=2 (0.103), k=4,m=3 (0.108)
# Good: k=4,m=4 (0.210), k=5,m=3 (0.227), k=6,m=2 (0.356)
# Poor: k=5,m=4 (0.958), k=6,m=3 (1.083), k=4,m=5 (1.422)

EASY_GROUPS = [(4, 2), (4, 3), (5, 2)]
MEDIUM_GROUPS = [(4, 4), (5, 3), (6, 2)]
HARD_GROUPS = [(5, 4), (6, 3), (4, 5)]

print(f"Easy groups (Run 2: 0.096-0.108): {EASY_GROUPS}")
print(f"Medium groups (Run 2: 0.210-0.356): {MEDIUM_GROUPS}")
print(f"Hard groups (Run 2: 0.958-1.422): {HARD_GROUPS}")
print()

# ============================================================================
# STEP 5: BUILD SIMPLIFIED RESIDUAL MODEL
# ============================================================================
print("STEP 5: Building Simplified Residual Model")
print("-"*70)

def build_simplified_residual_model(p_shape, k_vocab_size=7, m_vocab_size=6):
    """
    Simplified residual architecture based on top submissions

    Key improvements over Run 4:
    - NO multi-head attention (removed complexity)
    - LayerNorm + BatchNorm combination (better than pure BatchNorm)
    - Only 2 residual blocks (reduced from 3-4)
    - Progressive width reduction (1024→512→256→128)
    - Progressive dropout (0.3→0.2→0.1)
    - Softplus output for log2 constraint
    """

    # Inputs
    n_input = Input(shape=(1,), name='n')
    k_input = Input(shape=(1,), name='k', dtype=tf.int32)
    m_input = Input(shape=(1,), name='m', dtype=tf.int32)
    P_input = Input(shape=(p_shape,), name='P_flat')

    # Embeddings (keep these, they work well)
    k_embed = Flatten()(Embedding(k_vocab_size, 32, name='k_embedding')(k_input))
    m_embed = Flatten()(Embedding(m_vocab_size, 32, name='m_embedding')(m_input))

    # Initial P processing (simpler than Run 4)
    x = Dense(256, activation='gelu', name='P_initial_1')(P_input)
    x = LayerNormalization(name='P_ln1')(x)  # LayerNorm instead of BatchNorm
    x = Dropout(0.3)(x)

    x = Dense(512, activation='gelu', name='P_initial_2')(x)
    x = LayerNormalization(name='P_ln2')(x)
    x = Dropout(0.2)(x)

    # Combine with embeddings
    x = Concatenate(name='combine_embeddings')([n_input, k_embed, m_embed, x])

    # Initial dense layer
    x = Dense(1024, activation='gelu', name='main_dense1')(x)
    x = LayerNormalization(name='main_ln1')(x)
    x = Dropout(0.3)(x)

    # Residual Block 1 (with skip connection)
    residual = x
    x = Dense(1024, activation='gelu', name='res1_dense1')(x)
    x = LayerNormalization(name='res1_ln1')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='gelu', name='res1_dense2')(x)
    x = LayerNormalization(name='res1_ln2')(x)
    x = Add(name='res1_add')([x, residual])  # Skip connection

    # Residual Block 2 (with skip connection)
    residual = x
    x = Dense(1024, activation='gelu', name='res2_dense1')(x)
    x = LayerNormalization(name='res2_ln1')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='gelu', name='res2_dense2')(x)
    x = LayerNormalization(name='res2_ln2')(x)
    x = Add(name='res2_add')([x, residual])  # Skip connection

    # Progressive width reduction (like top submissions)
    x = Dense(512, activation='gelu', name='reduction1')(x)
    x = LayerNormalization(name='reduction_ln1')(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation='gelu', name='reduction2')(x)
    x = LayerNormalization(name='reduction_ln2')(x)
    x = Dropout(0.1)(x)

    x = Dense(128, activation='gelu', name='reduction3')(x)
    x = LayerNormalization(name='reduction_ln3')(x)
    x = Dropout(0.1)(x)

    # Output: Predict log2(m-height) then convert to m-height
    # Using Softplus to ensure output ≥ 1 (like top submissions)
    log2_pred = Dense(1, activation='linear', name='log2_output')(x)
    log2_pred_positive = Lambda(lambda z: tf.nn.softplus(z), name='softplus')(log2_pred)
    m_height_pred = Lambda(lambda z: tf.pow(2.0, z), name='m_height')(log2_pred_positive)

    model = Model(
        inputs=[n_input, k_input, m_input, P_input],
        outputs=m_height_pred,
        name='simplified_residual_model_v5'
    )

    return model

p_shape = P_train.shape[1]
model = build_simplified_residual_model(
    p_shape,
    k_vocab_size=k_values.max()+1,
    m_vocab_size=m_values.max()+1
)

print(f"Model built successfully!")
print(f"Total parameters: {model.count_params():,}")
print()
print("Architecture highlights:")
print("  ✅ NO multi-head attention (removed)")
print("  ✅ LayerNorm throughout (better for residual nets)")
print("  ✅ 2 residual blocks (simplified from 3-4)")
print("  ✅ Progressive width: 1024→512→256→128")
print("  ✅ Progressive dropout: 0.3→0.2→0.1")
print("  ✅ Softplus output for log2 stability")
print()

# ============================================================================
# STEP 6: DEFINE LOSS FUNCTION
# ============================================================================
print("STEP 6: Defining Loss Function")
print("-"*70)

def log2_mse_loss(y_true, y_pred):
    """MSE in log2 space"""
    epsilon = 1e-7
    y_true = tf.maximum(y_true, epsilon)
    y_pred = tf.maximum(y_pred, epsilon)

    log2_true = tf.math.log(y_true) / tf.math.log(2.0)
    log2_pred = tf.math.log(y_pred) / tf.math.log(2.0)

    return tf.reduce_mean(tf.square(log2_true - log2_pred))

print("Custom log2-MSE loss function defined")
print()

# ============================================================================
# STEP 7: MILD GROUP WEIGHTING (CRITICAL FIX!)
# ============================================================================
print("STEP 7: Computing MILD Group Weights")
print("-"*70)

# CRITICAL: Run 4 used 5.0x weighting which caused 60-70% degradation in easy cases
# Run 5: Use MILD weighting (1.0-2.0x max) to avoid overfitting to hard cases

# Group weights based on difficulty (MILD, not aggressive!)
group_weights = {
    (4, 2): 1.0,   # Easy
    (4, 3): 1.0,   # Easy
    (4, 4): 1.0,   # Medium
    (4, 5): 2.0,   # Hard (was 5.0+ in Run 4 - TOO AGGRESSIVE!)
    (5, 2): 1.0,   # Easy
    (5, 3): 1.0,   # Medium
    (5, 4): 1.5,   # Hard (was 3.0+ in Run 4)
    (6, 2): 1.2,   # Medium (was 2.0 in Run 4)
    (6, 3): 1.5,   # Hard (was 3.0+ in Run 4)
}

# Compute sample weights for training
sample_weights_train = np.ones(len(y_train), dtype=np.float32)

for i, (k, m) in enumerate(zip(k_train, m_train)):
    key = (k[0], m[0])
    if key in group_weights:
        sample_weights_train[i] = group_weights[key]

print("Group weights (MILD - not aggressive):")
for (k, m), weight in sorted(group_weights.items()):
    print(f"  k={k}, m={m}: {weight:.1f}x")
print()
print("CRITICAL: Max weight is 2.0x (was 5.0x+ in Run 4)")
print("This prevents overfitting to hard cases at expense of easy cases")
print()

# ============================================================================
# STEP 8: COMPILE MODEL WITH IMPROVED OPTIMIZER
# ============================================================================
print("STEP 8: Compiling Model")
print("-"*70)

# AdamW with gradient clipping for stability
optimizer = AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4,  # Moderate weight decay
    clipnorm=1.0  # Add gradient clipping
)

model.compile(
    optimizer=optimizer,
    loss=log2_mse_loss,
    metrics=[log2_mse_loss]
)

print("Model compiled with AdamW optimizer")
print(f"  Learning rate: 1e-3")
print(f"  Weight decay: 1e-4")
print(f"  Gradient clipping: 1.0 (NEW!)")
print()

# ============================================================================
# STEP 9: SETUP IMPROVED CALLBACKS
# ============================================================================
print("STEP 9: Setting Up Training Callbacks")
print("-"*70)

def lr_schedule(epoch, lr):
    """Exponential decay schedule (like top submissions)"""
    if epoch < 10:
        return lr  # Warmup phase
    else:
        return lr * 0.95  # Decay by 5% each epoch

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=1)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=40,  # Increased from 25 (Run 4) for better convergence
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,  # Less aggressive than 0.5
        patience=15,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_run5.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    lr_scheduler  # Exponential decay
]

print("Callbacks configured:")
print("  - EarlyStopping (patience=40, up from 25)")
print("  - ReduceLROnPlateau (patience=15, factor=0.7)")
print("  - ModelCheckpoint (best_model_run5.h5)")
print("  - Exponential LR decay (0.95 per epoch after warmup)")
print()

# ============================================================================
# STEP 10: TRAIN MODEL
# ============================================================================
print("="*70)
print("STEP 10: TRAINING MODEL")
print("="*70)
print(f"Batch size: 256")
print(f"Max epochs: 250 (increased for better convergence)")
print(f"Strategy: Simplified architecture + MILD weighting")
print()

history = model.fit(
    [n_train, k_train, m_train, P_train],
    y_train,
    sample_weight=sample_weights_train,  # MILD weights (1.0-2.0x)
    validation_data=([n_val, k_val, m_val, P_val], y_val),
    epochs=250,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

print()
print("Training completed!")
print()

# ============================================================================
# STEP 11: EVALUATE MODEL
# ============================================================================
print("="*70)
print("STEP 11: EVALUATING MODEL")
print("="*70)

model.load_weights('best_model_run5.h5')

y_pred_train = model.predict([n_train, k_train, m_train, P_train], verbose=0).flatten()
y_pred_val = model.predict([n_val, k_val, m_val, P_val], verbose=0).flatten()

def compute_log2_mse(y_true, y_pred):
    epsilon = 1e-7
    y_true = np.maximum(y_true, epsilon)
    y_pred = np.maximum(y_pred, epsilon)
    log2_true = np.log2(y_true)
    log2_pred = np.log2(y_pred)
    return np.mean((log2_true - log2_pred) ** 2)

train_log2_mse = compute_log2_mse(y_train, y_pred_train)
val_log2_mse = compute_log2_mse(y_val, y_pred_val)

print(f"Training log2-MSE: {train_log2_mse:.6f}")
print(f"Validation log2-MSE: {val_log2_mse:.6f}")
print(f"Train-Val gap: {abs(train_log2_mse - val_log2_mse):.6f}")
print()

print("Prediction Statistics:")
print(f"  Train predictions - Min: {y_pred_train.min():.4f}, Max: {y_pred_train.max():.4f}")
print(f"  Val predictions - Min: {y_pred_val.min():.4f}, Max: {y_pred_val.max():.4f}")
print(f"  All predictions ≥ 1.0: {(y_pred_val.min() >= 1.0)}")
print()

# ============================================================================
# STEP 12: DETAILED PER-GROUP ANALYSIS vs BASELINES
# ============================================================================
print("="*70)
print("PER-GROUP PERFORMANCE ANALYSIS vs BASELINES")
print("="*70)

group_metrics = defaultdict(lambda: {'true': [], 'pred': []})

for i in range(len(y_val)):
    k = k_val[i, 0]
    m = m_val[i, 0]
    group_metrics[(k, m)]['true'].append(y_val[i])
    group_metrics[(k, m)]['pred'].append(y_pred_val[i])

# Save to run_5 folder
import os
os.makedirs('run_5', exist_ok=True)

# Reference values from Run 2 and Run 4
run2_reference = {
    (4, 2): 0.103043, (4, 3): 0.107653, (4, 4): 0.209655, (4, 5): 1.422106,
    (5, 2): 0.096024, (5, 3): 0.227183, (5, 4): 0.957722,
    (6, 2): 0.355892, (6, 3): 1.083320
}

run4_reference = {
    (4, 2): 0.167,  # DEGRADED 62% from Run 2!
    (4, 3): 0.115,
    (4, 4): 0.275,  # DEGRADED 31% from Run 2
    (4, 5): 1.379,  # Small improvement
    (5, 2): 0.152,  # DEGRADED 58% from Run 2!
    (5, 3): 0.282,  # DEGRADED 24% from Run 2
    (5, 4): 0.953,  # Small improvement (0.5%)
    (6, 2): 0.384,  # DEGRADED 8% from Run 2
    (6, 3): 1.072,  # Small improvement (1%)
}

with open('run_5/detailed_comparison.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("DETAILED PER-GROUP COMPARISON - RUN 5 vs RUN 2 vs RUN 4\n")
    f.write("="*70 + "\n\n")

    f.write("BASELINES:\n")
    f.write("  Run 2: Balanced baseline (no aggressive weighting)\n")
    f.write("  Run 4: Aggressive weighting (5.0x) - degraded easy cases 60-70%!\n")
    f.write("  Run 5: MILD weighting (2.0x max) - should fix degradation\n\n")

    print("\nValidation Log2-MSE by (k,m) combination:")
    print("Run 2 | Run 4 | Run 5 | Change from Run 2 | Change from Run 4")
    print("-"*70)
    f.write("Validation Log2-MSE by (k,m) combination:\n")
    f.write("-"*70 + "\n")
    f.write(f"{'k,m':<8} {'Run 2':<10} {'Run 4':<10} {'Run 5':<10} {'vs R2':<12} {'vs R4':<12} {'n':<6}\n")
    f.write("-"*70 + "\n")

    easy_sum, medium_sum, hard_sum = 0, 0, 0
    easy_count, medium_count, hard_count = 0, 0, 0

    for (k, m), data in sorted(group_metrics.items()):
        true_vals = np.array(data['true'])
        pred_vals = np.array(data['pred'])
        group_log2_mse = compute_log2_mse(true_vals, pred_vals)

        run2_val = run2_reference.get((k, m), None)
        run4_val = run4_reference.get((k, m), None)

        if run2_val:
            change_r2 = ((run2_val - group_log2_mse) / run2_val) * 100
            change_r2_str = f"{change_r2:+.1f}%"
        else:
            change_r2_str = "N/A"

        if run4_val:
            change_r4 = ((run4_val - group_log2_mse) / run4_val) * 100
            change_r4_str = f"{change_r4:+.1f}%"
        else:
            change_r4_str = "N/A"

        run2_str = f"{run2_val:.6f}" if run2_val else "N/A"
        run4_str = f"{run4_val:.6f}" if run4_val else "N/A"

        output_line = f"k={k},m={m}  {run2_str:<10} {run4_str:<10} {group_log2_mse:.6f}  {change_r2_str:<12} {change_r4_str:<12} n={len(true_vals)}"
        print(output_line)
        f.write(output_line + "\n")

        # Track by difficulty
        if (k, m) in EASY_GROUPS:
            easy_sum += group_log2_mse
            easy_count += 1
        elif (k, m) in MEDIUM_GROUPS:
            medium_sum += group_log2_mse
            medium_count += 1
        elif (k, m) in HARD_GROUPS:
            hard_sum += group_log2_mse
            hard_count += 1

    print()
    f.write("\n")
    f.write("="*70 + "\n")
    f.write("SUMMARY BY DIFFICULTY TIER\n")
    f.write("="*70 + "\n")

    if easy_count > 0:
        easy_avg = easy_sum / easy_count
        f.write(f"Easy cases average: {easy_avg:.6f} (Target: < 0.15)\n")
        print(f"Easy cases average: {easy_avg:.6f} (Target: < 0.15)")

    if medium_count > 0:
        medium_avg = medium_sum / medium_count
        f.write(f"Medium cases average: {medium_avg:.6f} (Target: 0.20-0.30)\n")
        print(f"Medium cases average: {medium_avg:.6f} (Target: 0.20-0.30)")

    if hard_count > 0:
        hard_avg = hard_sum / hard_count
        f.write(f"Hard cases average: {hard_avg:.6f} (Target: 0.80-1.30)\n")
        print(f"Hard cases average: {hard_avg:.6f} (Target: 0.80-1.30)")

    f.write(f"\nOverall average: {val_log2_mse:.6f} (Target: < 0.374)\n")

print()

# ============================================================================
# STEP 13: GENERATE PLOTS
# ============================================================================
print("STEP 13: Generating Plots")
print("-"*70)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Log2-MSE Loss', fontsize=12)
plt.title('Training History - Run 5', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
start_epoch = int(len(history.history['loss']) * 0.2)
plt.plot(range(start_epoch, len(history.history['loss'])),
         history.history['loss'][start_epoch:],
         label='Train Loss', linewidth=2)
plt.plot(range(start_epoch, len(history.history['val_loss'])),
         history.history['val_loss'][start_epoch:],
         label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Log2-MSE Loss', fontsize=12)
plt.title('Training History (Last 80%)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('run_5/training_history.png', dpi=150, bbox_inches='tight')
print("Saved: run_5/training_history.png")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_val, alpha=0.3, s=10)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True m-Height', fontsize=12)
plt.ylabel('Predicted m-Height', fontsize=12)
plt.title('Predictions vs True Values (Validation Set)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_val, y_pred_val, alpha=0.3, s=10)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True m-Height (log scale)', fontsize=12)
plt.ylabel('Predicted m-Height (log scale)', fontsize=12)
plt.title('Predictions vs True Values - Log Scale', fontsize=14, fontweight='bold')
plt.xscale('log')
plt.yscale('log')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('run_5/predictions_scatter.png', dpi=150, bbox_inches='tight')
print("Saved: run_5/predictions_scatter.png")

plt.close('all')
print()

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================
print("="*70)
print("FINAL RESULTS SUMMARY - RUN 5")
print("="*70)
print(f"Training samples: {len(y_train):,}")
print(f"Validation samples: {len(y_val):,}")
print(f"Model parameters: {model.count_params():,}")
print()
print(f"Training log2-MSE:   {train_log2_mse:.6f}")
print(f"Validation log2-MSE: {val_log2_mse:.6f}")
print(f"Train-Val gap:       {abs(train_log2_mse - val_log2_mse):.6f}")
print(f"Target to beat:      0.374000")
print()

if val_log2_mse < 0.374:
    improvement = ((0.374 - val_log2_mse) / 0.374) * 100
    print(f"✅ SUCCESS! Beat target by {improvement:.1f}%")
    print(f"   Improvement: {0.374 - val_log2_mse:.6f}")
else:
    deficit = ((val_log2_mse - 0.374) / 0.374) * 100
    print(f"⚠️  Did not beat target (worse by {deficit:.1f}%)")
    print(f"   Need to improve by: {val_log2_mse - 0.374:.6f}")

print()
print("All predictions ≥ 1.0:", "✅ Yes" if y_pred_val.min() >= 1.0 else "❌ No")
print(f"Prediction range: [{y_pred_val.min():.2f}, {y_pred_val.max():.2f}]")
print()

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================
print("="*70)
print("COMPREHENSIVE COMPARISON: RUN 2 vs RUN 4 vs RUN 5")
print("="*70)
print("Run 2: Val log2-MSE = ~0.48 (baseline, balanced)")
print("Run 4: Val log2-MSE = 0.538 (aggressive weighting - degraded easy cases!)")
print(f"Run 5: Val log2-MSE = {val_log2_mse:.6f} (MILD weighting - FIX!)")
print()

print("KEY INSIGHTS:")
print("✅ Run 4 Problem: Aggressive 5.0x weighting degraded easy cases 60-70%")
print("✅ Run 5 Fix: Reduced to 2.0x max weighting")
print("✅ Run 5 Improvements:")
print("   - Simplified architecture (removed attention)")
print("   - LayerNorm for better gradient flow")
print("   - Progressive dropout pattern")
print("   - Better learning rate scheduling")
print()

print("="*70)
print("DELIVERABLES SAVED TO run_5/")
print("="*70)
print("  1. best_model_run5.h5 - Trained model weights")
print("  2. run_5/training_history.png - Loss curves")
print("  3. run_5/predictions_scatter.png - Prediction quality plots")
print("  4. run_5/detailed_comparison.txt - Complete comparison vs Run 2 and Run 4")
print("="*70)
print()
print("Training complete! 🎉")
