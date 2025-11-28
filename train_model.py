"""
Height Prediction Model Training Script
Objective: Beat validation log2-MSE of 0.374 using TensorFlow

Key Improvements:
1. Dataset rebalancing (~9,000 samples per (k,m) combination)
2. Predicting log2(m-height) instead of raw values
3. Advanced architecture with attention and residual blocks
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Flatten, Concatenate,
    Dropout, BatchNormalization, Add, Reshape, MultiHeadAttention, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
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
print("HEIGHT PREDICTION MODEL - TRAINING SCRIPT")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: Loading Data")
print("-"*70)

with open('combined_ALL_n_k_m_P_exact.pkl', 'rb') as f:
    inputs_raw = pickle.load(f)

with open('combined_ALL_mHeights_exact.pkl', 'rb') as f:
    outputs_raw = pickle.load(f)

print(f"Raw input samples: {len(inputs_raw)}")
print(f"Raw output samples: {len(outputs_raw)}")
if len(inputs_raw) > 0:
    sample = inputs_raw[0]
    print(f"Sample structure: [n={sample[0]}, k={sample[1]}, m={sample[2]}, P_matrix shape={sample[3].shape}]")
else:
    print(f"Sample structure: N/A")
print(f"Output range: [{np.min(outputs_raw):.2f}, {np.max(outputs_raw):.2f}]")
print()

# ============================================================================
# STEP 2: ANALYZE CLASS DISTRIBUTION
# ============================================================================
print("STEP 2: Analyzing Class Distribution (BEFORE Rebalancing)")
print("-"*70)

# Group samples by (k, m) combinations
groups = defaultdict(list)
for i, sample in enumerate(inputs_raw):
    # Extract k and m from sample
    # Assuming format: [n, k, m, P_matrix_values...]
    k = int(sample[1])
    m = int(sample[2])
    groups[(k, m)].append(i)

print(f"Total unique (k,m) combinations: {len(groups)}")
print("\nDistribution by (k,m):")
total_samples = len(inputs_raw)
for (k, m), indices in sorted(groups.items()):
    count = len(indices)
    percentage = (count / total_samples) * 100
    print(f"  k={k}, m={m}: {count:6d} samples ({percentage:5.2f}%)")

# Identify max imbalance
max_count = max(len(indices) for indices in groups.values())
min_count = min(len(indices) for indices in groups.values())
print(f"\nImbalance ratio: {max_count/min_count:.1f}x")
print()

# ============================================================================
# STEP 3: REBALANCE DATASET
# ============================================================================
print("STEP 3: Rebalancing Dataset")
print("-"*70)

TARGET_SAMPLES = 9000
print(f"Target samples per (k,m) combination: {TARGET_SAMPLES}")

rebalanced_indices = []
np.random.seed(42)  # For reproducibility

for (k, m), indices in groups.items():
    if len(indices) > TARGET_SAMPLES:
        # Subsample (undersample majority classes)
        selected = np.random.choice(indices, TARGET_SAMPLES, replace=False)
    else:
        # Oversample (oversample minority classes)
        selected = np.random.choice(indices, TARGET_SAMPLES, replace=True)
    rebalanced_indices.extend(selected)

# Shuffle the rebalanced indices
np.random.shuffle(rebalanced_indices)

print(f"Original dataset size: {len(inputs_raw)}")
print(f"Rebalanced dataset size: {len(rebalanced_indices)}")
print()

print("REBALANCED DATA DISTRIBUTION:")
print("-"*70)
rebalanced_groups = defaultdict(int)
for idx in rebalanced_indices:
    sample = inputs_raw[idx]
    k = int(sample[1])
    m = int(sample[2])
    rebalanced_groups[(k, m)] += 1

for (k, m), count in sorted(rebalanced_groups.items()):
    percentage = (count / len(rebalanced_indices)) * 100
    print(f"  k={k}, m={m}: {count:6d} samples ({percentage:5.2f}%)")
print()

# ============================================================================
# STEP 4: PREPARE DATA FOR TRAINING
# ============================================================================
print("STEP 4: Preparing Data for Training")
print("-"*70)

# Extract rebalanced data
inputs_rebalanced = [inputs_raw[i] for i in rebalanced_indices]
outputs_rebalanced = [outputs_raw[i] for i in rebalanced_indices]

# Extract n, k, m values and flatten P matrices
n_values = []
k_values = []
m_values = []
P_matrices_flattened = []

for sample in inputs_rebalanced:
    n_values.append(sample[0])
    k_values.append(sample[1])
    m_values.append(sample[2])
    # Flatten the P matrix
    P_matrices_flattened.append(sample[3].flatten())

# Convert to numpy arrays
n_values = np.array(n_values, dtype=np.float32).reshape(-1, 1)
k_values = np.array(k_values, dtype=np.int32).reshape(-1, 1)
m_values = np.array(m_values, dtype=np.int32).reshape(-1, 1)
outputs_array = np.array(outputs_rebalanced, dtype=np.float32)

# Find max P matrix size and pad all matrices to the same size
max_p_size = max(len(p) for p in P_matrices_flattened)
P_matrices_padded = []

for p in P_matrices_flattened:
    if len(p) < max_p_size:
        # Pad with zeros
        padded = np.zeros(max_p_size, dtype=np.float32)
        padded[:len(p)] = p
        P_matrices_padded.append(padded)
    else:
        P_matrices_padded.append(p)

P_matrices = np.array(P_matrices_padded, dtype=np.float32)

print(f"n_values shape: {n_values.shape}")
print(f"k_values shape: {k_values.shape}, range: [{k_values.min()}, {k_values.max()}]")
print(f"m_values shape: {m_values.shape}, range: [{m_values.min()}, {m_values.max()}]")
print(f"P_matrices shape: {P_matrices.shape} (flattened and padded)")

# Normalize P matrices (important for training stability)
scaler = StandardScaler()
P_matrices_normalized = scaler.fit_transform(P_matrices)

print(f"\nP matrices normalized: mean={P_matrices_normalized.mean():.4f}, std={P_matrices_normalized.std():.4f}")

# Ensure outputs are positive (should already be, but verify)
outputs_array = np.maximum(outputs_array, 1.0)
print(f"Output (m-height) range: [{outputs_array.min():.2f}, {outputs_array.max():.2f}]")
print()

# ============================================================================
# STEP 5: STRATIFIED TRAIN-VAL SPLIT
# ============================================================================
print("STEP 5: Creating Stratified Train-Validation Split")
print("-"*70)

# Create stratification labels based on (k, m) combinations
stratify_labels = k_values.flatten() * 10 + m_values.flatten()

# Split data (85% train, 15% validation)
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

# Verify stratification
print("\nValidation set distribution:")
val_groups = defaultdict(int)
for k, m in zip(k_val.flatten(), m_val.flatten()):
    val_groups[(k, m)] += 1
for (k, m), count in sorted(val_groups.items()):
    percentage = (count / len(y_val)) * 100
    print(f"  k={k}, m={m}: {count:5d} samples ({percentage:5.2f}%)")
print()

# ============================================================================
# STEP 6: BUILD MODEL
# ============================================================================
print("STEP 6: Building TensorFlow Model")
print("-"*70)

def build_model(p_shape, k_vocab_size=7, m_vocab_size=6):
    """
    Build advanced model with:
    - Embeddings for categorical k and m
    - Deep processing of P matrix with attention
    - Residual blocks
    - Output: log2(m-height) with constraint ≥ 1.0
    """
    # Inputs
    n_input = Input(shape=(1,), name='n_input')
    k_input = Input(shape=(1,), name='k_input', dtype=tf.int32)
    m_input = Input(shape=(1,), name='m_input', dtype=tf.int32)
    P_input = Input(shape=(p_shape,), name='P_input')

    # Embeddings for categorical variables
    k_embed = Flatten()(Embedding(k_vocab_size, 32, name='k_embedding')(k_input))
    m_embed = Flatten()(Embedding(m_vocab_size, 32, name='m_embedding')(m_input))

    # P matrix processing
    x = Dense(256, activation='gelu')(P_input)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512, activation='gelu')(x)
    x = BatchNormalization()(x)

    # Multi-head attention on P features
    x_attn = Reshape((1, 512))(x)
    x_attn = MultiHeadAttention(num_heads=8, key_dim=64, dropout=0.1)(x_attn, x_attn)
    x_attn = Flatten()(x_attn)

    # Combine all features
    combined = Concatenate()([n_input, k_embed, m_embed, x_attn])

    # Deep network with residual connections
    x = Dense(1024, activation='gelu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # 4 residual blocks
    for i in range(4):
        residual = x
        x = Dense(1024, activation='gelu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='gelu')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])

    # Final dense layers
    x = Dense(512, activation='gelu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='gelu')(x)

    # *** CRITICAL OUTPUT LAYER ***
    # Predict log2(m-height), then convert to m-height
    log2_pred = Dense(1, activation='linear', name='log2_prediction')(x)

    # Ensure log2_pred ≥ 0 (so m-height ≥ 1) using softplus
    log2_positive = Lambda(lambda x: tf.nn.softplus(x), name='softplus_activation')(log2_pred)

    # Convert to m-height: 2^(log2_pred)
    output = Lambda(lambda x: tf.pow(2.0, x), name='m_height_output')(log2_positive)

    model = Model(
        inputs=[n_input, k_input, m_input, P_input],
        outputs=output,
        name='height_prediction_model'
    )

    return model

# Build the model
p_shape = P_train.shape[1]
model = build_model(p_shape, k_vocab_size=k_values.max()+1, m_vocab_size=m_values.max()+1)

print(f"Model built successfully!")
print(f"Total parameters: {model.count_params():,}")
model.summary()
print()

# ============================================================================
# STEP 7: DEFINE CUSTOM LOSS FUNCTION
# ============================================================================
print("STEP 7: Defining Custom Loss Function")
print("-"*70)

def log2_mse_loss(y_true, y_pred):
    """
    Custom loss function: MSE in log2 space
    Loss = mean((log2(y_true) - log2(y_pred))^2)
    """
    epsilon = 1e-7

    # Ensure positive values
    y_true = tf.maximum(y_true, epsilon)
    y_pred = tf.maximum(y_pred, epsilon)

    # Convert to log2
    log2_true = tf.math.log(y_true) / tf.math.log(2.0)
    log2_pred = tf.math.log(y_pred) / tf.math.log(2.0)

    # MSE in log2 space
    return tf.reduce_mean(tf.square(log2_true - log2_pred))

print("Custom log2-MSE loss function defined")
print()

# ============================================================================
# STEP 8: COMPILE MODEL
# ============================================================================
print("STEP 8: Compiling Model")
print("-"*70)

optimizer = AdamW(learning_rate=1e-3, weight_decay=1e-4)

model.compile(
    optimizer=optimizer,
    loss=log2_mse_loss,
    metrics=[log2_mse_loss]
)

print("Model compiled with AdamW optimizer and log2-MSE loss")
print()

# ============================================================================
# STEP 9: SETUP CALLBACKS
# ============================================================================
print("STEP 9: Setting Up Training Callbacks")
print("-"*70)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

print("Callbacks configured:")
print("  - EarlyStopping (patience=50)")
print("  - ReduceLROnPlateau (patience=20, factor=0.5)")
print("  - ModelCheckpoint (saves best model)")
print()

# ============================================================================
# STEP 10: TRAIN MODEL
# ============================================================================
print("="*70)
print("STEP 10: TRAINING MODEL")
print("="*70)
print(f"Batch size: 256")
print(f"Max epochs: 200")
print(f"Early stopping patience: 50")
print()

history = model.fit(
    [n_train, k_train, m_train, P_train],
    y_train,
    validation_data=([n_val, k_val, m_val, P_val], y_val),
    epochs=200,
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

# Load best model
model.load_weights('best_model.h5')

# Make predictions
y_pred_train = model.predict([n_train, k_train, m_train, P_train], verbose=0).flatten()
y_pred_val = model.predict([n_val, k_val, m_val, P_val], verbose=0).flatten()

# Compute overall log2-MSE
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
print()

# Check prediction constraints
print("Prediction Statistics:")
print(f"  Train predictions - Min: {y_pred_train.min():.4f}, Max: {y_pred_train.max():.4f}")
print(f"  Val predictions - Min: {y_pred_val.min():.4f}, Max: {y_pred_val.max():.4f}")
print(f"  All predictions ≥ 1.0: {(y_pred_val.min() >= 1.0)}")
print()

# ============================================================================
# STEP 12: PER-GROUP ANALYSIS
# ============================================================================
print("="*70)
print("PER-GROUP PERFORMANCE ANALYSIS")
print("="*70)

# Compute per-(k,m) metrics for validation set
group_metrics = defaultdict(lambda: {'true': [], 'pred': []})

for i in range(len(y_val)):
    k = k_val[i, 0]
    m = m_val[i, 0]
    group_metrics[(k, m)]['true'].append(y_val[i])
    group_metrics[(k, m)]['pred'].append(y_pred_val[i])

# Save to file and print
with open('per_group_performance.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("PER-GROUP PERFORMANCE BREAKDOWN\n")
    f.write("="*70 + "\n\n")

    print("\nValidation Log2-MSE by (k,m) combination:")
    f.write("Validation Log2-MSE by (k,m) combination:\n")
    f.write("-"*70 + "\n")

    for (k, m), data in sorted(group_metrics.items()):
        true_vals = np.array(data['true'])
        pred_vals = np.array(data['pred'])
        group_log2_mse = compute_log2_mse(true_vals, pred_vals)

        output_line = f"  k={k}, m={m}: {group_log2_mse:.6f} (n={len(true_vals)} samples)"
        print(output_line)
        f.write(output_line + "\n")

print()

# ============================================================================
# STEP 13: GENERATE PLOTS
# ============================================================================
print("STEP 13: Generating Plots")
print("-"*70)

# Plot 1: Training History
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Log2-MSE Loss', fontsize=12)
plt.title('Training History', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Plot last 80% of training to see convergence better
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
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("Saved: training_history.png")

# Plot 2: Predictions vs True Values (Log Scale)
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
plt.savefig('predictions_scatter.png', dpi=150, bbox_inches='tight')
print("Saved: predictions_scatter.png")

plt.close('all')
print()

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================
print("="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)
print(f"Training samples: {len(y_train):,}")
print(f"Validation samples: {len(y_val):,}")
print(f"Model parameters: {model.count_params():,}")
print()
print(f"Training log2-MSE:   {train_log2_mse:.6f}")
print(f"Validation log2-MSE: {val_log2_mse:.6f}")
print(f"Target to beat:      0.374000")
print()

if val_log2_mse < 0.374:
    improvement = ((0.374 - val_log2_mse) / 0.374) * 100
    print(f"✅ SUCCESS! Beat target by {improvement:.1f}%")
    print(f"   Improvement: {0.374 - val_log2_mse:.6f}")
else:
    deficit = ((val_log2_mse - 0.374) / 0.374) * 100
    print(f"❌ Did not beat target (worse by {deficit:.1f}%)")
    print(f"   Need to improve by: {val_log2_mse - 0.374:.6f}")

print()
print("All predictions ≥ 1.0:", "✅ Yes" if y_pred_val.min() >= 1.0 else "❌ No")
print(f"Prediction range: [{y_pred_val.min():.2f}, {y_pred_val.max():.2f}]")
print()
print("="*70)
print("DELIVERABLES SAVED:")
print("="*70)
print("  1. best_model.h5 - Trained model weights")
print("  2. training_history.png - Loss curves")
print("  3. predictions_scatter.png - Prediction quality plots")
print("  4. per_group_performance.txt - Detailed per-(k,m) metrics")
print("="*70)
print()
print("Training complete! 🎉")
