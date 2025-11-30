"""
Height Prediction Model Training Script - Run 7
Objective: Fix Run 6 data leakage and improve model generalization

Run 7 Strategy - Critical Fixes:
1. ✅ Move augmentation AFTER train-val split (FIX DATA LEAKAGE!)
2. ✅ Reduce augmentation to 3/2/1 copies for hard/medium/easy
3. ✅ Increase noise scale to 5% (better generalization)
4. ✅ Add early stopping on easy case performance
5. ✅ Implement balanced loss function
6. ✅ Create separate tier metrics tracking
7. ✅ Add L2 regularization

Expected Performance:
- Easy cases: < 0.15 log2-MSE (NO degradation from Run 2!)
- Medium cases: 0.20-0.30 log2-MSE
- Hard cases: 0.70-1.20 log2-MSE
- Overall: 0.28-0.35 (beat 0.374 target with VALID metrics)
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Flatten, Concatenate,
    Dropout, Add, LayerNormalization, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    Callback, LearningRateScheduler
)
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras import regularizers
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
print("HEIGHT PREDICTION MODEL - RUN 7 (DATA LEAKAGE FIXED)")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print()

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("STEP 1: Loading Data")
print("-"*70)

with open('data/combined_final_n_k_m_P.pkl', 'rb') as f:
    inputs_raw = pickle.load(f)

with open('data/combined_final_mHeights.pkl', 'rb') as f:
    outputs_raw = pickle.load(f)

print(f"Raw input samples: {len(inputs_raw)}")
print(f"Raw output samples: {len(outputs_raw)}")
if len(inputs_raw) > 0:
    sample = inputs_raw[0]
    print(f"Sample structure: [n={sample[0]}, k={sample[1]}, m={sample[2]}, P_matrix shape={sample[3].shape}]")
print(f"Output range: [{np.min(outputs_raw):.2f}, {np.max(outputs_raw):.2f}]")
print()

# ============================================================================
# STEP 2: TRAIN-VAL SPLIT (BEFORE AUGMENTATION!)
# ============================================================================
print("STEP 2: Train-Val Split (BEFORE Augmentation - Critical Fix!)")
print("-"*70)

# Create stratification labels
stratify_labels = np.array([k * 10 + m for n, k, m, P in inputs_raw])

# Split on ORIGINAL data only
(inputs_train_orig, inputs_val,
 outputs_train_orig, outputs_val) = train_test_split(
    inputs_raw,
    outputs_raw,
    test_size=0.15,
    random_state=42,
    stratify=stratify_labels
)

print(f"Training samples (original): {len(inputs_train_orig)}")
print(f"Validation samples (original, NO augmentation): {len(inputs_val)}")
print()
print("✅ CRITICAL FIX: Validation set contains ONLY original samples")
print("✅ No data leakage - augmented copies stay in training set only")
print()

# ============================================================================
# STEP 3: DATA AUGMENTATION (TRAINING SET ONLY!)
# ============================================================================
print("STEP 3: Data Augmentation (Training Set ONLY)")
print("-"*70)

def augment_sample(n, k, m, P_matrix, m_height, noise_scale=0.05):
    """
    Augment a single sample with 5% noise (increased from 2%)

    Args:
        n: float value
        k, m: integers
        P_matrix: probability matrix
        m_height: target value
        noise_scale: scale of noise to add (default 5%)

    Returns:
        Augmented (n, k, m, P_matrix), m_height
    """
    # Add noise to n (±5%)
    n_aug = n * (1 + np.random.uniform(-noise_scale, noise_scale))
    n_aug = max(1.0, n_aug)

    # Add noise to P matrix
    P_aug = P_matrix.copy()
    if P_aug.size > 0:
        noise = np.random.normal(0, noise_scale, P_aug.shape)
        P_aug = P_aug + noise
        P_aug = np.clip(P_aug, 0, 1)

        # Renormalize rows to sum to 1
        if len(P_aug.shape) == 2:
            row_sums = P_aug.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-7)
            P_aug = P_aug / row_sums

    return (n_aug, k, m, P_aug), m_height


# Define complexity groups
EASY_GROUPS = [(4, 2), (4, 3), (5, 2)]
MEDIUM_GROUPS = [(4, 4), (5, 3), (6, 2)]
HARD_GROUPS = [(5, 4), (6, 3), (4, 5)]

# Augmentation config - REDUCED from Run 6
augmentation_config = {
    # Hard cases: +3 copies (was 50 in Run 6!)
    (5, 4): 3,
    (6, 3): 3,
    (4, 5): 3,

    # Medium cases: +2 copies (was 25 in Run 6!)
    (4, 4): 2,
    (5, 3): 2,
    (6, 2): 2,

    # Easy cases: +1 copy (was 0 in Run 6 - caused degradation!)
    (4, 2): 1,
    (4, 3): 1,
    (5, 2): 1,
}

# Augment ONLY training data
inputs_train_aug = []
outputs_train_aug = []

print("Augmenting training data...")
print("Strategy:")
print("  - Hard cases (k=5,m=4 | k=6,m=3 | k=4,m=5): +3 copies (was 50!)")
print("  - Medium cases (k=4,m=4 | k=5,m=3 | k=6,m=2): +2 copies (was 25!)")
print("  - Easy cases (k=4,m=2 | k=4,m=3 | k=5,m=2): +1 copy (was 0!)")
print("  - Noise scale: 5% (was 2%)")
print()

for sample, target in zip(inputs_train_orig, outputs_train_orig):
    n, k, m, P_matrix = sample

    # Always keep original
    inputs_train_aug.append(sample)
    outputs_train_aug.append(target)

    # Add augmented copies
    num_copies = augmentation_config.get((k, m), 0)

    for _ in range(num_copies):
        aug_sample, aug_target = augment_sample(n, k, m, P_matrix, target, noise_scale=0.05)
        inputs_train_aug.append(aug_sample)
        outputs_train_aug.append(aug_target)

print(f"Training samples (original): {len(inputs_train_orig)}")
print(f"Training samples (augmented): {len(inputs_train_aug)}")
print(f"Augmentation ratio: {len(inputs_train_aug) / len(inputs_train_orig):.2f}x")
print()
print("✅ Validation set: {0} samples (NO augmentation)".format(len(inputs_val)))
print()

# ============================================================================
# STEP 4: PREPARE DATA FOR TRAINING
# ============================================================================
print("STEP 4: Preparing Data for Training")
print("-"*70)

def prepare_data(inputs_list, outputs_list):
    """Convert list of samples to arrays"""
    n_values = []
    k_values = []
    m_values = []
    P_matrices_flattened = []

    for sample in inputs_list:
        n_values.append(sample[0])
        k_values.append(sample[1])
        m_values.append(sample[2])
        P_matrices_flattened.append(sample[3].flatten())

    n_values = np.array(n_values, dtype=np.float32).reshape(-1, 1)
    k_values = np.array(k_values, dtype=np.int32).reshape(-1, 1)
    m_values = np.array(m_values, dtype=np.int32).reshape(-1, 1)
    outputs_array = np.array(outputs_list, dtype=np.float32)

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

    return n_values, k_values, m_values, P_matrices, outputs_array

# Prepare training data (augmented)
n_train, k_train, m_train, P_train_raw, y_train = prepare_data(inputs_train_aug, outputs_train_aug)

# Prepare validation data (original only)
n_val, k_val, m_val, P_val_raw, y_val = prepare_data(inputs_val, outputs_val)

# Normalize P matrices using ONLY training statistics
scaler = StandardScaler()
P_train = scaler.fit_transform(P_train_raw)
P_val = scaler.transform(P_val_raw)  # Use training statistics

# Ensure outputs >= 1.0
y_train = np.maximum(y_train, 1.0)
y_val = np.maximum(y_val, 1.0)

print(f"Training set:")
print(f"  n_values: {n_train.shape}")
print(f"  k_values: {k_train.shape}, range: [{k_train.min()}, {k_train.max()}]")
print(f"  m_values: {m_train.shape}, range: [{m_train.min()}, {m_train.max()}]")
print(f"  P_matrices: {P_train.shape}")
print(f"  y_train: {y_train.shape}, range: [{y_train.min():.2f}, {y_train.max():.2f}]")
print()
print(f"Validation set:")
print(f"  n_values: {n_val.shape}")
print(f"  k_values: {k_val.shape}, range: [{k_val.min()}, {k_val.max()}]")
print(f"  m_values: {m_val.shape}, range: [{m_val.min()}, {m_val.max()}]")
print(f"  P_matrices: {P_val.shape}")
print(f"  y_val: {y_val.shape}, range: [{y_val.min():.2f}, {y_val.max():.2f}]")
print()

# ============================================================================
# STEP 5: BUILD MODEL WITH L2 REGULARIZATION
# ============================================================================
print("STEP 5: Building Model with L2 Regularization")
print("-"*70)

def build_model_run7(p_shape, k_vocab_size=7, m_vocab_size=6):
    """
    Run 7 model with L2 regularization

    Improvements:
    - L2 regularization (1e-4) on all Dense layers
    - LayerNorm for stable training
    - 2 residual blocks
    - Progressive width reduction
    - Progressive dropout
    """

    # L2 regularizer
    l2_reg = regularizers.l2(1e-4)

    # Inputs
    n_input = Input(shape=(1,), name='n')
    k_input = Input(shape=(1,), name='k', dtype=tf.int32)
    m_input = Input(shape=(1,), name='m', dtype=tf.int32)
    P_input = Input(shape=(p_shape,), name='P_flat')

    # Embeddings with L2
    k_embed = Flatten()(Embedding(
        k_vocab_size, 32,
        embeddings_regularizer=l2_reg,
        name='k_embedding'
    )(k_input))

    m_embed = Flatten()(Embedding(
        m_vocab_size, 32,
        embeddings_regularizer=l2_reg,
        name='m_embedding'
    )(m_input))

    # Initial P processing
    x = Dense(256, activation='gelu', kernel_regularizer=l2_reg, name='P_initial_1')(P_input)
    x = LayerNormalization(name='P_ln1')(x)
    x = Dropout(0.3)(x)

    x = Dense(512, activation='gelu', kernel_regularizer=l2_reg, name='P_initial_2')(x)
    x = LayerNormalization(name='P_ln2')(x)
    x = Dropout(0.2)(x)

    # Combine with embeddings
    x = Concatenate(name='combine_embeddings')([n_input, k_embed, m_embed, x])

    # Initial dense layer
    x = Dense(1024, activation='gelu', kernel_regularizer=l2_reg, name='main_dense1')(x)
    x = LayerNormalization(name='main_ln1')(x)
    x = Dropout(0.3)(x)

    # Residual Block 1
    residual = x
    x = Dense(1024, activation='gelu', kernel_regularizer=l2_reg, name='res1_dense1')(x)
    x = LayerNormalization(name='res1_ln1')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='gelu', kernel_regularizer=l2_reg, name='res1_dense2')(x)
    x = LayerNormalization(name='res1_ln2')(x)
    x = Add(name='res1_add')([x, residual])

    # Residual Block 2
    residual = x
    x = Dense(1024, activation='gelu', kernel_regularizer=l2_reg, name='res2_dense1')(x)
    x = LayerNormalization(name='res2_ln1')(x)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='gelu', kernel_regularizer=l2_reg, name='res2_dense2')(x)
    x = LayerNormalization(name='res2_ln2')(x)
    x = Add(name='res2_add')([x, residual])

    # Progressive width reduction
    x = Dense(512, activation='gelu', kernel_regularizer=l2_reg, name='reduction1')(x)
    x = LayerNormalization(name='reduction_ln1')(x)
    x = Dropout(0.2)(x)

    x = Dense(256, activation='gelu', kernel_regularizer=l2_reg, name='reduction2')(x)
    x = LayerNormalization(name='reduction_ln2')(x)
    x = Dropout(0.1)(x)

    x = Dense(128, activation='gelu', kernel_regularizer=l2_reg, name='reduction3')(x)
    x = LayerNormalization(name='reduction_ln3')(x)
    x = Dropout(0.1)(x)

    # Output: log2(m-height) then convert to m-height
    log2_pred = Dense(1, activation='linear', kernel_regularizer=l2_reg, name='log2_output')(x)
    log2_pred_positive = Lambda(lambda z: tf.nn.softplus(z), name='softplus')(log2_pred)
    m_height_pred = Lambda(lambda z: tf.pow(2.0, z), name='m_height')(log2_pred_positive)

    model = Model(
        inputs=[n_input, k_input, m_input, P_input],
        outputs=m_height_pred,
        name='run7_model'
    )

    return model

p_shape = P_train.shape[1]
model = build_model_run7(
    p_shape,
    k_vocab_size=k_train.max()+1,
    m_vocab_size=m_train.max()+1
)

print(f"Model built successfully!")
print(f"Total parameters: {model.count_params():,}")
print()
print("Architecture improvements:")
print("  ✅ L2 regularization (1e-4) on all Dense layers")
print("  ✅ LayerNorm for stable gradients")
print("  ✅ 2 residual blocks")
print("  ✅ Progressive width: 1024→512→256→128")
print("  ✅ Progressive dropout: 0.3→0.2→0.1")
print()

# ============================================================================
# STEP 6: BALANCED LOSS FUNCTION
# ============================================================================
print("STEP 6: Defining Balanced Loss Function")
print("-"*70)

# Group loss weights - prioritize easy cases to prevent degradation
GROUP_LOSS_WEIGHTS = {
    # Easy groups: Higher weight (3.0x) to prevent degradation
    (4, 2): 3.0,
    (4, 3): 3.0,
    (5, 2): 3.0,

    # Medium groups: Moderate weight (1.5x)
    (4, 4): 1.5,
    (5, 3): 1.5,
    (6, 2): 1.5,

    # Hard groups: Normal weight (1.0x)
    (5, 4): 1.0,
    (6, 3): 1.0,
    (4, 5): 1.0,
}

def balanced_log2_mse_loss(y_true, y_pred):
    """
    Balanced log2-MSE loss with group weighting

    This is the base loss - sample weights are applied separately
    """
    epsilon = 1e-7
    y_true = tf.maximum(y_true, epsilon)
    y_pred = tf.maximum(y_pred, epsilon)

    log2_true = tf.math.log(y_true) / tf.math.log(2.0)
    log2_pred = tf.math.log(y_pred) / tf.math.log(2.0)

    return tf.reduce_mean(tf.square(log2_true - log2_pred))

print("Balanced loss function defined")
print()
print("Group loss weights (applied via sample weights):")
for (k, m), weight in sorted(GROUP_LOSS_WEIGHTS.items()):
    difficulty = "Easy" if (k, m) in EASY_GROUPS else ("Medium" if (k, m) in MEDIUM_GROUPS else "Hard")
    print(f"  k={k}, m={m} ({difficulty:6s}): {weight:.1f}x")
print()

# Compute sample weights for training
sample_weights_train = np.ones(len(y_train), dtype=np.float32)

for i, (k, m) in enumerate(zip(k_train, m_train)):
    key = (k[0], m[0])
    if key in GROUP_LOSS_WEIGHTS:
        sample_weights_train[i] = GROUP_LOSS_WEIGHTS[key]

print(f"Sample weights applied: {np.unique(sample_weights_train)}")
print()

# ============================================================================
# STEP 7: COMPILE MODEL
# ============================================================================
print("STEP 7: Compiling Model")
print("-"*70)

optimizer = AdamW(
    learning_rate=1e-3,
    weight_decay=1e-4,
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer,
    loss=balanced_log2_mse_loss,
    metrics=[balanced_log2_mse_loss]
)

print("Model compiled with AdamW optimizer")
print(f"  Learning rate: 1e-3")
print(f"  Weight decay: 1e-4")
print(f"  Gradient clipping: 1.0")
print()

# ============================================================================
# STEP 8: CALLBACKS WITH EASY CASE MONITORING
# ============================================================================
print("STEP 8: Setting Up Callbacks with Easy Case Monitoring")
print("-"*70)

def compute_log2_mse(y_true, y_pred):
    """Compute log2-MSE"""
    epsilon = 1e-7
    y_true = np.maximum(y_true, epsilon)
    y_pred = np.maximum(y_pred, epsilon)
    log2_true = np.log2(y_true)
    log2_pred = np.log2(y_pred)
    return np.mean((log2_true - log2_pred) ** 2)


class TierMetricsCallback(Callback):
    """
    Monitor performance on Easy/Medium/Hard tiers separately
    Stop training if easy cases degrade beyond threshold
    """

    def __init__(self, val_data, easy_threshold=0.15, patience=5):
        super().__init__()
        self.n_val, self.k_val, self.m_val, self.P_val, self.y_val = val_data
        self.easy_threshold = easy_threshold
        self.patience = patience
        self.easy_violations = 0
        self.best_balanced_metric = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions
        y_pred = self.model.predict(
            [self.n_val, self.k_val, self.m_val, self.P_val],
            verbose=0
        ).flatten()

        # Compute per-group metrics
        easy_losses = []
        medium_losses = []
        hard_losses = []

        for (k, m) in EASY_GROUPS:
            mask = (self.k_val.flatten() == k) & (self.m_val.flatten() == m)
            if mask.sum() > 0:
                loss = compute_log2_mse(self.y_val[mask], y_pred[mask])
                easy_losses.append(loss)

        for (k, m) in MEDIUM_GROUPS:
            mask = (self.k_val.flatten() == k) & (self.m_val.flatten() == m)
            if mask.sum() > 0:
                loss = compute_log2_mse(self.y_val[mask], y_pred[mask])
                medium_losses.append(loss)

        for (k, m) in HARD_GROUPS:
            mask = (self.k_val.flatten() == k) & (self.m_val.flatten() == m)
            if mask.sum() > 0:
                loss = compute_log2_mse(self.y_val[mask], y_pred[mask])
                hard_losses.append(loss)

        # Compute tier averages
        easy_avg = np.mean(easy_losses) if easy_losses else 0
        medium_avg = np.mean(medium_losses) if medium_losses else 0
        hard_avg = np.mean(hard_losses) if hard_losses else 0

        # Balanced metric (equal weight to each tier)
        balanced_metric = (easy_avg + medium_avg + hard_avg) / 3

        # Track best balanced metric
        if balanced_metric < self.best_balanced_metric:
            self.best_balanced_metric = balanced_metric

        # Print tier metrics
        print(f"\n  Tier Metrics - Easy: {easy_avg:.4f} | Medium: {medium_avg:.4f} | Hard: {hard_avg:.4f} | Balanced: {balanced_metric:.4f}")

        # Check easy case threshold
        if easy_avg > self.easy_threshold:
            self.easy_violations += 1
            print(f"  ⚠️  WARNING: Easy cases at {easy_avg:.4f} (threshold: {self.easy_threshold}) - violation {self.easy_violations}/{self.patience}")

            if self.easy_violations >= self.patience:
                print(f"\n  🛑 STOPPING: Easy cases degraded for {self.patience} consecutive epochs")
                self.model.stop_training = True
        else:
            self.easy_violations = 0  # Reset counter
            print(f"  ✅ Easy cases OK ({easy_avg:.4f} < {self.easy_threshold})")


def lr_schedule(epoch, lr):
    """Exponential decay schedule"""
    if epoch < 10:
        return lr  # Warmup phase
    else:
        return lr * 0.95  # Decay by 5% each epoch

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)

# Create tier metrics callback
tier_callback = TierMetricsCallback(
    val_data=(n_val, k_val, m_val, P_val, y_val),
    easy_threshold=0.15,
    patience=5
)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=40,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=15,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model_run7.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    lr_scheduler,
    tier_callback  # NEW: Monitor easy case performance
]

print("Callbacks configured:")
print("  - EarlyStopping (patience=40)")
print("  - ReduceLROnPlateau (patience=15, factor=0.7)")
print("  - ModelCheckpoint (best_model_run7.h5)")
print("  - LR scheduler (exponential decay)")
print("  - TierMetricsCallback (easy case monitoring, threshold=0.15)")
print()

# ============================================================================
# STEP 9: TRAIN MODEL
# ============================================================================
print("="*70)
print("STEP 9: TRAINING MODEL")
print("="*70)
print(f"Batch size: 256")
print(f"Max epochs: 250")
print(f"Training samples: {len(y_train):,} (augmented)")
print(f"Validation samples: {len(y_val):,} (original only)")
print()
print("Key improvements from Run 6:")
print("  ✅ Data augmentation AFTER split (no leakage!)")
print("  ✅ Reduced augmentation (3/2/1 vs 50/25/0)")
print("  ✅ Increased noise (5% vs 2%)")
print("  ✅ Easy case monitoring")
print("  ✅ Balanced loss weighting")
print("  ✅ L2 regularization")
print()

history = model.fit(
    [n_train, k_train, m_train, P_train],
    y_train,
    sample_weight=sample_weights_train,
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
# STEP 10: EVALUATE MODEL
# ============================================================================
print("="*70)
print("STEP 10: EVALUATING MODEL")
print("="*70)

model.load_weights('best_model_run7.h5')

y_pred_train = model.predict([n_train, k_train, m_train, P_train], verbose=0).flatten()
y_pred_val = model.predict([n_val, k_val, m_val, P_val], verbose=0).flatten()

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
# STEP 11: DETAILED PER-GROUP ANALYSIS
# ============================================================================
print("="*70)
print("DETAILED PER-GROUP COMPARISON")
print("="*70)

group_metrics = defaultdict(lambda: {'true': [], 'pred': []})

for i in range(len(y_val)):
    k = k_val[i, 0]
    m = m_val[i, 0]
    group_metrics[(k, m)]['true'].append(y_val[i])
    group_metrics[(k, m)]['pred'].append(y_pred_val[i])

# Save to run_7 folder
import os
os.makedirs('run_7', exist_ok=True)

# Reference values
run2_reference = {
    (4, 2): 0.103043, (4, 3): 0.107653, (4, 4): 0.209655, (4, 5): 1.422106,
    (5, 2): 0.096024, (5, 3): 0.227183, (5, 4): 0.957722,
    (6, 2): 0.355892, (6, 3): 1.083320
}

run6_reference = {
    (4, 2): 0.405360, (4, 3): 0.460462, (4, 4): 0.127370, (4, 5): 0.224046,
    (5, 2): 0.450569, (5, 3): 0.137842, (5, 4): 0.200213,
    (6, 2): 0.155255, (6, 3): 0.168260
}

with open('run_7/detailed_comparison.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("DETAILED PER-GROUP COMPARISON - RUN 7 vs RUN 2 vs RUN 6\n")
    f.write("="*70 + "\n\n")

    f.write("COMPARISON:\n")
    f.write("  Run 2: Balanced baseline (no aggressive weighting)\n")
    f.write("  Run 6: Data leakage + excessive augmentation (INVALID)\n")
    f.write("  Run 7: Fixed data leakage + balanced approach (VALID)\n\n")

    print("\nValidation Log2-MSE by (k,m) combination:")
    print("Run 2     | Run 6*    | Run 7     | vs R2      | vs R6      | n")
    print("-"*80)
    f.write("Validation Log2-MSE by (k,m) combination:\n")
    f.write("-"*70 + "\n")
    f.write(f"{'k,m':<8} {'Run 2':<10} {'Run 6*':<10} {'Run 7':<10} {'vs R2':<10} {'vs R6':<10} {'n':<6}\n")
    f.write("-"*70 + "\n")

    easy_sum, medium_sum, hard_sum = 0, 0, 0
    easy_count, medium_count, hard_count = 0, 0, 0

    for (k, m), data in sorted(group_metrics.items()):
        true_vals = np.array(data['true'])
        pred_vals = np.array(data['pred'])
        group_log2_mse = compute_log2_mse(true_vals, pred_vals)

        run2_val = run2_reference.get((k, m), None)
        run6_val = run6_reference.get((k, m), None)

        if run2_val:
            change_r2 = ((run2_val - group_log2_mse) / run2_val) * 100
            change_r2_str = f"{change_r2:+.1f}%"
        else:
            change_r2_str = "N/A"

        if run6_val:
            change_r6 = ((run6_val - group_log2_mse) / run6_val) * 100
            change_r6_str = f"{change_r6:+.1f}%"
        else:
            change_r6_str = "N/A"

        run2_str = f"{run2_val:.6f}" if run2_val else "N/A"
        run6_str = f"{run6_val:.6f}" if run6_val else "N/A"

        output_line = f"k={k},m={m}  {run2_str:<10} {run6_str:<10} {group_log2_mse:.6f}  {change_r2_str:<10} {change_r6_str:<10} n={len(true_vals)}"
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
    print("* Run 6 had data leakage - metrics are invalid")
    print()
    f.write("\n* Run 6 had data leakage - metrics are invalid\n\n")
    f.write("="*70 + "\n")
    f.write("SUMMARY BY DIFFICULTY TIER\n")
    f.write("="*70 + "\n")

    if easy_count > 0:
        easy_avg = easy_sum / easy_count
        f.write(f"Easy cases average: {easy_avg:.6f} (Target: < 0.15, Run 2: ~0.10)\n")
        print(f"Easy cases average: {easy_avg:.6f} (Target: < 0.15, Run 2: ~0.10)")

    if medium_count > 0:
        medium_avg = medium_sum / medium_count
        f.write(f"Medium cases average: {medium_avg:.6f} (Target: 0.20-0.30, Run 2: ~0.26)\n")
        print(f"Medium cases average: {medium_avg:.6f} (Target: 0.20-0.30, Run 2: ~0.26)")

    if hard_count > 0:
        hard_avg = hard_sum / hard_count
        f.write(f"Hard cases average: {hard_avg:.6f} (Target: 0.70-1.20, Run 2: ~1.14)\n")
        print(f"Hard cases average: {hard_avg:.6f} (Target: 0.70-1.20, Run 2: ~1.14)")

    # Balanced metric (equal weight to each tier)
    balanced_metric = (easy_avg + medium_avg + hard_avg) / 3
    f.write(f"\nBalanced metric (equal tier weight): {balanced_metric:.6f}\n")
    f.write(f"Overall average (sample-weighted): {val_log2_mse:.6f} (Target: < 0.374)\n")
    print(f"\nBalanced metric (equal tier weight): {balanced_metric:.6f}")
    print(f"Overall average (sample-weighted): {val_log2_mse:.6f} (Target: < 0.374)")

print()

# ============================================================================
# STEP 12: GENERATE PLOTS
# ============================================================================
print("STEP 12: Generating Plots")
print("-"*70)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Log2-MSE Loss', fontsize=12)
plt.title('Training History - Run 7', fontsize=14, fontweight='bold')
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
plt.savefig('run_7/training_history_run7.png', dpi=150, bbox_inches='tight')
print("Saved: run_7/training_history_run7.png")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_val, alpha=0.3, s=10)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True m-Height', fontsize=12)
plt.ylabel('Predicted m-Height', fontsize=12)
plt.title('Predictions vs True Values - Run 7 (Validation)', fontsize=14, fontweight='bold')
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
plt.savefig('run_7/predictions_scatter_run7.png', dpi=150, bbox_inches='tight')
print("Saved: run_7/predictions_scatter_run7.png")

plt.close('all')
print()

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================
print("="*70)
print("FINAL RESULTS SUMMARY - RUN 7")
print("="*70)
print(f"Training samples: {len(y_train):,} (augmented)")
print(f"Validation samples: {len(y_val):,} (original only - NO augmentation)")
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

# Tier summary
print("="*70)
print("TIER PERFORMANCE SUMMARY")
print("="*70)
print(f"Easy cases:   {easy_avg:.6f} (Target: < 0.15, Run 2 baseline: ~0.10)")
print(f"Medium cases: {medium_avg:.6f} (Target: 0.20-0.30, Run 2 baseline: ~0.26)")
print(f"Hard cases:   {hard_avg:.6f} (Target: 0.70-1.20, Run 2 baseline: ~1.14)")
print(f"Balanced:     {balanced_metric:.6f} (Equal weight to each tier)")
print()

# Comparison to Run 6
print("="*70)
print("RUN 7 vs RUN 6 COMPARISON")
print("="*70)
print("Run 6 Issues (INVALID):")
print("  ❌ Data leakage (augmentation before split)")
print("  ❌ 26x augmentation (excessive)")
print("  ❌ 2% noise (too small)")
print("  ❌ Easy cases degraded 3-4x")
print()
print("Run 7 Fixes (VALID):")
print("  ✅ Augmentation AFTER split (no leakage)")
print(f"  ✅ {len(y_train) / len(inputs_train_orig):.1f}x augmentation (reasonable)")
print("  ✅ 5% noise (better generalization)")
print("  ✅ Easy case monitoring")
print("  ✅ Balanced loss weighting")
print("  ✅ L2 regularization")
print()

print("="*70)
print("DELIVERABLES SAVED TO run_7/")
print("="*70)
print("  1. best_model_run7.h5 - Trained model weights")
print("  2. run_7/training_history_run7.png - Loss curves")
print("  3. run_7/predictions_scatter_run7.png - Prediction quality plots")
print("  4. run_7/detailed_comparison.txt - Complete comparison")
print("="*70)
print()
print("Training complete! 🎉")
print()
print("Next steps:")
print("  1. Review per-group metrics (especially easy cases)")
print("  2. Compare train-val gap (should be 0.02-0.08)")
print("  3. If results are good, use this model for final predictions")
print("="*70)
