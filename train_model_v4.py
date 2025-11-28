"""
Height Prediction Model Training Script - Run 4
Objective: Beat validation log2-MSE of 0.374 with improved regularization

Run 4 Improvements (Based on Run 2 Analysis):
1. Stronger regularization (higher dropout: 0.4/0.35/0.3, weight decay: 1e-3)
2. More aggressive early stopping (patience=25, since val loss plateaus ~100-120 epochs)
3. Enhanced weighting for problematic groups (k=4,m=5, k=6,m=3, k=5,m=4)
4. Adjusted curriculum learning (focus on hard groups earlier)
5. Better learning rate scheduling with more aggressive decay
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Flatten, Concatenate,
    Dropout, BatchNormalization, Add, Reshape, MultiHeadAttention, Lambda, Layer, Multiply
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
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
print("HEIGHT PREDICTION MODEL - RUN 4 (ENHANCED REGULARIZATION)")
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
# STEP 4: DEFINE COMPLEXITY GROUPS (BASED ON RUN 2 OBSERVATIONS)
# ============================================================================
print("STEP 4: Defining Complexity Groups (Run 2 Analysis)")
print("-"*70)

# Based on Run 2 per-group performance:
# Best: k=5,m=2 (0.096), k=4,m=2 (0.103), k=4,m=3 (0.108)
# Good: k=4,m=4 (0.210), k=5,m=3 (0.227), k=6,m=2 (0.356)
# Poor: k=5,m=4 (0.958), k=6,m=3 (1.083), k=4,m=5 (1.422)

EASY_GROUPS = [(4, 2), (4, 3), (5, 2)]
MEDIUM_GROUPS = [(4, 4), (5, 3), (6, 2)]
HARD_GROUPS = [(5, 4), (6, 3), (4, 5)]

def get_complexity_tier(k, m):
    """Returns 0 (easy), 1 (medium), or 2 (hard)"""
    if (k, m) in EASY_GROUPS:
        return 0
    elif (k, m) in MEDIUM_GROUPS:
        return 1
    elif (k, m) in HARD_GROUPS:
        return 2
    else:
        return 1  # default to medium

# Compute complexity tiers for training data
complexity_train = np.array([
    get_complexity_tier(k[0], m[0])
    for k, m in zip(k_train, m_train)
], dtype=np.int32)

print(f"Easy groups (Run 2: 0.096-0.108): {EASY_GROUPS}")
print(f"Medium groups (Run 2: 0.210-0.356): {MEDIUM_GROUPS}")
print(f"Hard groups (Run 2: 0.958-1.422): {HARD_GROUPS}")
print(f"\nTraining distribution:")
print(f"  Easy: {np.sum(complexity_train == 0)} samples")
print(f"  Medium: {np.sum(complexity_train == 1)} samples")
print(f"  Hard: {np.sum(complexity_train == 2)} samples")
print()

# ============================================================================
# STEP 5: BUILD MIXTURE-OF-EXPERTS MODEL (STRONGER REGULARIZATION)
# ============================================================================
print("STEP 5: Building MoE Model with Enhanced Regularization")
print("-"*70)

class ComplexityRouter(Layer):
    """Routes input to different expert pathways based on k,m complexity"""

    def __init__(self, **kwargs):
        super(ComplexityRouter, self).__init__(**kwargs)

    def build(self, input_shape):
        # Learnable gating mechanism
        self.gate = Dense(3, activation='softmax', name='complexity_gate')
        super().build(input_shape)

    def call(self, inputs):
        """Returns gating weights for 3 experts [easy, medium, hard]"""
        return self.gate(inputs)

def build_expert_pathway(input_tensor, input_dim, name_prefix, num_heads, key_dim, dense_units, dropout_rate):
    """Build an expert pathway with specified capacity and stronger regularization"""
    # Multi-head attention
    x_reshaped = Reshape((1, input_dim))(input_tensor)
    x_attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=0.15,  # Increased from 0.1
        name=f'{name_prefix}_attention'
    )(x_reshaped, x_reshaped)
    x_attn = Flatten()(x_attn)

    # Dense processing with higher dropout
    x = Dense(dense_units, activation='gelu', name=f'{name_prefix}_dense1')(x_attn)
    x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = Dropout(dropout_rate)(x)  # Increased dropout

    x = Dense(dense_units, activation='gelu', name=f'{name_prefix}_dense2')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn2')(x)

    return x

def build_moe_model(p_shape, k_vocab_size=7, m_vocab_size=6):
    """
    Mixture-of-Experts model with enhanced regularization to combat overfitting
    """
    # Inputs
    n_input = Input(shape=(1,), name='n_input')
    k_input = Input(shape=(1,), name='k_input', dtype=tf.int32)
    m_input = Input(shape=(1,), name='m_input', dtype=tf.int32)
    P_input = Input(shape=(p_shape,), name='P_input')

    # Embeddings
    k_embed = Flatten()(Embedding(k_vocab_size, 64, name='k_embedding')(k_input))
    m_embed = Flatten()(Embedding(m_vocab_size, 64, name='m_embedding')(m_input))

    # Complexity routing features
    routing_features = Concatenate()([k_embed, m_embed])
    router = ComplexityRouter(name='router')
    gate_weights = router(routing_features)  # [batch, 3]

    # P matrix initial processing with higher dropout
    P_processed = Dense(512, activation='gelu', name='P_initial')(P_input)
    P_processed = BatchNormalization(name='P_bn')(P_processed)
    P_processed = Dropout(0.25)(P_processed)  # Increased from 0.2

    # Expert pathways with different capacities and higher dropout
    expert_easy_raw = build_expert_pathway(
        P_processed, 512, 'expert_easy',
        num_heads=4, key_dim=64, dense_units=256, dropout_rate=0.35
    )

    expert_medium_raw = build_expert_pathway(
        P_processed, 512, 'expert_medium',
        num_heads=8, key_dim=64, dense_units=512, dropout_rate=0.4
    )

    expert_hard_raw = build_expert_pathway(
        P_processed, 512, 'expert_hard',
        num_heads=16, key_dim=128, dense_units=768, dropout_rate=0.4
    )

    # Project all experts to common dimension (512)
    expert_easy = Dense(512, activation='gelu', name='expert_easy_proj')(expert_easy_raw)
    expert_medium = Dense(512, activation='gelu', name='expert_medium_proj')(expert_medium_raw)
    expert_hard = Dense(512, activation='gelu', name='expert_hard_proj')(expert_hard_raw)

    # Weighted combination of experts
    gate_0 = Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(gate_weights)
    gate_1 = Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(gate_weights)
    gate_2 = Lambda(lambda x: tf.expand_dims(x[:, 2], axis=-1))(gate_weights)

    weighted_easy = Multiply()([expert_easy, gate_0])
    weighted_medium = Multiply()([expert_medium, gate_1])
    weighted_hard = Multiply()([expert_hard, gate_2])

    expert_output = Add()([weighted_easy, weighted_medium, weighted_hard])

    # Combine all features
    combined = Concatenate()([n_input, k_embed, m_embed, expert_output])

    # Shared deep network with residual connections and higher dropout
    x = Dense(1024, activation='gelu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.45)(x)  # Increased from 0.35

    # 3 residual blocks with increased dropout
    for i in range(3):
        residual = x
        x = Dense(1024, activation='gelu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)  # Increased from 0.3
        x = Dense(1024, activation='gelu')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])

    # Final layers with higher dropout
    x = Dense(512, activation='gelu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)  # Increased from 0.2
    x = Dense(256, activation='gelu')(x)
    x = Dropout(0.2)(x)  # Added dropout

    # Output layer: predict log2(m-height), then convert to m-height
    log2_pred = Dense(1, activation='linear', name='log2_prediction')(x)
    log2_positive = Lambda(lambda x: tf.nn.softplus(x), name='softplus')(log2_pred)
    output = Lambda(lambda x: tf.pow(2.0, x), name='m_height')(log2_positive)

    model = Model(
        inputs=[n_input, k_input, m_input, P_input],
        outputs=output,
        name='moe_height_predictor_v4'
    )

    return model

p_shape = P_train.shape[1]
model = build_moe_model(p_shape, k_vocab_size=k_values.max()+1, m_vocab_size=m_values.max()+1)

print(f"Model built with enhanced regularization!")
print(f"Total parameters: {model.count_params():,}")
print(f"Regularization improvements:")
print(f"  - P matrix dropout: 0.20 → 0.25")
print(f"  - Expert dropout: 0.30 → 0.35-0.40")
print(f"  - Main network dropout: 0.35 → 0.45")
print(f"  - Residual dropout: 0.30 → 0.40")
print(f"  - Final layer dropout: 0.20 → 0.30")
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
# STEP 7: ENHANCED CURRICULUM LEARNING (MORE AGGRESSIVE)
# ============================================================================
print("STEP 7: Setting Up Enhanced Curriculum Learning")
print("-"*70)

class EnhancedCurriculumLearning(Callback):
    """
    More aggressive curriculum learning based on Run 2 analysis:
    - Epochs 0-30: Focus on easy groups (weight 1.0 for easy, 0.4 for medium, 0.2 for hard)
    - Epochs 31-60: Include medium groups (weight 1.0 for easy, 1.0 for medium, 0.5 for hard)
    - Epochs 61+: All groups with VERY aggressive weighting on k=4,m=5, k=6,m=3, k=5,m=4
    """

    def __init__(self, k_train, m_train):
        super().__init__()
        self.k_train = k_train.flatten()
        self.m_train = m_train.flatten()
        self.sample_weights = np.ones(len(k_train), dtype=np.float32)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 30:
            # Phase 1: Easy groups (shorter phase)
            for i, (k, m) in enumerate(zip(self.k_train, self.m_train)):
                if (k, m) in EASY_GROUPS:
                    self.sample_weights[i] = 1.0
                elif (k, m) in MEDIUM_GROUPS:
                    self.sample_weights[i] = 0.4
                else:  # hard
                    self.sample_weights[i] = 0.2

        elif epoch < 60:
            # Phase 2: Easy + Medium groups
            for i, (k, m) in enumerate(zip(self.k_train, self.m_train)):
                if (k, m) in EASY_GROUPS:
                    self.sample_weights[i] = 1.0
                elif (k, m) in MEDIUM_GROUPS:
                    self.sample_weights[i] = 1.0
                else:  # hard
                    self.sample_weights[i] = 0.5

        else:
            # Phase 3: ALL groups with VERY aggressive weighting on worst performers
            # Based on Run 2: k=4,m=5 (1.422), k=6,m=3 (1.083), k=5,m=4 (0.958)
            hard_weight = min(5.0, 2.0 + (epoch - 60) * 0.08)  # More aggressive growth
            for i, (k, m) in enumerate(zip(self.k_train, self.m_train)):
                if (k, m) in EASY_GROUPS:
                    self.sample_weights[i] = 1.0
                elif (k, m) in MEDIUM_GROUPS:
                    self.sample_weights[i] = 1.5
                elif (k, m) == (4, 5):  # WORST performer (1.422)
                    self.sample_weights[i] = hard_weight * 1.3
                elif (k, m) == (6, 3):  # Second worst (1.083)
                    self.sample_weights[i] = hard_weight * 1.1
                elif (k, m) == (5, 4):  # Third worst (0.958)
                    self.sample_weights[i] = hard_weight
                else:
                    self.sample_weights[i] = hard_weight * 0.7

        # Update model's sample weights
        self.model.sample_weight = self.sample_weights

curriculum_callback = EnhancedCurriculumLearning(k_train, m_train)

print("Enhanced curriculum learning phases:")
print("  Phase 1 (epochs 0-30): Focus on easy groups")
print("  Phase 2 (epochs 31-60): Add medium groups")
print("  Phase 3 (epochs 61+): AGGRESSIVE weighting on hard groups")
print("    - k=4,m=5: weight up to 6.5x (worst: 1.422)")
print("    - k=6,m=3: weight up to 5.5x (2nd worst: 1.083)")
print("    - k=5,m=4: weight up to 5.0x (3rd worst: 0.958)")
print()

# ============================================================================
# STEP 8: COMPILE MODEL WITH HIGHER WEIGHT DECAY
# ============================================================================
print("STEP 8: Compiling Model")
print("-"*70)

# Run 4: Higher weight decay to combat overfitting
optimizer = AdamW(learning_rate=7e-4, weight_decay=1e-3)  # Increased from 5e-4

model.compile(
    optimizer=optimizer,
    loss=log2_mse_loss,
    metrics=[log2_mse_loss]
)

print("Model compiled with AdamW optimizer")
print(f"  Learning rate: 7e-4")
print(f"  Weight decay: 1e-3 (INCREASED from 5e-4)")
print()

# ============================================================================
# STEP 9: SETUP CALLBACKS (MORE AGGRESSIVE EARLY STOPPING)
# ============================================================================
print("STEP 9: Setting Up Training Callbacks")
print("-"*70)

class CosineAnnealingWarmRestarts(Callback):
    """Cosine annealing with warm restarts (shorter periods)"""

    def __init__(self, initial_lr=7e-4, min_lr=1e-6, restart_period=30):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.restart_period = restart_period

    def on_epoch_begin(self, epoch, logs=None):
        # Compute current position in cycle
        cycle_position = epoch % self.restart_period
        # Cosine annealing
        lr = self.min_lr + (self.initial_lr - self.min_lr) * \
             (1 + np.cos(np.pi * cycle_position / self.restart_period)) / 2
        self.model.optimizer.learning_rate.assign(lr)

cosine_lr = CosineAnnealingWarmRestarts(initial_lr=7e-4, min_lr=1e-6, restart_period=30)

callbacks = [
    curriculum_callback,
    cosine_lr,
    EarlyStopping(
        monitor='val_loss',
        patience=25,  # REDUCED from 50 (Run 2 shows plateau ~100-120 epochs)
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,  # More aggressive LR reduction
        min_lr=1e-7,
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
print("  - Enhanced Curriculum Learning (3 phases, more aggressive)")
print("  - Cosine Annealing LR (warm restarts every 30 epochs)")
print("  - EarlyStopping (patience=25, REDUCED from 50)")
print("  - ReduceLROnPlateau (patience=10, factor=0.5)")
print("  - ModelCheckpoint")
print()

# ============================================================================
# STEP 10: TRAIN MODEL
# ============================================================================
print("="*70)
print("STEP 10: TRAINING MODEL")
print("="*70)
print(f"Batch size: 256")
print(f"Max epochs: 200 (reduced from 250)")
print(f"Strategy: Enhanced curriculum + stronger regularization")
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

model.load_weights('best_model.h5')

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
# STEP 12: PER-GROUP ANALYSIS
# ============================================================================
print("="*70)
print("PER-GROUP PERFORMANCE ANALYSIS")
print("="*70)

group_metrics = defaultdict(lambda: {'true': [], 'pred': []})

for i in range(len(y_val)):
    k = k_val[i, 0]
    m = m_val[i, 0]
    group_metrics[(k, m)]['true'].append(y_val[i])
    group_metrics[(k, m)]['pred'].append(y_pred_val[i])

# Save to run_4 folder
import os
os.makedirs('run_4', exist_ok=True)

with open('run_4/per_group_performance.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("PER-GROUP PERFORMANCE BREAKDOWN - RUN 4\n")
    f.write("="*70 + "\n\n")
    f.write("Run 2 baseline for comparison:\n")
    f.write("  Easy: k=5,m=2 (0.096), k=4,m=2 (0.103), k=4,m=3 (0.108)\n")
    f.write("  Medium: k=4,m=4 (0.210), k=5,m=3 (0.227), k=6,m=2 (0.356)\n")
    f.write("  Hard: k=5,m=4 (0.958), k=6,m=3 (1.083), k=4,m=5 (1.422)\n\n")

    print("\nValidation Log2-MSE by (k,m) combination:")
    print("(Run 2 baseline in parentheses)")
    print("-"*70)
    f.write("Validation Log2-MSE by (k,m) combination:\n")
    f.write("-"*70 + "\n")

    # Reference values from Run 2
    run2_reference = {
        (4, 2): 0.103043, (4, 3): 0.107653, (4, 4): 0.209655, (4, 5): 1.422106,
        (5, 2): 0.096024, (5, 3): 0.227183, (5, 4): 0.957722,
        (6, 2): 0.355892, (6, 3): 1.083320
    }

    for (k, m), data in sorted(group_metrics.items()):
        true_vals = np.array(data['true'])
        pred_vals = np.array(data['pred'])
        group_log2_mse = compute_log2_mse(true_vals, pred_vals)

        run2_val = run2_reference.get((k, m), None)
        if run2_val:
            improvement = ((run2_val - group_log2_mse) / run2_val) * 100
            output_line = f"  k={k}, m={m}: {group_log2_mse:.6f} (Run 2: {run2_val:.6f}, change: {improvement:+.1f}%) n={len(true_vals)}"
        else:
            output_line = f"  k={k}, m={m}: {group_log2_mse:.6f} (n={len(true_vals)} samples)"

        print(output_line)
        f.write(output_line + "\n")

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
plt.title('Training History - Run 4', fontsize=14, fontweight='bold')
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
plt.savefig('run_4/training_history.png', dpi=150, bbox_inches='tight')
print("Saved: run_4/training_history.png")

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
plt.savefig('run_4/predictions_scatter.png', dpi=150, bbox_inches='tight')
print("Saved: run_4/predictions_scatter.png")

plt.close('all')
print()

# ============================================================================
# FINAL RESULTS SUMMARY
# ============================================================================
print("="*70)
print("FINAL RESULTS SUMMARY - RUN 4")
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

# Compare to previous runs
print("="*70)
print("COMPARISON TO PREVIOUS RUNS")
print("="*70)
print("Run 1: Val log2-MSE = 0.495 (baseline)")
print("Run 2: Val log2-MSE = 0.507 (group-weighted loss)")
print(f"Run 4: Val log2-MSE = {val_log2_mse:.6f}")
if val_log2_mse < 0.495:
    run1_improvement = ((0.495 - val_log2_mse) / 0.495) * 100
    print(f"       Improvement over Run 1: {run1_improvement:.1f}%")
print()

print("="*70)
print("DELIVERABLES SAVED TO run_4/")
print("="*70)
print("  1. best_model.h5 - Trained model weights")
print("  2. run_4/training_history.png - Loss curves")
print("  3. run_4/predictions_scatter.png - Prediction quality plots")
print("  4. run_4/per_group_performance.txt - Detailed per-(k,m) metrics with Run 2 comparison")
print("="*70)
print()
print("Training complete! 🎉")
