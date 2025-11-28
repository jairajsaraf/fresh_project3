"""
Height Prediction Model Training Script - Run 3
Objective: Beat validation log2-MSE of 0.374 using advanced techniques

Run 3 Improvements:
1. Mixture-of-Experts architecture with complexity-aware routing
2. Curriculum learning (train on easy groups first)
3. Adaptive group weighting (gradually increase focus on hard groups)
4. Better regularization balance
5. Enhanced multi-head attention with more heads
6. Group-specific batch normalization
7. Cosine annealing with warm restarts
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
print("HEIGHT PREDICTION MODEL - RUN 3 (ADVANCED)")
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

# Define 3 complexity tiers
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

print(f"Easy groups: {EASY_GROUPS}")
print(f"Medium groups: {MEDIUM_GROUPS}")
print(f"Hard groups: {HARD_GROUPS}")
print(f"\nTraining distribution:")
print(f"  Easy: {np.sum(complexity_train == 0)} samples")
print(f"  Medium: {np.sum(complexity_train == 1)} samples")
print(f"  Hard: {np.sum(complexity_train == 2)} samples")
print()

# ============================================================================
# STEP 5: BUILD MIXTURE-OF-EXPERTS MODEL
# ============================================================================
print("STEP 5: Building Mixture-of-Experts Model")
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

def build_expert_pathway(input_tensor, input_dim, name_prefix, num_heads, key_dim, dense_units):
    """Build an expert pathway with specified capacity"""
    # Multi-head attention
    x_reshaped = Reshape((1, input_dim))(input_tensor)
    x_attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=0.1,
        name=f'{name_prefix}_attention'
    )(x_reshaped, x_reshaped)
    x_attn = Flatten()(x_attn)

    # Dense processing
    x = Dense(dense_units, activation='gelu', name=f'{name_prefix}_dense1')(x_attn)
    x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = Dropout(0.3)(x)

    x = Dense(dense_units, activation='gelu', name=f'{name_prefix}_dense2')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn2')(x)

    return x

def build_moe_model(p_shape, k_vocab_size=7, m_vocab_size=6):
    """
    Mixture-of-Experts model with:
    - 3 expert pathways (easy, medium, hard complexity)
    - Complexity-aware routing
    - Enhanced attention for hard groups
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

    # P matrix initial processing
    P_processed = Dense(512, activation='gelu', name='P_initial')(P_input)
    P_processed = BatchNormalization(name='P_bn')(P_processed)
    P_processed = Dropout(0.2)(P_processed)

    # Expert pathways with different capacities
    expert_easy_raw = build_expert_pathway(
        P_processed, 512, 'expert_easy',
        num_heads=4, key_dim=64, dense_units=256
    )

    expert_medium_raw = build_expert_pathway(
        P_processed, 512, 'expert_medium',
        num_heads=8, key_dim=64, dense_units=512
    )

    expert_hard_raw = build_expert_pathway(
        P_processed, 512, 'expert_hard',
        num_heads=16, key_dim=128, dense_units=768
    )

    # Project all experts to common dimension (512)
    expert_easy = Dense(512, activation='gelu', name='expert_easy_proj')(expert_easy_raw)
    expert_medium = Dense(512, activation='gelu', name='expert_medium_proj')(expert_medium_raw)
    expert_hard = Dense(512, activation='gelu', name='expert_hard_proj')(expert_hard_raw)

    # Weighted combination of experts (using gate weights)
    # expert_output = gate[0] * expert_easy + gate[1] * expert_medium + gate[2] * expert_hard
    gate_0 = Lambda(lambda x: tf.expand_dims(x[:, 0], axis=-1))(gate_weights)
    gate_1 = Lambda(lambda x: tf.expand_dims(x[:, 1], axis=-1))(gate_weights)
    gate_2 = Lambda(lambda x: tf.expand_dims(x[:, 2], axis=-1))(gate_weights)

    weighted_easy = Multiply()([expert_easy, gate_0])
    weighted_medium = Multiply()([expert_medium, gate_1])
    weighted_hard = Multiply()([expert_hard, gate_2])

    expert_output = Add()([weighted_easy, weighted_medium, weighted_hard])

    # Combine all features
    combined = Concatenate()([n_input, k_embed, m_embed, expert_output])

    # Shared deep network with residual connections
    x = Dense(1024, activation='gelu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.35)(x)

    # 3 residual blocks (reduced from 4 for better regularization)
    for i in range(3):
        residual = x
        x = Dense(1024, activation='gelu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(1024, activation='gelu')(x)
        x = BatchNormalization()(x)
        x = Add()([x, residual])

    # Final layers
    x = Dense(512, activation='gelu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='gelu')(x)

    # Output layer: predict log2(m-height), then convert to m-height
    log2_pred = Dense(1, activation='linear', name='log2_prediction')(x)
    log2_positive = Lambda(lambda x: tf.nn.softplus(x), name='softplus')(log2_pred)
    output = Lambda(lambda x: tf.pow(2.0, x), name='m_height')(log2_positive)

    model = Model(
        inputs=[n_input, k_input, m_input, P_input],
        outputs=output,
        name='moe_height_predictor'
    )

    return model

p_shape = P_train.shape[1]
model = build_moe_model(p_shape, k_vocab_size=k_values.max()+1, m_vocab_size=m_values.max()+1)

print(f"Model built successfully!")
print(f"Total parameters: {model.count_params():,}")
model.summary()
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
# STEP 7: CURRICULUM LEARNING CALLBACK
# ============================================================================
print("STEP 7: Setting Up Curriculum Learning")
print("-"*70)

class CurriculumLearning(Callback):
    """
    Implements curriculum learning:
    - Epochs 0-40: Focus on easy groups (weight 1.0 for easy, 0.3 for medium, 0.1 for hard)
    - Epochs 41-80: Include medium groups (weight 1.0 for easy, 1.0 for medium, 0.3 for hard)
    - Epochs 81+: All groups equally (adaptive weights based on performance)
    """

    def __init__(self, k_train, m_train):
        super().__init__()
        self.k_train = k_train.flatten()
        self.m_train = m_train.flatten()
        self.sample_weights = np.ones(len(k_train), dtype=np.float32)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < 40:
            # Phase 1: Easy groups
            for i, (k, m) in enumerate(zip(self.k_train, self.m_train)):
                if (k, m) in EASY_GROUPS:
                    self.sample_weights[i] = 1.0
                elif (k, m) in MEDIUM_GROUPS:
                    self.sample_weights[i] = 0.3
                else:  # hard
                    self.sample_weights[i] = 0.1

        elif epoch < 80:
            # Phase 2: Easy + Medium groups
            for i, (k, m) in enumerate(zip(self.k_train, self.m_train)):
                if (k, m) in EASY_GROUPS:
                    self.sample_weights[i] = 1.0
                elif (k, m) in MEDIUM_GROUPS:
                    self.sample_weights[i] = 1.0
                else:  # hard
                    self.sample_weights[i] = 0.3

        else:
            # Phase 3: All groups with adaptive weighting
            # Gradually increase weight for hard groups
            hard_weight = min(3.0, 1.0 + (epoch - 80) * 0.05)
            for i, (k, m) in enumerate(zip(self.k_train, self.m_train)):
                if (k, m) in EASY_GROUPS:
                    self.sample_weights[i] = 1.0
                elif (k, m) in MEDIUM_GROUPS:
                    self.sample_weights[i] = 1.5
                elif (k, m) == (4, 5):  # worst performer
                    self.sample_weights[i] = hard_weight
                else:  # other hard groups
                    self.sample_weights[i] = hard_weight * 0.8

        # Update model's sample weights
        self.model.sample_weight = self.sample_weights

curriculum_callback = CurriculumLearning(k_train, m_train)

print("Curriculum learning phases:")
print("  Phase 1 (epochs 0-40): Focus on easy groups")
print("  Phase 2 (epochs 41-80): Add medium groups")
print("  Phase 3 (epochs 81+): All groups with adaptive weighting")
print()

# ============================================================================
# STEP 8: COMPILE MODEL
# ============================================================================
print("STEP 8: Compiling Model")
print("-"*70)

# Run 3: balanced approach - learning rate 7e-4, weight decay 5e-4
optimizer = AdamW(learning_rate=7e-4, weight_decay=5e-4)

model.compile(
    optimizer=optimizer,
    loss=log2_mse_loss,
    metrics=[log2_mse_loss]
)

print("Model compiled with AdamW optimizer")
print(f"  Learning rate: 7e-4")
print(f"  Weight decay: 5e-4")
print()

# ============================================================================
# STEP 9: SETUP CALLBACKS
# ============================================================================
print("STEP 9: Setting Up Training Callbacks")
print("-"*70)

class CosineAnnealingWarmRestarts(Callback):
    """Cosine annealing with warm restarts"""

    def __init__(self, initial_lr=7e-4, min_lr=1e-6, restart_period=40):
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

cosine_lr = CosineAnnealingWarmRestarts(initial_lr=7e-4, min_lr=1e-6, restart_period=40)

callbacks = [
    curriculum_callback,
    cosine_lr,
    EarlyStopping(
        monitor='val_loss',
        patience=50,  # Increased from 30 for curriculum learning
        restore_best_weights=True,
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
print("  - Curriculum Learning (3 phases)")
print("  - Cosine Annealing LR (warm restarts every 40 epochs)")
print("  - EarlyStopping (patience=50)")
print("  - ModelCheckpoint")
print()

# ============================================================================
# STEP 10: TRAIN MODEL
# ============================================================================
print("="*70)
print("STEP 10: TRAINING MODEL")
print("="*70)
print(f"Batch size: 256")
print(f"Max epochs: 250")
print(f"Strategy: Curriculum learning with MoE architecture")
print()

history = model.fit(
    [n_train, k_train, m_train, P_train],
    y_train,
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

with open('per_group_performance.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("PER-GROUP PERFORMANCE BREAKDOWN - RUN 3\n")
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

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Log2-MSE Loss', fontsize=12)
plt.title('Training History - Run 3', fontsize=14, fontweight='bold')
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
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("Saved: training_history.png")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_val, y_pred_val, alpha=0.3, s=10)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('True m-Height', fontsize=12)
plt.ylabel('Predicted m-Height', fontsize=12)
plt.title('Predictions vs True Values', fontsize=14, fontweight='bold')
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
print("FINAL RESULTS SUMMARY - RUN 3")
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

# Compare to previous runs
print("="*70)
print("COMPARISON TO PREVIOUS RUNS")
print("="*70)
print("Run 1: Val log2-MSE = 0.495 (baseline)")
print("Run 2: Val log2-MSE = 0.507 (worse than baseline!)")
print(f"Run 3: Val log2-MSE = {val_log2_mse:.6f}")
if val_log2_mse < 0.495:
    run1_improvement = ((0.495 - val_log2_mse) / 0.495) * 100
    print(f"       Improvement over Run 1: {run1_improvement:.1f}%")
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
