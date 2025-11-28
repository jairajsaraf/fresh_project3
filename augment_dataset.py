"""
Advanced Data Augmentation Script for Height Prediction

Objectives:
1. Properly merge DS-1, DS-2, and DS-3 (fixing the corrupted combined dataset)
2. Apply sophisticated augmentation techniques to balance all (k,m) groups
3. Target: 12,000 samples per (k,m) combination for robust training

Augmentation Techniques:
- Gaussian noise injection to P matrices (preserves matrix structure)
- Controlled perturbations with magnitude scaling
- Interpolation between similar samples (SMOTE-like)
- Bootstrap sampling with variations
- Domain-aware transformations

This is CPU-intensive and separate from GPU training.
"""

import numpy as np
import pickle
from collections import defaultdict
import argparse

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("ADVANCED DATA AUGMENTATION FOR HEIGHT PREDICTION")
print("="*70)

# ============================================================================
# STEP 1: LOAD ALL SOURCE DATASETS
# ============================================================================
print("\nSTEP 1: Loading Source Datasets")
print("-"*70)

def load_dataset(inputs_file, outputs_file, name):
    """Load a dataset from pickle files"""
    print(f"Loading {name}...")
    with open(inputs_file, 'rb') as f:
        inputs = pickle.load(f)
    with open(outputs_file, 'rb') as f:
        outputs = pickle.load(f)
    print(f"  Loaded {len(inputs):,} samples")
    return inputs, outputs

# Load all three datasets
ds1_inputs, ds1_outputs = load_dataset(
    'DS-1-samples_n_k_m_P.pkl',
    'DS-1-samples_mHeights.pkl',
    'DS-1'
)

ds2_inputs, ds2_outputs = load_dataset(
    'DS-2-Train-n_k_m_P.pkl',
    'DS-2-Train-mHeights.pkl',
    'DS-2'
)

ds3_inputs, ds3_outputs = load_dataset(
    'DS-3-Train-n_k_m_P.pkl',
    'DS-3-Train-mHeights.pkl',
    'DS-3 (NEW!)'
)

# ============================================================================
# STEP 2: MERGE ALL DATASETS PROPERLY
# ============================================================================
print("\nSTEP 2: Merging All Datasets")
print("-"*70)

all_inputs = ds1_inputs + ds2_inputs + ds3_inputs
all_outputs = ds1_outputs + ds2_outputs + ds3_outputs

print(f"Total merged samples: {len(all_inputs):,}")
print(f"DS-1 contribution: {len(ds1_inputs):,} ({100*len(ds1_inputs)/len(all_inputs):.1f}%)")
print(f"DS-2 contribution: {len(ds2_inputs):,} ({100*len(ds2_inputs)/len(all_inputs):.1f}%)")
print(f"DS-3 contribution: {len(ds3_inputs):,} ({100*len(ds3_inputs)/len(all_inputs):.1f}%)")

# ============================================================================
# STEP 3: ANALYZE DISTRIBUTION
# ============================================================================
print("\nSTEP 3: Analyzing (k,m) Distribution")
print("-"*70)

# Group samples by (k,m)
groups = defaultdict(list)
for i, sample in enumerate(all_inputs):
    k = int(sample[1])
    m = int(sample[2])
    groups[(k, m)].append(i)

print(f"\nCurrent distribution:")
for (k, m), indices in sorted(groups.items()):
    count = len(indices)
    percentage = (count / len(all_inputs)) * 100
    print(f"  k={k}, m={m}: {count:6d} samples ({percentage:5.2f}%)")

# Calculate statistics
counts = [len(indices) for indices in groups.values()]
min_count = min(counts)
max_count = max(counts)
print(f"\nImbalance ratio: {max_count/min_count:.1f}x")

# ============================================================================
# STEP 4: DEFINE AUGMENTATION TECHNIQUES
# ============================================================================
print("\nSTEP 4: Defining Augmentation Techniques")
print("-"*70)

def add_gaussian_noise(P_matrix, noise_level=0.05):
    """Add Gaussian noise to P matrix"""
    noise = np.random.normal(0, noise_level, P_matrix.shape)
    return P_matrix + noise * np.std(P_matrix)

def perturb_matrix(P_matrix, perturbation_strength=0.03):
    """Apply controlled random perturbations"""
    perturbation = np.random.uniform(-perturbation_strength, perturbation_strength, P_matrix.shape)
    return P_matrix * (1 + perturbation)

def interpolate_samples(sample1, sample2, alpha=None):
    """
    Interpolate between two samples (SMOTE-like)
    Only interpolates P matrices (k, m, n must match)
    """
    if alpha is None:
        alpha = np.random.uniform(0.3, 0.7)  # Avoid extremes

    # Check if samples are compatible (same k, m, n)
    if (sample1[0] != sample2[0] or  # n
        sample1[1] != sample2[1] or  # k
        sample1[2] != sample2[2]):   # m
        return None

    # Interpolate P matrix
    P1 = sample1[3]
    P2 = sample2[3]
    P_new = alpha * P1 + (1 - alpha) * P2

    return [sample1[0], sample1[1], sample1[2], P_new]

def augment_sample(sample, output, technique='mixed'):
    """
    Augment a single sample using various techniques
    Returns: (augmented_sample, estimated_output)
    """
    n, k, m, P = sample[0], sample[1], sample[2], sample[3].copy()

    if technique == 'noise':
        P_aug = add_gaussian_noise(P, noise_level=0.05)
    elif technique == 'perturb':
        P_aug = perturb_matrix(P, perturbation_strength=0.03)
    elif technique == 'strong_noise':
        P_aug = add_gaussian_noise(P, noise_level=0.08)
    elif technique == 'strong_perturb':
        P_aug = perturb_matrix(P, perturbation_strength=0.05)
    elif technique == 'mixed':
        # Combine techniques
        P_aug = add_gaussian_noise(P, noise_level=0.04)
        P_aug = perturb_matrix(P_aug, perturbation_strength=0.02)
    else:
        P_aug = P

    # For augmented samples, we use the same output
    # (This is an approximation - the true output might differ slightly)
    return [n, k, m, P_aug], output

print("Augmentation techniques defined:")
print("  1. Gaussian noise injection (preserves statistical properties)")
print("  2. Controlled perturbations (domain-aware scaling)")
print("  3. Sample interpolation (SMOTE-like for same k,m,n groups)")
print("  4. Mixed techniques (combines multiple approaches)")

# ============================================================================
# STEP 5: DETERMINE AUGMENTATION STRATEGY
# ============================================================================
print("\nSTEP 5: Determining Augmentation Strategy")
print("-"*70)

TARGET_SAMPLES = 12000  # Target samples per (k,m) group
print(f"Target: {TARGET_SAMPLES:,} samples per (k,m) combination")

augmentation_plan = {}
for (k, m), indices in sorted(groups.items()):
    current_count = len(indices)
    needed = max(0, TARGET_SAMPLES - current_count)
    augmentation_plan[(k, m)] = {
        'current': current_count,
        'needed': needed,
        'target': TARGET_SAMPLES
    }

    if needed > 0:
        print(f"  k={k}, m={m}: Need {needed:,} augmented samples (have {current_count:,})")
    else:
        # Subsample to target
        print(f"  k={k}, m={m}: Will subsample to {TARGET_SAMPLES:,} (have {current_count:,})")

# ============================================================================
# STEP 6: PERFORM AUGMENTATION
# ============================================================================
print("\nSTEP 6: Performing Data Augmentation")
print("-"*70)

augmented_inputs = []
augmented_outputs = []

print("Processing groups...")
total_groups = len(augmentation_plan)
for i, ((k, m), plan) in enumerate(sorted(augmentation_plan.items())):
    print(f"  Processing group {i+1}/{total_groups}: k={k}, m={m}...", end=' ')
    indices = groups[(k, m)]
    current_count = plan['current']
    needed = plan['needed']
    target = plan['target']

    if current_count >= target:
        # Subsample to target
        selected_indices = np.random.choice(indices, target, replace=False)
        for idx in selected_indices:
            augmented_inputs.append(all_inputs[idx])
            augmented_outputs.append(all_outputs[idx])
    else:
        # First, add all original samples
        for idx in indices:
            augmented_inputs.append(all_inputs[idx])
            augmented_outputs.append(all_outputs[idx])

        # Then, generate augmented samples
        techniques = ['noise', 'perturb', 'strong_noise', 'strong_perturb', 'mixed']

        for _ in range(needed):
            # Randomly select a sample from this group
            idx = np.random.choice(indices)
            sample = all_inputs[idx]
            output = all_outputs[idx]

            # Choose augmentation technique
            technique = np.random.choice(techniques)

            # Special case: Try interpolation 30% of the time if we have enough samples
            if len(indices) > 1 and np.random.random() < 0.3:
                idx2 = np.random.choice(indices)
                sample2 = all_inputs[idx2]
                output2 = all_outputs[idx2]

                # Interpolate samples
                alpha = np.random.uniform(0.3, 0.7)
                interpolated_sample = interpolate_samples(sample, sample2, alpha)

                if interpolated_sample is not None:
                    # Interpolate output as well
                    # Use geometric mean for m-height (log-space interpolation)
                    log_output = alpha * np.log(output + 1e-7) + (1 - alpha) * np.log(output2 + 1e-7)
                    interpolated_output = np.exp(log_output)

                    augmented_inputs.append(interpolated_sample)
                    augmented_outputs.append(interpolated_output)
                    continue

            # Apply augmentation
            aug_sample, aug_output = augment_sample(sample, output, technique)
            augmented_inputs.append(aug_sample)
            augmented_outputs.append(aug_output)

    print("Done")

print(f"\nAugmentation complete!")
print(f"Total augmented samples: {len(augmented_inputs):,}")

# ============================================================================
# STEP 7: VERIFY BALANCED DISTRIBUTION
# ============================================================================
print("\nSTEP 7: Verifying Balanced Distribution")
print("-"*70)

# Verify distribution
final_groups = defaultdict(int)
for sample in augmented_inputs:
    k = int(sample[1])
    m = int(sample[2])
    final_groups[(k, m)] += 1

print(f"\nFinal distribution:")
for (k, m), count in sorted(final_groups.items()):
    percentage = (count / len(augmented_inputs)) * 100
    print(f"  k={k}, m={m}: {count:6d} samples ({percentage:5.2f}%)")

# ============================================================================
# STEP 8: SAVE AUGMENTED DATASET
# ============================================================================
print("\nSTEP 8: Saving Augmented Dataset")
print("-"*70)

output_inputs_file = 'augmented_n_k_m_P.pkl'
output_outputs_file = 'augmented_mHeights.pkl'

print(f"Saving to {output_inputs_file}...")
with open(output_inputs_file, 'wb') as f:
    pickle.dump(augmented_inputs, f)

print(f"Saving to {output_outputs_file}...")
with open(output_outputs_file, 'wb') as f:
    pickle.dump(augmented_outputs, f)

print(f"\n✓ Saved {len(augmented_inputs):,} samples")

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("AUGMENTATION SUMMARY")
print("="*70)

print(f"\nOriginal (DS-1 + DS-2 + DS-3): {len(all_inputs):,} samples")
print(f"Augmented dataset: {len(augmented_inputs):,} samples")
print(f"Increase: {len(augmented_inputs) - len(all_inputs):,} samples")

print(f"\nBalance:")
counts = list(final_groups.values())
min_count = min(counts)
max_count = max(counts)
print(f"  Min samples per (k,m): {min_count:,}")
print(f"  Max samples per (k,m): {max_count:,}")
print(f"  Imbalance ratio: {max_count/min_count:.2f}x")

print(f"\nTarget achieved: {min_count >= TARGET_SAMPLES * 0.95}")
print("\n✓ Data augmentation complete!")
print("="*70)
