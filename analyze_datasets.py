"""
Dataset Analysis Script
Analyzes DS-1, DS-2, DS-3 and combined dataset to understand:
- Sample counts
- (k,m) distribution
- Value ranges
- Data quality
- Augmentation opportunities
"""

import numpy as np
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt

def load_dataset(inputs_file, outputs_file):
    """Load a dataset from pickle files"""
    with open(inputs_file, 'rb') as f:
        inputs = pickle.load(f)
    with open(outputs_file, 'rb') as f:
        outputs = pickle.load(f)
    return inputs, outputs

def analyze_dataset(inputs, outputs, name):
    """Analyze a single dataset"""
    print(f"\n{'='*70}")
    print(f"ANALYZING {name}")
    print(f"{'='*70}")

    print(f"Total samples: {len(inputs)}")
    print(f"Output samples: {len(outputs)}")

    # Analyze structure
    if len(inputs) > 0:
        sample = inputs[0]
        if isinstance(sample, list):
            print(f"Sample structure: [n={sample[0]}, k={sample[1]}, m={sample[2]}, P_matrix shape={sample[3].shape}]")
        else:
            print(f"Sample type: {type(sample)}")

    # Analyze outputs
    outputs_array = np.array(outputs)
    print(f"\nOutput (m-height) statistics:")
    print(f"  Min: {outputs_array.min():.2f}")
    print(f"  Max: {outputs_array.max():.2f}")
    print(f"  Mean: {outputs_array.mean():.2f}")
    print(f"  Median: {np.median(outputs_array):.2f}")
    print(f"  Std: {outputs_array.std():.2f}")

    # Analyze (k,m) distribution
    km_dist = defaultdict(int)
    n_values = []
    p_shapes = []

    for sample in inputs:
        if isinstance(sample, list):
            k = int(sample[1])
            m = int(sample[2])
            n = sample[0]
            km_dist[(k, m)] += 1
            n_values.append(n)
            p_shapes.append(sample[3].shape)

    print(f"\n(k,m) Distribution:")
    total = sum(km_dist.values())
    for (k, m), count in sorted(km_dist.items()):
        percentage = (count / total) * 100
        print(f"  k={k}, m={m}: {count:6d} samples ({percentage:5.2f}%)")

    # Analyze n values
    if n_values:
        n_array = np.array(n_values)
        print(f"\nn values:")
        print(f"  Range: [{n_array.min()}, {n_array.max()}]")
        print(f"  Unique: {len(np.unique(n_array))}")

    # Analyze P matrix shapes
    if p_shapes:
        unique_shapes = set(p_shapes)
        print(f"\nP matrix shapes (unique): {len(unique_shapes)}")
        for shape in sorted(unique_shapes):
            count = p_shapes.count(shape)
            print(f"  {shape}: {count} samples")

    return km_dist, outputs_array, n_array if n_values else None

def compare_datasets(ds1_km, ds2_km, ds3_km, combined_km):
    """Compare distributions across datasets"""
    print(f"\n{'='*70}")
    print(f"CROSS-DATASET COMPARISON")
    print(f"{'='*70}")

    all_km = set(ds1_km.keys()) | set(ds2_km.keys()) | set(ds3_km.keys()) | set(combined_km.keys())

    print(f"\n{'(k,m)':<10} {'DS-1':<10} {'DS-2':<10} {'DS-3':<10} {'Combined':<10} {'Total':<10}")
    print("-" * 70)

    for km in sorted(all_km):
        ds1_count = ds1_km.get(km, 0)
        ds2_count = ds2_km.get(km, 0)
        ds3_count = ds3_km.get(km, 0)
        comb_count = combined_km.get(km, 0)
        total = ds1_count + ds2_count + ds3_count

        print(f"{km!s:<10} {ds1_count:<10} {ds2_count:<10} {ds3_count:<10} {comb_count:<10} {total:<10}")

    # Check if combined matches sum
    print(f"\nNote: Combined should include DS-1 + DS-2 (and possibly DS-3)")

def identify_augmentation_opportunities(ds1_km, ds2_km, ds3_km, combined_km):
    """Identify where we need more data"""
    print(f"\n{'='*70}")
    print(f"AUGMENTATION OPPORTUNITIES")
    print(f"{'='*70}")

    # Get total available samples per (k,m) from all sources
    total_available = defaultdict(int)
    for km in set(ds1_km.keys()) | set(ds2_km.keys()) | set(ds3_km.keys()):
        total_available[km] = ds1_km.get(km, 0) + ds2_km.get(km, 0) + ds3_km.get(km, 0)

    # Find min and max
    if total_available:
        min_count = min(total_available.values())
        max_count = max(total_available.values())

        print(f"\nCurrent available data (DS-1 + DS-2 + DS-3):")
        print(f"  Min samples for any (k,m): {min_count}")
        print(f"  Max samples for any (k,m): {max_count}")
        print(f"  Imbalance ratio: {max_count/min_count:.1f}x")

        # Recommend target
        target = max(10000, min_count)  # At least 10k per group
        print(f"\nRecommended augmentation target: {target} samples per (k,m)")

        print(f"\nAugmentation needed per (k,m):")
        for km in sorted(total_available.keys()):
            current = total_available[km]
            needed = max(0, target - current)
            if needed > 0:
                print(f"  {km}: Need {needed:,} more samples (have {current:,})")
            else:
                print(f"  {km}: Sufficient (have {current:,})")

# Main analysis
print("="*70)
print("HEIGHT PREDICTION DATASET ANALYSIS")
print("="*70)

# Load all datasets
print("\nLoading datasets...")
ds1_inputs, ds1_outputs = load_dataset('DS-1-samples_n_k_m_P.pkl', 'DS-1-samples_mHeights.pkl')
ds2_inputs, ds2_outputs = load_dataset('DS-2-Train-n_k_m_P.pkl', 'DS-2-Train-mHeights.pkl')
ds3_inputs, ds3_outputs = load_dataset('DS-3-Train-n_k_m_P.pkl', 'DS-3-Train-mHeights.pkl')
combined_inputs, combined_outputs = load_dataset('combined_ALL_n_k_m_P_exact.pkl', 'combined_ALL_mHeights_exact.pkl')

# Analyze each dataset
ds1_km, ds1_out, ds1_n = analyze_dataset(ds1_inputs, ds1_outputs, "DS-1 (Samples)")
ds2_km, ds2_out, ds2_n = analyze_dataset(ds2_inputs, ds2_outputs, "DS-2 (Train)")
ds3_km, ds3_out, ds3_n = analyze_dataset(ds3_inputs, ds3_outputs, "DS-3 (Train - NEW)")
combined_km, combined_out, combined_n = analyze_dataset(combined_inputs, combined_outputs, "Combined Dataset")

# Compare datasets
compare_datasets(ds1_km, ds2_km, ds3_km, combined_km)

# Identify augmentation opportunities
identify_augmentation_opportunities(ds1_km, ds2_km, ds3_km, combined_km)

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
