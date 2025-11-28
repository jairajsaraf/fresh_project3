"""Quick script to inspect the data structure"""
import pickle
import numpy as np

# Load the data
with open('combined_ALL_n_k_m_P_exact.pkl', 'rb') as f:
    inputs_raw = pickle.load(f)

with open('combined_ALL_mHeights_exact.pkl', 'rb') as f:
    outputs_raw = pickle.load(f)

print("Input data structure:")
print(f"Type: {type(inputs_raw)}")
print(f"Length: {len(inputs_raw)}")
print(f"\nFirst sample:")
print(f"Type of first sample: {type(inputs_raw[0])}")
print(f"First sample: {inputs_raw[0]}")
print(f"Length of first sample: {len(inputs_raw[0])}")

print("\n\nBreaking down first sample:")
for i, elem in enumerate(inputs_raw[0]):
    print(f"Element {i}: type={type(elem)}, value/shape={elem if not isinstance(elem, np.ndarray) else elem.shape}, sample={elem if not isinstance(elem, np.ndarray) else elem[:5] if len(elem) > 5 else elem}")

print("\n\nOutput data structure:")
print(f"Type: {type(outputs_raw)}")
print(f"Length: {len(outputs_raw)}")
print(f"First output: {outputs_raw[0]}")
print(f"Type of first output: {type(outputs_raw[0])}")

# Check a few more samples
print("\n\nSample n, k, m values from first 10 samples:")
for i in range(min(10, len(inputs_raw))):
    sample = inputs_raw[i]
    print(f"Sample {i}: n={sample[0]}, k={sample[1]}, m={sample[2]}, P_shape={sample[3].shape if isinstance(sample[3], np.ndarray) else 'not array'}")
