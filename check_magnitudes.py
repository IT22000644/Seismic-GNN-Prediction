import numpy as np
import collections

d = np.load('data/processed/model_ready/gnn_data.npy', allow_pickle=True).item()
y = d['y']

print(f"Total samples: {len(y)}")
print(f"Min: {y.min():.2f}, Max: {y.max():.2f}")
print(f"Mean: {y.mean():.2f}, Std: {y.std():.2f}")
print(f"\nMagnitude distribution (rounded to 0.1):")

counts = collections.Counter(np.round(y, 1))
for mag in sorted(counts.keys()):
    print(f"  {mag:.1f}: {counts[mag]:4d} ({counts[mag]/len(y)*100:.1f}%)")

print(f"\nLarge earthquakes (>= 5.0): {(y >= 5.0).sum()} ({(y >= 5.0).sum()/len(y)*100:.2f}%)")
print(f"Very large (>= 6.0): {(y >= 6.0).sum()} ({(y >= 6.0).sum()/len(y)*100:.2f}%)")
print(f"Major (>= 7.0): {(y >= 7.0).sum()} ({(y >= 7.0).sum()/len(y)*100:.2f}%)")
