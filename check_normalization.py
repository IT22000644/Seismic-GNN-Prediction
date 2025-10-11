import pandas as pd
import os

csv_path = 'data/processed/earthquakes_with_features.csv'
print(f"Checking {csv_path}...")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    print(f"\nColumns with 'mag': {[c for c in df.columns if 'mag' in c.lower()]}")

    if 'magnitude_normalized' in df.columns:
        print(f"\n✓ magnitude_normalized EXISTS!")
        print(f"  Min: {df['magnitude_normalized'].min():.6f}")
        print(f"  Max: {df['magnitude_normalized'].max():.6f}")
        print(f"  Mean: {df['magnitude_normalized'].mean():.6f}")
    else:
        print("\n✗ magnitude_normalized does NOT exist")
        print("  The normalization cell did not execute!")
else:
    print(f"CSV file not found at {csv_path}")

# Check scaler
scaler_path = 'data/processed/scalers/scaler_target.pkl'
print(f"\nChecking {scaler_path}...")
print(f"  Exists: {os.path.exists(scaler_path)}")
