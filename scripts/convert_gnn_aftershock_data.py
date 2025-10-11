"""
Convert aftershock GNN data from NumPy format to PyTorch Geometric format.

Loads the gnn_data_aftershock.npy file and converts it to a PyTorch Geometric
Data object with binary classification labels.
"""

import numpy as np
import torch
from torch_geometric.data import Data
import os

def convert_aftershock_data():
    """Convert aftershock NumPy data to PyTorch Geometric format."""

    print("="*70)
    print("CONVERTING AFTERSHOCK GNN DATA TO PYTORCH GEOMETRIC FORMAT")
    print("="*70)

    # Load the NumPy data
    data_path = '../data/processed/model_ready/gnn_data_aftershock.npy'
    print(f"\nLoading data from: {data_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Aftershock GNN data not found at {data_path}.\n"
            "Please run the feature engineering notebook (03_feature_engineering.ipynb) "
            "and ensure the aftershock labeling section has been executed."
        )

    gnn_data = np.load(data_path, allow_pickle=True).item()

    # Extract components
    x = torch.tensor(gnn_data['x'], dtype=torch.float)
    y = torch.tensor(gnn_data['y'], dtype=torch.long)  # Binary labels (0 or 1)
    edge_index = torch.tensor(gnn_data['edge_index'], dtype=torch.long)

    train_mask = torch.tensor(gnn_data['train_mask'], dtype=torch.bool)
    val_mask = torch.tensor(gnn_data['val_mask'], dtype=torch.bool)
    test_mask = torch.tensor(gnn_data['test_mask'], dtype=torch.bool)

    # Create PyTorch Geometric Data object
    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    print(f"✓ Converted successfully!")
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

    # Validate data
    print("\n" + "="*70)
    print("VALIDATING PYTORCH GEOMETRIC GRAPH")
    print("="*70)

    print(f"\n📊 Graph Summary:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Node features: {data.num_node_features}")

    print(f"\n🎯 Target Variable (Aftershock Triggering):")
    print(f"  Class 0 (no aftershock): {(y == 0).sum().item()} ({(y == 0).float().mean()*100:.1f}%)")
    print(f"  Class 1 (triggers aftershock): {(y == 1).sum().item()} ({(y == 1).float().mean()*100:.1f}%)")

    print(f"\n✂️ Data Splits:")
    print(f"  Train: {train_mask.sum().item()} nodes ({train_mask.float().mean()*100:.1f}%)")
    print(f"    - Class 0: {(y[train_mask] == 0).sum().item()}, Class 1: {(y[train_mask] == 1).sum().item()}")
    print(f"  Val: {val_mask.sum().item()} nodes ({val_mask.float().mean()*100:.1f}%)")
    print(f"    - Class 0: {(y[val_mask] == 0).sum().item()}, Class 1: {(y[val_mask] == 1).sum().item()}")
    print(f"  Test: {test_mask.sum().item()} nodes ({test_mask.float().mean()*100:.1f}%)")
    print(f"    - Class 0: {(y[test_mask] == 0).sum().item()}, Class 1: {(y[test_mask] == 1).sum().item()}")

    # Check for NaN/Inf
    has_nan = torch.isnan(x).any()
    has_inf = torch.isinf(x).any()

    if has_nan or has_inf:
        print(f"\n⚠️ WARNING: Data contains NaN or Inf values!")
        print(f"  NaN values: {has_nan}")
        print(f"  Inf values: {has_inf}")
    else:
        print(f"\n✓ No NaN or Inf values detected")

    # Check edge index validity
    max_node_idx = edge_index.max().item()
    if max_node_idx >= data.num_nodes:
        print(f"\n⚠️ WARNING: Edge index contains invalid node indices!")
        print(f"  Max index: {max_node_idx}, Num nodes: {data.num_nodes}")
    else:
        print(f"✓ All edge indices are valid")

    # Save the PyTorch Geometric data
    output_path = '../data/processed/model_ready/earthquake_graph_aftershock.pt'
    torch.save(data, output_path)

    print(f"\n{'='*70}")
    print("✓ VALIDATION COMPLETE - Graph is ready for training!")
    print(f"{'='*70}")

    print(f"\nSaved to: {output_path}")

    return data


if __name__ == '__main__':
    convert_aftershock_data()
    print("\n✓ Conversion complete!")
