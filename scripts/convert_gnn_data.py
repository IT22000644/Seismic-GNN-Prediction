"""
Convert your gnn_data.npy (dictionary format) to PyTorch Geometric Data object.

Your gnn_data.npy structure (from feature engineering):
{
    'x': node features [num_nodes, num_features],
    'y': target magnitudes [num_nodes],
    'edge_index': edge connectivity [2, num_edges],
    'train_mask': boolean mask for training,
    'val_mask': boolean mask for validation,
    'test_mask': boolean mask for test,
    'num_features': int
}
"""

import numpy as np
import torch
from torch_geometric.data import Data
import os

# =============================================================================
# Convert gnn_data.npy to PyTorch Geometric format
# =============================================================================

def convert_gnn_data_to_pyg(npy_path='processed/model_ready/gnn_data.npy', 
                           save_path='processed/model_ready/earthquake_graph.pt'):
    """
    Convert your gnn_data.npy dictionary to PyG Data object.
    
    Args:
        npy_path: path to gnn_data.npy
        save_path: where to save the PyG graph
    
    Returns:
        PyG Data object ready for training
    """
    
    print(f"Loading data from: {npy_path}")
    gnn_dict = np.load(npy_path, allow_pickle=True).item()
    
    # Extract components
    node_features = gnn_dict['x']
    targets = gnn_dict['y']
    edge_index = gnn_dict['edge_index']
    train_mask = gnn_dict['train_mask']
    val_mask = gnn_dict['val_mask']
    test_mask = gnn_dict['test_mask']
    
    # Edge index should be [2, num_edges] format
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T
    
    # Calculate edge weights (Euclidean distances between connected nodes)
    from scipy.spatial.distance import euclidean
    
    edge_weights = []
    for i in range(edge_index.shape[1]):
        src_node = edge_index[0, i]
        dst_node = edge_index[1, i]
        src_coords = node_features[src_node, :2]
        dst_coords = node_features[dst_node, :2]
        dist = euclidean(src_coords, dst_coords)
        edge_weights.append(dist)
    
    edge_weights = np.array(edge_weights)
    
    # Create PyTorch Geometric Data object
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1),
        y=torch.tensor(targets, dtype=torch.float32),
        train_mask=torch.tensor(train_mask, dtype=torch.bool),
        val_mask=torch.tensor(val_mask, dtype=torch.bool),
        test_mask=torch.tensor(test_mask, dtype=torch.bool)
    )
    
    # Save the PyG Data object
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)
    
    print(f"✓ Converted successfully!")
    print(f"  Nodes: {data.num_nodes}, Edges: {data.num_edges}, Features: {data.num_node_features}")
    print(f"  Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    print(f"  Saved to: {save_path}")
    
    return data


# =============================================================================
# Quick validation function
# =============================================================================

def validate_pyg_graph(graph_path='processed/model_ready/earthquake_graph.pt'):
    """
    Validate the created PyG graph and print summary.
    """
    print("="*70)
    print("VALIDATING PYTORCH GEOMETRIC GRAPH")
    print("="*70)
    
    data = torch.load(graph_path, weights_only=False)
    
    print(f"\n📊 Graph Summary:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Node features: {data.num_node_features}")
    
    print(f"\n🎯 Target Variable (Magnitude):")
    print(f"  Min: {data.y.min().item():.2f}")
    print(f"  Max: {data.y.max().item():.2f}")
    print(f"  Mean: {data.y.mean().item():.2f}")
    print(f"  Std: {data.y.std().item():.2f}")
    
    print(f"\n✂️ Data Splits:")
    print(f"  Train: {data.train_mask.sum().item()} nodes ({data.train_mask.sum().item()/data.num_nodes*100:.1f}%)")
    print(f"  Val: {data.val_mask.sum().item()} nodes ({data.val_mask.sum().item()/data.num_nodes*100:.1f}%)")
    print(f"  Test: {data.test_mask.sum().item()} nodes ({data.test_mask.sum().item()/data.num_nodes*100:.1f}%)")
    
    print(f"\n🔗 Edge Statistics:")
    print(f"  Average degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"  Has self-loops: {data.has_self_loops()}")
    print(f"  Is undirected: {data.is_undirected()}")
    
    if hasattr(data, 'edge_attr'):
        print(f"\n📏 Edge Attributes (distances):")
        print(f"  Min distance: {data.edge_attr.min().item():.4f}")
        print(f"  Max distance: {data.edge_attr.max().item():.4f}")
        print(f"  Mean distance: {data.edge_attr.mean().item():.4f}")
    
    print(f"\n{'='*70}")
    print("✓ VALIDATION COMPLETE - Graph is ready for training!")
    print(f"{'='*70}")
    
    return data


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    """
    Run this script to convert your gnn_data.npy to PyG format.
    
    Usage:
        python convert_gnn_data.py
    
    Or in Jupyter:
        %run convert_gnn_data.py
    """
    
    # Paths
    INPUT_PATH = '../data/processed/model_ready/gnn_data.npy'
    OUTPUT_PATH = '../data/processed/model_ready/earthquake_graph.pt'
    
    # Check if input file exists
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Input file not found at {INPUT_PATH}")
        print(f"\nPlease ensure you've run the feature engineering notebook first.")
        exit(1)
    
    # Convert
    data = convert_gnn_data_to_pyg(
        npy_path=INPUT_PATH,
        save_path=OUTPUT_PATH
    )
    
    # Validate
    print("\n")
    validate_pyg_graph(OUTPUT_PATH)
    
    print(f"\n{'='*70}")
    print("NEXT STEPS:")
    print(f"{'='*70}")
    print(f"\n1. Use this graph in your training script:")
    print(f"   data = torch.load('{OUTPUT_PATH}')")
    print(f"\n2. The data already has train/val/test masks")
    print(f"\n3. Run the GNN training scripts from the previous artifacts")
    print(f"\n4. Models will use:")
    print(f"   - data.x: node features")
    print(f"   - data.edge_index: graph structure")
    print(f"   - data.edge_attr: edge distances")
    print(f"   - data.y: target magnitudes")
    print(f"   - data.train_mask/val_mask/test_mask: splits")