"""
Script to read and display all keys in an H5 file.
"""
import h5py
import numpy as np
import sys


def print_h5_structure(h5_path, indent=0):
    """
    Recursively print the structure of an H5 file.
    
    Args:
        h5_path: Path to the H5 file
        indent: Indentation level for nested structures
    """
    prefix = "  " * indent
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\n{'='*60}")
        print(f"H5 File: {h5_path}")
        print(f"{'='*60}")
        
        def print_structure(name, obj):
            current_indent = "  " * (indent + 1)
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                print(f"{current_indent}{name}: Dataset shape={shape}, dtype={dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{current_indent}{name}: Group")
        
        f.visititems(print_structure)
        
        # Print summary of top-level keys
        print(f"\n{prefix}Top-level keys: {list(f.keys())}")
        
        # Try to show sample data for each dataset
        print(f"\n{prefix}Sample data:")
        for key in f.keys():
            try:
                data = f[key]
                if isinstance(data, h5py.Dataset):
                    print(f"{prefix}  {key}:")
                    print(f"{prefix}    shape: {data.shape}")
                    print(f"{prefix}    dtype: {data.dtype}")
                    # Show first few values for 1D arrays
                    if len(data.shape) == 1:
                        sample = data[:5]
                        print(f"{prefix}    first 5 values: {sample}")
                    elif len(data.shape) == 2:
                        print(f"{prefix}    first row: {data[0, :5]}")
            except Exception as e:
                print(f"{prefix}  {key}: Error reading - {e}")


def read_all_keys(h5_path):
    """
    Read and return all keys from an H5 file.
    
    Args:
        h5_path: Path to the H5 file
        
    Returns:
        Dictionary with key information
    """
    keys_info = {}
    
    with h5py.File(h5_path, 'r') as f:
        def collect_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                keys_info[name] = {
                    'type': 'Dataset',
                    'shape': obj.shape,
                    'dtype': obj.dtype
                }
            elif isinstance(obj, h5py.Group):
                keys_info[name] = {
                    'type': 'Group'
                }
        
        f.visititems(collect_keys)
    
    return keys_info


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_h5_keys.py <path_to_h5_file>")
        print("\nExample:")
        print("  python read_h5_keys.py /path/to/features.h5")
        sys.exit(1)
    
    h5_path = sys.argv[1]
    print_h5_structure(h5_path)
    
    # Also output as python dict for programmatic use
    keys_info = read_all_keys(h5_path)
    print(f"\n{'='*60}")
    print("Keys as dictionary:")
    print(f"{'='*60}")
    for key, info in keys_info.items():
        print(f"  '{key}': {info}")
