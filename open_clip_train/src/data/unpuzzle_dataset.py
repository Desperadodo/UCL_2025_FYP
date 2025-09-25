"""
Unpuzzle Dataset - Sanitized Version for Public Repository
This module provides a dataset interface for WSI-CLIP training without exposing internal data paths.

For public use, this module requires:
1. Proper configuration of data paths via environment variables or configuration files
2. H5 files containing WSI patch features in the specified format
3. CSV files with image-text pair annotations
"""

import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Any, Tuple
import h5py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WSIGigapathCsv(Dataset):
    """
    WSI GigaPath CSV Dataset - Sanitized version for public use.
    
    This dataset loads WSI features from H5 files and corresponding text annotations
    from CSV files for CLIP training.
    
    Expected H5 format:
    - 'features': [N, D] array of patch features
    - 'coords_yx': [N, 2] array of patch coordinates
    
    Expected CSV format:
    - 'h5_path': Path to H5 file (relative or absolute)
    - 'text': Text annotation corresponding to the WSI
    - 'case_id': Optional case identifier for multi-positive loss
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 csv_dir: str,
                 tokenizer=None,
                 h5_key_feat: str = "features",
                 h5_key_coord: str = "coords_yx",
                 path_col: str = "h5_path",
                 max_patches: Optional[int] = None):
        """
        Initialize WSI GigaPath dataset.
        
        Args:
            df: DataFrame containing dataset annotations
            csv_dir: Directory containing the CSV file (for relative paths)
            tokenizer: Text tokenizer (optional)
            h5_key_feat: Key for features in H5 files
            h5_key_coord: Key for coordinates in H5 files
            path_col: Column name containing H5 file paths
            max_patches: Maximum number of patches to use per WSI
        """
        self.df = df
        self.csv_dir = csv_dir
        self.tokenizer = tokenizer
        self.h5_key_feat = h5_key_feat
        self.h5_key_coord = h5_key_coord
        self.path_col = path_col
        self.max_patches = max_patches
        
        # Validate required columns
        required_cols = [path_col]
        if path_col == "h5_path":
            required_cols.append("text")
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Initialized WSI dataset with {len(df)} samples")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
            - 'image': WSI patch features [P, D]
            - 'text': Text annotation
            - 'case_id': Case identifier (if available)
        """
        row = self.df.iloc[idx]
        
        # Get H5 file path
        h5_path = row[self.path_col]
        if not os.path.isabs(h5_path):
            h5_path = os.path.join(self.csv_dir, h5_path)
        
        # Load WSI features
        try:
            with h5py.File(h5_path, 'r') as f:
                features = torch.from_numpy(f[self.h5_key_feat][()]).float()
                coords = torch.from_numpy(f[self.h5_key_coord][()]).float()
                
                # Apply patch sampling if needed
                if self.max_patches and len(features) > self.max_patches:
                    # Random sampling for training diversity
                    indices = torch.randperm(len(features))[:self.max_patches]
                    features = features[indices]
                    coords = coords[indices]
                elif self.max_patches and len(features) < self.max_patches:
                    # Pad with zeros if fewer patches than needed
                    pad_len = self.max_patches - len(features)
                    feat_pad = torch.zeros(pad_len, features.shape[1])
                    coord_pad = torch.zeros(pad_len, coords.shape[1])
                    features = torch.cat([features, feat_pad], dim=0)
                    coords = torch.cat([coords, coord_pad], dim=0)
                
        except Exception as e:
            logger.warning(f"Error loading H5 file {h5_path}: {e}")
            # Return zero features as fallback
            if self.max_patches:
                features = torch.zeros(self.max_patches, 1536)  # Default GigaPath dimension
                coords = torch.zeros(self.max_patches, 2)
            else:
                features = torch.zeros(1024, 1536)  # Default fallback
                coords = torch.zeros(1024, 2)
        
        # Get text annotation
        text = row.get("text", "").strip()
        if not text:
            # Try alternative column names
            for alt_col in ["caption", "title", "description"]:
                if alt_col in row and row[alt_col]:
                    text = str(row[alt_col]).strip()
                    break
        
        if not text:
            logger.warning(f"No text annotation found for sample {idx}")
            text = "Unknown WSI"  # Fallback text
        
        # Get case ID for multi-positive loss
        case_id = row.get("case_id", f"sample_{idx}")
        
        return {
            "image": features,
            "text": text,
            "case_id": case_id,
            "coords": coords,
            "h5_path": h5_path
        }


def collate_gigapath(batch: List[Dict[str, Any]], max_patches: Optional[int] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str], List[str]]:
    """
    Collate function for WSI GigaPath dataset.
    
    Args:
        batch: List of samples from the dataset
        max_patches: Maximum number of patches per WSI
        
    Returns:
        Tuple of (patch_features, patch_positions, text_annotations, case_ids)
    """
    patch_feats_list = []
    patch_positions = []
    text_ids = []
    case_ids = []
    
    for sample in batch:
        patch_feats_list.append(sample["image"])
        patch_positions.append(sample["coords"])
        text_ids.append(sample["text"])
        case_ids.append(sample["case_id"])
    
    return patch_feats_list, patch_positions, text_ids, case_ids


def get_unpuzzle_dataset(args, preprocess_fn=None, is_train=True, epoch=0, tokenizer=None):
    """
    Get Unpuzzle WSI dataset for CLIP training.
    
    This is a sanitized version that requires proper configuration of data paths.
    
    Args:
        args: Training arguments containing data paths and configuration
        preprocess_fn: Preprocessing function (not used for WSI data)
        is_train: Whether this is training data
        epoch: Current epoch (for distributed training)
        tokenizer: Text tokenizer
        
    Returns:
        DataInfo object containing DataLoader and sampler
    """
    
    # Get data path from arguments
    csv_path = args.train_data if is_train else args.val_data
    if not csv_path:
        raise ValueError(f"Data path not specified for {'training' if is_train else 'validation'}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    path_col = getattr(args, "csv_img_key", None) or "h5_path"
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_path}: {e}")
    
    # Create dataset
    ds = WSIGigapathCsv(
        df=df,
        csv_dir=csv_dir,
        tokenizer=tokenizer,
        h5_key_feat="features",
        h5_key_coord="coords_yx",
        path_col=path_col,
        max_patches=getattr(args, 'patches_per_wsi', None)
    )
    
    # Set up sampler for distributed training
    sampler = None
    if getattr(args, 'distributed', False) and is_train:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            ds,
            num_replicas=getattr(args, 'world_size', 1),
            rank=getattr(args, 'rank', 0),
            shuffle=True,
            seed=getattr(args, 'seed', 0),
            drop_last=is_train,
        )
    
    # Create collate function
    collate_fn = lambda batch: collate_gigapath(batch, max_patches=getattr(args, 'patches_per_wsi', None))
    
    # Create data loader
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=is_train and sampler is None,
        num_workers=getattr(args, 'workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=is_train,
    )
    
    # Set dataset info
    dl.num_samples = len(ds)
    dl.num_batches = len(dl)
    
    # Create DataInfo object (simplified version)
    class DataInfo:
        def __init__(self, dataloader, sampler=None, shared_epoch=None):
            self.dataloader = dataloader
            self.sampler = sampler
            self.shared_epoch = shared_epoch
    
    class SharedEpoch:
        def __init__(self, epoch):
            self.epoch = epoch
    
    shared_epoch = SharedEpoch(epoch)
    return DataInfo(dl, sampler, shared_epoch)


# Example usage and configuration
if __name__ == "__main__":
    print("WSI Unpuzzle Dataset - Sanitized Version")
    print("=" * 50)
    
    print("\nExpected data format:")
    print("CSV columns: h5_path, text, case_id (optional)")
    print("H5 keys: features, coords_yx")
    print("Features shape: [N, 1536] for GigaPath")
    print("Coordinates shape: [N, 2]")
    
    print("\nConfiguration requirements:")
    print("1. Set train_data and val_data paths in training arguments")
    print("2. Ensure H5 files contain features and coords_yx keys")
    print("3. Configure patches_per_wsi for patch sampling")
    print("4. Set up distributed training parameters if needed")
    
    print("\nExample CSV format:")
    print("h5_path,text,case_id")
    print("sample1.h5,This is a breast cancer tissue sample,case_001")
    print("sample2.h5,Pathology report shows malignant cells,case_002")