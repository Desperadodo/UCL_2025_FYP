import h5py
import os
import torch
import random
from typing import Dict, List, Any
from PIL import Image
from .collator import SFTDataCollatorWith4DAttentionMask
import torch.nn.functional as F

class H5DynamicCollator(SFTDataCollatorWith4DAttentionMask):
    """
    Dynamic H5 file collator for WSI data processing.
    
    This collator efficiently handles H5 files containing WSI patch features,
    dynamically loading and processing them during batch creation.
    """
    
    def __init__(self, max_patches=2048, **kwargs):
        """Initialize the H5 dynamic collator.
        
        Args:
            max_patches (int): Maximum number of patches to process per WSI
            **kwargs: Additional arguments passed to parent collator
        """
        super().__init__(**kwargs)
        self.max_patches = max_patches
        self.h5_key_feat = "features"      # H5 key for patch features
        self.h5_key_coord = "coords_yx"    # H5 key for patch coordinates
    
    def _is_h5_file(self, file_path: str) -> bool:
        """Check if the file is an H5 file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if the file is an H5 file
        """
        return isinstance(file_path, str) and file_path.endswith('.h5')
    
    def _has_h5_images(self, features: List[Dict[str, Any]]) -> bool:
        """Check if the batch contains H5 files.
        
        Args:
            features (List[Dict[str, Any]]): List of feature dictionaries
            
        Returns:
            bool: True if any feature contains H5 files
        """
        for feature in features:
            images = feature.get("images", [])
            if images and any(self._is_h5_file(str(img)) for img in images):
                return True
        return False
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Process a batch of features, handling H5 files dynamically.
        
        This method intelligently detects H5 files and applies appropriate processing:
        - If no H5 files are present, uses standard parent collator processing
        - If H5 files are present, dynamically loads and processes them
        
        Args:
            features (List[Dict[str, Any]]): List of feature dictionaries
            
        Returns:
            Dict[str, torch.Tensor]: Processed batch with pixel_values and patch_coords
        """
        # If no H5 files are present, use standard parent collator processing
        if not self._has_h5_images(features):
            return super().__call__(features)
        
        # H5 files detected, use custom H5 processing logic
        batch_patch_feats = []
        batch_coords = []
        
        # Process H5 files in each feature
        for feature in features:
            images = feature.get("images", [])
            
            if images and self._is_h5_file(str(images[0])):
                h5_path = str(images[0])
                
                if os.path.exists(h5_path):
                    # Dynamically load H5 file features
                    try:
                        with h5py.File(h5_path, "r") as f:
                            feats = torch.from_numpy(f[self.h5_key_feat][()])    # [P, D]
                            coords = torch.from_numpy(f[self.h5_key_coord][()])  # [P, 2]
                            
                            # Random sampling of patches for efficient processing

                            num_patches = feats.shape[0]
                            
                            if num_patches > self.max_patches:
                                # More patches than needed: randomly sample
                                indices = torch.randperm(num_patches)[:self.max_patches]
                                feats = feats[indices]
                                coords = coords[indices]
                            elif num_patches < self.max_patches:
                                # Fewer patches than needed: pad with zeros
                                pad_len = self.max_patches - num_patches
                                
                                # Pad features to match max_patches
                                feat_pad = torch.zeros(pad_len, feats.shape[1], dtype=feats.dtype)
                                feats = torch.cat([feats, feat_pad], dim=0)
                                
                                # Pad coordinates to match max_patches
                                coord_pad = torch.zeros(pad_len, coords.shape[1], dtype=coords.dtype)
                                coords = torch.cat([coords, coord_pad], dim=0)
                            # else: num_patches == self.max_patches, no adjustment needed

                                
                            batch_patch_feats.append(feats)
                            batch_coords.append(coords)
                            
                    except Exception as e:
                        print(f"Error loading H5 file {h5_path}: {e}")
                        # Use zero features as fallback
                        # Note: Feature dimension should be configurable
                        feat_dim = 1536  # TODO: Make this configurable
                        batch_patch_feats.append(torch.zeros(self.max_patches, feat_dim))
                        batch_coords.append(torch.zeros(self.max_patches, 2))
                else:
                    print(f"H5 file {h5_path} does not exist")
                    # H5 file does not exist, use zero features
                    feat_dim = 1536  # TODO: Make this configurable
                    batch_patch_feats.append(torch.zeros(self.max_patches, feat_dim))
                    batch_coords.append(torch.zeros(self.max_patches, 2))
            else:
                print(f"No H5 file found in feature")
                # No H5 file, use zero features
                feat_dim = 1536  # TODO: Make this configurable
                batch_patch_feats.append(torch.zeros(self.max_patches, feat_dim))
                batch_coords.append(torch.zeros(self.max_patches, 2))
        
        # Process text part - clear images to avoid mm_plugin processing H5 files
        text_features = []
        for feature in features:
            text_feature = feature.copy()
            text_feature["images"] = []  # Clear H5 file paths
            text_feature["videos"] = []
            text_feature["audios"] = []
            text_feature["is_h5_processed"] = True  # Flag to signal custom processing
            text_features.append(text_feature)
        
        # Intercept custom attention masks before calling the parent collator
        custom_masks = [feature.pop("attention_mask", None) for feature in text_features]

        # Use the parent class to process the text part
        text_batch = super().__call__(text_features)
         
        # Manually pad and stack the intercepted attention masks (only 2D causal masks)
        if custom_masks:
            max_len = text_batch["input_ids"].shape[1]
            padded_masks = []
            for mask_list in custom_masks:
                if mask_list is None:
                    continue
                mask = torch.as_tensor(mask_list, dtype=torch.float32)
                # Only handle 2D masks here. Skip 1D or other shapes
                if mask.dim() != 2:
                    continue
                pad_len_h = max_len - mask.shape[0]
                pad_len_w = max_len - mask.shape[1]
                padded_mask = F.pad(mask, (0, pad_len_w, 0, pad_len_h), value=0.0)
                padded_masks.append(padded_mask)
            if len(padded_masks) > 0:
                # Stack masks: [B, 1, T, T]
                text_batch["attention_mask"] = torch.stack(padded_masks).unsqueeze(1)

        # Process H5 features
        if batch_patch_feats:
            # All feature tensors should have the same length at this point
            pixel_values = torch.stack(batch_patch_feats)  # [B, P, D]
        else:
            pixel_values = None
            print(f"Warning: pixel_values is None - no H5 features processed")
        
        # Note: Debug code removed - pixel_values should contain actual features
        
        # Combine text batch and H5 features
        result = text_batch.copy()
        # Remove internal flags that should not be passed to model/generate
        if "is_h5_processed" in result:
            result.pop("is_h5_processed", None)
        result["pixel_values"] = pixel_values  # Standard format, compatible with CustomLlavaModel
        result["patch_coords"] = batch_coords  # List[Tensor[P, 2]]
        
        return result 