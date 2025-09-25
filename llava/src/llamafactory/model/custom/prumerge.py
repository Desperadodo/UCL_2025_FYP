import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def complement_idx(idx, dim):
    """
    Compute the complement of a set of indices.
    """
    device = idx.device
    a = torch.arange(dim, device=device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl


class PruMergeModule(nn.Module):
    """
    PruMerge module for CLS-attention guided pruning and merging of visual tokens.
    
    This module implements a custom pruning and merging strategy for WSI visual tokens,
    based on the attention scores from the [CLS] token to patch tokens.
    """
    
    def __init__(self, target_token_num=256, target_layer=11, k_per_token=32):
        super().__init__()
        self.target_token_num = target_token_num  # Target number of tokens after pruning
        self.target_layer = target_layer          # Vision encoder layer to extract attention from
        self.k_per_token = k_per_token            # Number of tokens to merge into each top-k token
        self._outputs = {}                        # Storage for hook outputs

    def _hook_q(self, module, input, output):
        self._outputs['q'] = output

    def _hook_k(self, module, input, output):
        self._outputs['k'] = output

    def forward(self, vision_tower: nn.Module, pixel_values: torch.Tensor, patch_coords: list):
        """
        Forward pass for PruMerge: CLS-attention guided pruning and merging.
        
        This method:
        1. Extracts attention scores from [CLS] token to patch tokens
        2. Prunes patches by selecting top-k based on attention scores
        3. Merges information from non-selected patches into selected ones
        
        Args:
            vision_tower (nn.Module): The vision encoder (e.g., GigaPath)
            pixel_values (torch.Tensor): Input patch features [B, P, D_in]
            patch_coords (list[torch.Tensor]): Patch coordinates [B, P, 2]
            
        Returns:
            torch.Tensor: Pruned and merged features [B, target_token_num+1, D_out]
        """
        # Convert list of tensors to a single tensor if needed
        if isinstance(patch_coords, list):
            patch_coords = torch.stack(patch_coords, dim=0)

        device = pixel_values.device
        dtype = pixel_values.dtype

        # 1. Register hooks to extract Key and Query from the target layer
        # Note: This path is specific to LongNetViT architecture
        # For other architectures, adjust the path accordingly
        vision_encoder = vision_tower.encoder
        
        q_proj_hook = vision_encoder.layers[self.target_layer].self_attn.q_proj.register_forward_hook(self._hook_q)
        k_proj_hook = vision_encoder.layers[self.target_layer].self_attn.k_proj.register_forward_hook(self._hook_k)

        # 2. Forward pass through vision tower to get features and trigger hooks
        # We need the full output including [CLS] token
        full_features = vision_tower(pixel_values, patch_coords, output_mode='all')
        
        # 3. Remove hooks immediately after use to avoid memory leaks
        q_proj_hook.remove()
        k_proj_hook.remove()

        # 4. Extract Q, K and compute attention scores
        # Shape: [B, N+1, C] where N is number of patches
        q = self._outputs['q']
        k = self._outputs['k']
        self._outputs = {}  # Clear for next forward pass
        
        if q is None or k is None:
            raise ValueError("Hooks did not capture Q and K values. Check layer path and architecture.")

        # Extract [CLS] token query (at index 0)
        cls_q = q[:, 0, :].unsqueeze(1)  # [B, 1, C]
        
        # Compute attention scores from [CLS] token to all other tokens
        # Shape: [B, 1, N+1] -> [B, N+1]
        attn = (cls_q @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
        cls_attn_to_all = F.softmax(attn, dim=-1).squeeze(1)  # [B, N+1]
        
        # Extract attention scores to patch tokens (exclude [CLS] token self-attention)
        cls_attn_to_patches = cls_attn_to_all[:, 1:]  # [B, N]

        # 5. Pruning: Select top-k tokens based on CLS attention scores
        _, topk_indices = torch.topk(cls_attn_to_patches, self.target_token_num, dim=1)  # [B, K]
        
        # Extract embeddings and keys for top-k and non-top-k tokens
        C = full_features.shape[-1]
        # Get patch features (exclude [CLS] token at index 0)
        patch_features = full_features[:, 1:]  # [B, N, C]
        patch_keys = k[:, 1:]                  # [B, N, C]
        
        # Gather top-k features and keys
        topk_features = torch.gather(patch_features, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C))  # [B, K, C]
        topk_keys = torch.gather(patch_keys, 1, topk_indices.unsqueeze(-1).expand(-1, -1, C))          # [B, K, C]
        
        # Get non-top-k indices and corresponding features
        non_topk_indices = complement_idx(topk_indices, patch_features.shape[1])
        non_topk_features = torch.gather(patch_features, 1, non_topk_indices.unsqueeze(-1).expand(-1, -1, C))
        non_topk_keys = torch.gather(patch_keys, 1, non_topk_indices.unsqueeze(-1).expand(-1, -1, C))
        non_topk_cls_attn = torch.gather(cls_attn_to_patches, 1, non_topk_indices)

        # 6. Merging: Fuse information from non-top-k tokens into top-k tokens
        # Normalize keys for similarity computation
        topk_keys_norm = F.normalize(topk_keys, p=2, dim=-1)
        non_topk_keys_norm = F.normalize(non_topk_keys, p=2, dim=-1)
        
        # Calculate similarity matrix between top-k and non-top-k tokens
        sim_matrix = torch.bmm(topk_keys_norm, non_topk_keys_norm.transpose(1, 2))  # [B, K, N-K]

        # Find most similar non-top-k tokens for each top-k token
        _, merge_indices = torch.topk(sim_matrix, k=self.k_per_token, dim=-1)  # [B, K, k_per_token]
        
        # Gather features and attention weights for merging
        # non_topk_features: [B, N-K, C], merge_indices: [B, K, k_per_token]
        # Use loop for straightforward gathering operation
        
        # Initialize merged features
        merged_features = torch.zeros_like(topk_features)
        
        for i in range(topk_features.shape[1]):
            # Get indices of non-topk tokens to merge into the i-th topk token
            current_merge_indices = merge_indices[:, i, :]  # [B, k_per_token]
            
            # Gather features and attention weights for merging
            tokens_to_merge = torch.gather(non_topk_features, 1, current_merge_indices.unsqueeze(-1).expand(-1, -1, C))  # [B, k_per_token, C]
            attn_to_merge = torch.gather(non_topk_cls_attn, 1, current_merge_indices)  # [B, k_per_token]
            
            # Compute weighted average of tokens to merge
            weighted_avg = torch.sum(tokens_to_merge * attn_to_merge.unsqueeze(-1), dim=1) / (torch.sum(attn_to_merge, dim=1, keepdim=True) + 1e-6)
            
            # Fuse merged information with original topk token
            merged_features[:, i, :] = topk_features[:, i, :] + weighted_avg
            
        # Add the original [CLS] token back to the sequence
        cls_token_feature = full_features[:, 0:1, :]
        final_features = torch.cat([cls_token_feature, merged_features], dim=1)

        return final_features 