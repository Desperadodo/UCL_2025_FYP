"""
WSI-CLIP Module - Sanitized Version for Public Repository

This module provides WSI-CLIP model implementations for whole slide image analysis.
It includes custom loss functions and model architectures designed for medical imaging.

Key Components:
- WSI_CLIP: Main model architecture for WSI-CLIP
- MultiPosClipLoss: Custom loss function for multi-positive training
- Custom model builders and factory functions

For usage examples, see the examples/ directory.
"""

from .factory import create_model, create_loss, list_models
from .model import WSI_CLIP, CustomCLIP
from .loss import MultiPosClipLoss, ClipLoss
from .custom_model_builder import VisionModelWithProjection, WSIModelWithProjection

__version__ = "1.0.0"
__author__ = "Research Team"

__all__ = [
    "WSI_CLIP",
    "CustomCLIP", 
    "MultiPosClipLoss",
    "ClipLoss",
    "create_model",
    "create_loss",
    "list_models",
    "VisionModelWithProjection",
    "WSIModelWithProjection"
]