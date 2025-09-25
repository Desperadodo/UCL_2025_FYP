import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from ModelBase.Get_ROI_model import build_ROI_backbone_model
from ModelBase.Get_WSI_model import build_WSI_backbone_model
from ModelBase.custom_decoder import CrossAttentionWrapper

class VisionModelWithProjection(nn.Module):
    """Wrapper class that adds projection layer to vision model to match CLIP embedding dimension."""
    
    def __init__(self, vision_model: nn.Module, embed_dim: int, proj_type: str = 'mlp'):
        super().__init__()
        self.vision_model = vision_model
        
        # Get the output dimension of the vision model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)  # Assuming standard image size
            output = vision_model(dummy_input)
            vision_dim = output.shape[-1]
        
        # Create projection layer
        if proj_type == 'linear':
            self.proj = nn.Linear(vision_dim, embed_dim, bias=False)
        elif proj_type == 'mlp':
            hidden_size = (vision_dim + embed_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(vision_dim, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, embed_dim, bias=False),
            )
        else:
            self.proj = nn.Identity()
            
    def forward(self, x):
        features = self.vision_model(x)
        return self.proj(features)

class WSIModelWithProjection(nn.Module):
    """Wrapper class that adds projection layer to vision model(with LongNet) to match CLIP embedding dimension."""
    def __init__(self, wsi_model: nn.Module, in_dim: int, out_dim: int):
        super().__init__()
        self.wsi_model = wsi_model
        self.proj = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, *args, **kwargs):
        feats = self.wsi_model(*args, **kwargs)
        return self.proj(feats)

def build_custom_vision_tower(
    model_name: str,
    model_cfg: Dict[str, Any],
    embed_dim: int = 768,
    vision_type: Optional[str] = None, # slide, patch
):

    proj_type = model_cfg.pop('proj_type', 'mlp')

    if vision_type == 'slide':
        # ------------------- LongNet / GigaPath  -------------------
        # return slide_backbone, roi_dim, slide_embed_dim
        if model_name == "gigapath":
            # model_name = "gigapath", vision_type = "slide"
            slide_backbone, _roi_dim, slide_embed_dim = build_WSI_backbone_model(
                model_name=model_name,
                **model_cfg
            )
            return WSIModelWithProjection(slide_backbone, slide_embed_dim, embed_dim)
        else:
            raise ValueError(f"non-gigapath WSI model is not supported yet")
    else:
        # ------------------- Normal ROI model (ROI, MIZero, ...) -------------------
        vision_model = build_ROI_backbone_model(
            model_name=model_name,
            num_classes=0,  
            edge_size=model_cfg.get('image_size', 224),
            **model_cfg
        )
        return VisionModelWithProjection(vision_model, embed_dim, proj_type)

def build_custom_text_tower(
    model_name: str,
    model_cfg: Dict[str, Any],
):
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **model_cfg)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# deprecated
# def build_custom_decoder_tower(
#     model_name: str,
#     model_cfg: Dict[str, Any],
# ):
#     model = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg)
#     tokenizer = AutoTokenizer.from_pretrained(model_name, **model_cfg)
#     model = CrossAttentionWrapper(model, tokenizer, **model_cfg)

#     return model

