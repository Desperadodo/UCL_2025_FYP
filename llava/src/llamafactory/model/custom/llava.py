import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel, CONFIG_MAPPING, GenerationMixin, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.utils import LossKwargs, is_torchdynamo_compiling
from abc import ABC, abstractmethod
import os
from typing import Optional, Tuple, List, Union, Dict, Any
from dataclasses import dataclass

# Import backbone model builders
# Note: These modules should be provided by the user's codebase
from ModelBase.Get_ROI_model import build_ROI_backbone_model
from ModelBase.Get_WSI_model import build_WSI_backbone_model
from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from .mm_utils import get_anyres_image_grid_shape
from .prumerge import PruMergeModule


# logger = logging.get_logger(__name__)


class CustomVisionConfig(PretrainedConfig):
    """
    Custom Vision Configuration for both ROI and WSI models.
    
    This configuration supports:
    - ROI models: CONCH, UNI, etc.
    - WSI models: Gigapath, MI-Zero.
    """
    
    model_type = "custom_vision"
    
    def __init__(
        self,
        # Basic configuration
        vision_mode: str = "wsi",                    # "roi" or "wsi"
        model_name: str = "gigapath",                   # "conch", "uni", "gigapath", "SlideViT", etc.
        hidden_size: int = 768,                      # Output feature dimension
        
        # ROI mode configuration
        image_size: int = 224,                       # ROI image size
        patch_size: int = 16,                        # Patch size
        num_channels: int = 3,                       # Number of image channels
        
        # WSI mode configuration (mainly for Gigapath)
        ROI_feature_dim: Optional[int] = 1536,       # Input ROI feature dimension for Gigapath
        slide_embed_dim: int = 768,                  # Slide-level embedding dimension
        MTL_token_num: int = 0,                      # Multi-task token number
        heatmap: bool = False,                       # Whether to generate heatmap
        
        # Weight configuration
        local_weight_path: Optional[str] = None,     # Local weight path
        pretrained: bool = True,                     # Whether to use pretrained weights
        
        # Other configurations
        patches_per_wsi: int = 2048,
        **kwargs,
    ):
        self.vision_mode = vision_mode
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        # ROI configuration
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        
        # WSI configuration
        self.ROI_feature_dim = ROI_feature_dim
        self.slide_embed_dim = slide_embed_dim
        self.MTL_token_num = MTL_token_num
        self.heatmap = heatmap
        
        # Weight configuration
        self.local_weight_path = local_weight_path
        self.pretrained = pretrained
        
        # Set default values based on model type
        if vision_mode == "wsi" and model_name.startswith("gigapath"):
            if ROI_feature_dim is None:
                self.ROI_feature_dim = 1536  # Default input dimension for Gigapath
            self.hidden_size = slide_embed_dim  # In WSI mode, hidden_size equals slide_embed_dim
        
        # Store other kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.patches_per_wsi = patches_per_wsi
        
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format for passing to build_*_backbone_model functions
        """
        config_dict = {}
        
        # Basic parameters
        config_dict["model_name"] = self.model_name
        config_dict["local_weight_path"] = self.local_weight_path
        
        if self.vision_mode == "wsi":
            # WSI mode parameters
            config_dict["ROI_feature_dim"] = self.ROI_feature_dim
            config_dict["MTL_token_num"] = self.MTL_token_num
            config_dict["heatmap"] = self.heatmap
        else:
            # ROI mode parameters
            config_dict["image_size"] = self.image_size
            config_dict["patch_size"] = self.patch_size
            config_dict["num_channels"] = self.num_channels
        
        # Add other attributes
        for key, value in self.__dict__.items():
            if key not in config_dict and not key.startswith('_'):
                config_dict[key] = value
        
        return config_dict


# Custom configuration
class CustomLLaVAModelConfig(PretrainedConfig):

    model_type = "custom_llava"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": CustomVisionConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_index=128256,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vision_feature_select_mode="prumerge",
        image_seq_length=576,
        multimodal_projector_bias=True,
        projector_dropout: float = 0.1,
        gamma_init: float = 2.0,
        gamma_min: float = 0.5,
        gamma_max: float = 1.5,
        **kwargs,
    ):
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.image_seq_length = image_seq_length
        self.projector_dropout = projector_dropout
        self.gamma_init = gamma_init
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_select_mode = vision_feature_select_mode
        self.vision_feature_layer = vision_feature_layer

        # Handle vision_config - support both Stage2 (object) and Stage3 (dict from checkpoint)
        # Stage2: Creating new config with CustomVisionConfig object
        # Stage3: Loading from checkpoint with dict format
        if isinstance(vision_config, dict):
            # Stage3: Loading from checkpoint, vision_config is dict
            self.vision_config = CustomVisionConfig(**vision_config)
        elif vision_config is None:
            # Stage2: Creating new config, use default
            self.vision_config = CustomVisionConfig()
        else:
            # Stage2: vision_config is already CustomVisionConfig object
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        self.text_config = text_config
        self.multimodal_projector_bias = multimodal_projector_bias

        super().__init__(**kwargs)


class CustomLlavaModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for Llava outputs, with hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    def __init__(
        self,
        last_hidden_state: torch.FloatTensor = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        attentions: Optional[Tuple[torch.FloatTensor]] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
    ):
        super().__init__(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )
        self.image_hidden_states = image_hidden_states


@dataclass
class CustomLlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for Llava causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`torch.FloatTensor`, *optional*):
            A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
            image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None


class CustomLlavaMultiModalProjector(nn.Module):
    def __init__(self, config: CustomLLaVAModelConfig):
        super().__init__()
        # We have hidden_size * the number of vision feature layers
        num_feature_layers = 1 if isinstance(config.vision_feature_layer, int) else len(config.vision_feature_layer)
        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size * num_feature_layers,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.ln1 = nn.LayerNorm(config.text_config.hidden_size)
        self.act = ACT2FN[config.projector_hidden_act]
        self.dropout = nn.Dropout(config.projector_dropout)

        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )
        self.ln2 = nn.LayerNorm(config.text_config.hidden_size)
        
        # Replace gamma with a logit for bounded mapping
        self.gamma_logit = nn.Parameter(torch.zeros(1))
        self.gamma_min = config.gamma_min
        self.gamma_max = config.gamma_max


    def forward(self, image_features):
        hidden_states = self.linear_1(image_features)
        hidden_states = self.ln1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        hidden_states = self.ln2(hidden_states)

        # Apply bounded gamma scaling
        gamma = self.gamma_min + torch.sigmoid(self.gamma_logit) * (self.gamma_max - self.gamma_min)
        hidden_states = gamma * hidden_states
        
        return hidden_states


class CustomLlavaPreTrainedModel(PreTrainedModel):
    config_class = CustomLLaVAModelConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = getattr(self.config, "initializer_range", self.config.get_text_config().initializer_range)

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()



class CustomLlavaModel(CustomLlavaPreTrainedModel):

    def __init__(self, config: CustomLLaVAModelConfig):
        super().__init__(config)
        
        if getattr(config.vision_config, "vision_mode", "roi") == "wsi":
            # build_WSI_backbone_model returns (slide_backbone, default_ROI_feature_dim, slide_embed_dim)
            wsi_backbone, _, _ = build_WSI_backbone_model(**config.vision_config.to_dict())
            self.vision_tower = wsi_backbone
            
            # Conditionally instantiate the PruMerge module
            if config.vision_feature_select_mode == 'prumerge':
                self.vision_feature_processor = PruMergeModule(
                    target_token_num=getattr(config, "image_seq_length", 256),
                )
            else:
                self.vision_feature_processor = None

        else:
            # For ROI models, we might not need PruMerge, or need a different configuration
            self.vision_tower = build_ROI_backbone_model(**config.vision_config.to_dict())
            self.vision_feature_processor = self.vision_tower # Pass-through for non-WSI models for now
            
        self.multi_modal_projector = CustomLlavaMultiModalProjector(config)
        
        # Conditionally create the text query projection layer
        if config.vision_feature_select_mode == 'prompt_topk':
            self.text_query_projection = nn.Linear(config.text_config.hidden_size, config.vision_config.slide_embed_dim)
        else:
            self.text_query_projection = None

        # fix: ensure LLM loads pretrained weights instead of random initialization
        # check if llm_name_or_path attribute is set (should be set when loading config)
        llm_path = getattr(config, 'llm_name_or_path', None)
        model_name_or_path = getattr(config, 'model_name_or_path', None)
        train_from_scratch = getattr(config, 'train_from_scratch', False)

        if llm_path and not train_from_scratch:
            # Load pretrained LLM weights if path is provided and not training from scratch
            self.language_model = AutoModel.from_pretrained(llm_path, config=config.text_config, ignore_mismatched_sizes=True)
        else:
            # Create LLM from config (random initialization)
            self.language_model = AutoModel.from_config(config.text_config)
            
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        patch_coords: Optional[List[torch.FloatTensor]] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Obtains image features from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)` or `(batch_size, num_patches, feature_dim)`):
               The tensors corresponding to the input images or patch features.
            patch_coords (`List[torch.FloatTensor]`, *optional*):
               List of coordinate tensors for WSI patch positions. Required for WSI models.
            vision_feature_layer (`Union[int, List[int]]`, *optional*):
                The index of the layer to select the vision feature. If multiple indices are provided,
                the vision feature of the corresponding indices will be concatenated to form the
                vision features.
            vision_feature_select_strategy (`str`, *optional*):
                The feature selection strategy used to select the vision feature from the vision backbone.
                Can be one of `"default"` or `"full"`
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        # Get the feature selection mode from the config
        select_mode = getattr(self.config, "vision_feature_select_mode", "prumerge")

        # Ensure patch_coords is a tensor before being used
        if patch_coords is not None and isinstance(patch_coords, list):
            patch_coords = torch.stack(patch_coords, dim=0)

        # Process features based on the selected mode
        if select_mode == 'prumerge':
            if isinstance(self.vision_feature_processor, PruMergeModule):
                # PruMergeModule handles the call to vision_tower internally
                image_features = self.vision_feature_processor(self.vision_tower, pixel_values, patch_coords)
            else:
                # print("Warning: vision_feature_select_mode is 'prumerge', but not using PruMergeModule. Getting raw features.")
                image_features = self.vision_tower(pixel_values, patch_coords, output_mode='all') if patch_coords is not None else self.vision_tower(pixel_values, output_mode='all')
        
        elif select_mode == 'prompt_topk':
            # 1. Get raw features from vision tower (includes [CLS] token)
            raw_image_features = self.vision_tower(pixel_values, patch_coords, output_mode='all') if patch_coords is not None else self.vision_tower(pixel_values, output_mode='all')

            # 2. Separate [CLS] token and patch features
            cls_token_feature = raw_image_features[:, 0:1, :]
            patch_features = raw_image_features[:, 1:, :]

            # 3. Generate query vector from prompt embeddings
            if inputs_embeds is not None:
                batch_size = inputs_embeds.shape[0]
                sampled_patch_features_list = []
                
                # Process each sample in the batch individually
                for i in range(batch_size):
                    # Find the prompt tokens for the current sample
                    if labels is not None:
                        # Training: Use labels to find the prompt by ignoring answer tokens
                        prompt_mask = (labels[i] == IGNORE_INDEX)
                        prompt_embeds = inputs_embeds[i][prompt_mask]
                    else:
                        # Inference: No labels available, use the entire input sequence as the prompt
                        prompt_embeds = inputs_embeds[i]

                    # If there are no prompt tokens, fallback to random sampling for this sample
                    if prompt_embeds.shape[0] == 0:
                        target_len_fallback = getattr(self.config, "image_seq_length", 256)
                        num_patches_fallback = patch_features[i].shape[0]
                        if num_patches_fallback > target_len_fallback:
                            indices = torch.randperm(num_patches_fallback, device=patch_features.device)[:target_len_fallback]
                            sampled_patch_features_list.append(patch_features[i, indices, :])
                        else:
                            pad_len = target_len_fallback - num_patches_fallback
                            padding = torch.zeros(pad_len, patch_features.shape[-1], dtype=patch_features.dtype, device=patch_features.device)
                            sampled_patch_features_list.append(torch.cat([patch_features[i], padding], dim=0))
                        continue

                    # Use mean pooling to get the query vector
                    query_vector = prompt_embeds.mean(dim=0, keepdim=True).unsqueeze(0)
                    
                    # Project the query vector to match visual feature dimensions
                    projected_query_vector = self.text_query_projection(query_vector)
                    
                    # Normalize for cosine similarity
                    query_vector_norm = nn.functional.normalize(projected_query_vector, p=2, dim=2)
                    patch_features_norm = nn.functional.normalize(patch_features[i].unsqueeze(0), p=2, dim=2)

                    # Calculate cosine similarity
                    similarity_scores = torch.bmm(patch_features_norm, query_vector_norm.transpose(1, 2)).squeeze()

                    # Select Top-K features and their scores to maintain differentiability
                    target_len = getattr(self.config, "image_seq_length", 256)
                    k = min(target_len, patch_features[i].shape[0])
                    topk_values, topk_indices = torch.topk(similarity_scores, k=k, dim=0)
                    
                    # Gather the top-k features
                    sampled_patch = torch.gather(patch_features[i], 0, topk_indices.unsqueeze(-1).expand(-1, patch_features.shape[-1]))
                    
                    # Apply rescaling for differentiable selection during training
                    # This rescaling trick is essential for making the selection differentiable during TRAINING.
                    # During INFERENCE, we want to pass the original, full-magnitude features
                    # to the LLM. Attenuating them degrades the information quality.
                    if self.training:
                        sampled_patch = sampled_patch * torch.sigmoid(topk_values).unsqueeze(-1)

                    # Pad if necessary
                    if k < target_len:
                        pad_len = target_len - k
                        padding = torch.zeros(pad_len, patch_features.shape[-1], dtype=patch_features.dtype, device=patch_features.device)
                        sampled_patch = torch.cat([sampled_patch, padding], dim=0)

                    sampled_patch_features_list.append(sampled_patch)

                sampled_patch_features = torch.stack(sampled_patch_features_list, dim=0)

            else:
                # Fallback to random sampling only if inputs_embeds is not available (edge case)
                target_len = getattr(self.config, "image_seq_length", 256)
                num_patches = patch_features.shape[1]
                if num_patches > target_len:
                    indices = torch.randperm(num_patches, device=patch_features.device)[:target_len]
                    sampled_patch_features = patch_features[:, indices, :]
                else: # Pad if necessary
                    pad_len = target_len - num_patches
                    padding = torch.zeros(patch_features.shape[0], pad_len, patch_features.shape[2],
                                          dtype=patch_features.dtype, device=patch_features.device)
                    sampled_patch_features = torch.cat([patch_features, padding], dim=1)

            # Concatenate [CLS] token back
            image_features = torch.cat([cls_token_feature, sampled_patch_features], dim=1)

        elif select_mode == 'random_sample':
            # 1. Get raw features from vision tower (includes [CLS] token)
            raw_image_features = self.vision_tower(pixel_values, patch_coords, output_mode='all') if patch_coords is not None else self.vision_tower(pixel_values, output_mode='all')

            # 2. Separate [CLS] token from patch features
            cls_token_feature = raw_image_features[:, 0:1, :]
            patch_features = raw_image_features[:, 1:, :]

            # 3. Determine target number of patch features
            target_len = getattr(self.config, "image_seq_length", 2048) 
            
            num_patches = patch_features.shape[1]

            # 4. Sample or pad patch features
            if num_patches > target_len:
                indices = torch.randperm(num_patches, device=patch_features.device)[:target_len]
                sampled_patch_features = patch_features[:, indices, :]
            elif num_patches < target_len:
                pad_len = target_len - num_patches
                padding = torch.zeros(patch_features.shape[0], pad_len, patch_features.shape[2],
                                      dtype=patch_features.dtype, device=patch_features.device)
                sampled_patch_features = torch.cat([patch_features, padding], dim=1)
            else:
                sampled_patch_features = patch_features

            # 5. Concatenate [CLS] token back
            image_features = torch.cat([cls_token_feature, sampled_patch_features], dim=1)
        
        else:  # 'none' or other modes
            # Get raw features from vision tower without any selection
            image_features = self.vision_tower(pixel_values, patch_coords, output_mode='all') if patch_coords is not None else self.vision_tower(pixel_values, output_mode='all')

        # Apply multimodal projection (common to all paths)
        image_features = self.multi_modal_projector(image_features)
        return image_features

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        patch_coords: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_sizes: torch.Tensor = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, CustomLlavaModelOutputWithPast]:
        # Sanity/eval hook: allow upstream to request zeroing image features after projection
        force_zero_image_features: bool = kwargs.pop("force_zero_image_features", False)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )


        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        
        inputs_embeds = inputs_embeds.clone()

        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values,
                patch_coords=patch_coords,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                image_sizes=image_sizes,
                inputs_embeds=inputs_embeds,
                labels=labels,
            ) # (batch_size, image_seq_length, embed_dim)
            
            """if torch.distributed.get_rank() == 0:
                print(f"Shape of image_features in forward: {image_features.shape}")"""
            # If requested, zero out the final image features so that placeholders are kept
            # but no visual information is injected into LLM.
            if force_zero_image_features:
                image_features = torch.zeros_like(image_features)

            # Find image token positions in input_ids
            if input_ids is not None:
                image_token_mask = (input_ids == self.config.image_token_index)
                n_image_tokens = image_token_mask.sum()

                # print(f"n_image_tokens: {n_image_tokens}")
                
                batch_size = input_ids.shape[0]
                
                # Check if the number of image features matches the number of image tokens
                num_patches = image_features.shape[1]
                
                # Replace image token embeddings with image features
                for i in range(batch_size):
                    # Find the indices of all image tokens for the current sample
                    image_token_indices = torch.where(input_ids[i] == self.config.image_token_index)[0]
                    num_image_tokens_in_sample = image_token_indices.numel()

                    if num_image_tokens_in_sample > 0:
                        # Ensure the number of patches matches the placeholder tokens
                        if num_patches != num_image_tokens_in_sample:
                            # This could be a sanity check or an error
                            # For now, let's assume they match or we can truncate/pad
                            # Here, we'll assume they match for simplicity as per design
                            pass

                        start_index = image_token_indices[0]
                        end_index = start_index + num_patches
                        
                        # Replace the placeholder embeddings with the actual image features
                        # print(f"Replacing image features at indices {start_index}:{end_index} with shape {image_features[i].shape}")
                        inputs_embeds[i, start_index:end_index] = image_features[i]
                    

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        return CustomLlavaModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class CustomLlavaForConditionalGeneration(CustomLlavaPreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: CustomLLaVAModelConfig, **kwargs):
        super().__init__(config)
        self.model = CustomLlavaModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_config(cls, config: CustomLLaVAModelConfig, **kwargs):
        """
        Creates a new instance of CustomLlavaForConditionalGeneration from a configuration.
        
        Args:
            config (CustomLLaVAModelConfig): The model configuration
            **kwargs: Additional keyword arguments passed to the model initialization
            
        Returns:
            CustomLlavaForConditionalGeneration: A new instance of the model
        """
        return cls(config=config, **kwargs)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        patch_coords: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, List[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CustomLlavaCausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            patch_coords=patch_coords,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,  # Pass labels to the model
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            image_sizes=image_sizes,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits for efficiency
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten tokens for loss computation
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.text_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return CustomLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        patch_coords=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        **kwargs,
    ):
        # Override to handle image inputs appropriately during generation

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        if cache_position[0] == 0:
            # During cached decoding, pixel values should be None since input ids don't contain image tokens
            # Otherwise, we need pixel values to be passed to the model
            model_inputs["pixel_values"] = pixel_values
            # Also pass patch_coords for WSI models
            if patch_coords is not None:
                model_inputs["patch_coords"] = patch_coords

        return model_inputs

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


# Register the custom model configuration and model class
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("custom_llava", CustomLLaVAModelConfig)
AutoModelForCausalLM.register(CustomLLaVAModelConfig, CustomLlavaForConditionalGeneration)