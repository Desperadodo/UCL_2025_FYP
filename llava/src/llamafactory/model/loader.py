# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead
import torch.nn as nn

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from ..extras.packages import is_transformers_version_greater_than
from .adapter import init_adapter
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model
from .custom.llava import CustomLLaVAModelConfig, CustomLlavaForConditionalGeneration
from .custom.mm_utils import DummyProcessor

if is_transformers_version_greater_than("4.46.0"):
    from transformers import AutoModelForImageTextToText


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments", model_name_or_path: str = None, custom: bool = False) -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    if model_name_or_path and model_args.llm_name_or_path:
        model_args.llm_name_or_path = try_download_model_from_other_hub(model_args,model_name_or_path=model_name_or_path)
    else:
        model_args.model_name_or_path = try_download_model_from_other_hub(model_args, custom=custom)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    if model_args.llm_name_or_path:
        model_args.llm_name_or_path = try_download_model_from_other_hub(model_args,model_name_or_path=model_args.llm_name_or_path)
        init_kwargs = _get_init_kwargs(model_args,model_name_or_path=model_args.llm_name_or_path)
    else:
        model_args.model_name_or_path = try_download_model_from_other_hub(model_args,model_name_or_path=model_args.model_name_or_path)
        init_kwargs = _get_init_kwargs(model_args,model_name_or_path=model_args.model_name_or_path)
    tokenizer_path = model_args.llm_name_or_path if model_args.llm_name_or_path else model_args.model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try the fast one
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)
    
    if model_args.llm_name_or_path:
        from .custom.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        
        # 添加 <image> token (这是数据中实际使用的token)
        tokenizer.add_tokens([DEFAULT_IMAGE_TOKEN], special_tokens=True)
        actual_image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        logger.info_rank0(f"Added {DEFAULT_IMAGE_TOKEN} token to tokenizer with ID: {actual_image_token_id}")
        
        # 更新model_args中的image_token_index为实际的token ID
        model_args.image_token_index = actual_image_token_id
        logger.info_rank0(f"Updated image_token_index to: {actual_image_token_id}")
        
        # 可选：根据配置添加其他tokens
        if getattr(model_args, "mm_use_im_patch_token", False):
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            logger.info_rank0(f"Added {DEFAULT_IMAGE_PATCH_TOKEN} token to tokenizer")
        if getattr(model_args, "mm_use_im_start_end", False):
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            logger.info_rank0(f"Added {DEFAULT_IM_START_TOKEN} and {DEFAULT_IM_END_TOKEN} tokens to tokenizer")
        
        if model_args.processor_name_or_path:
            processor_path = model_args.processor_name_or_path
            logger.info_rank0(f"Using processor from: {processor_path}")
        else:
            processor_path = model_args.model_name_or_path
        try:
            processor = AutoProcessor.from_pretrained(processor_path, **init_kwargs)
            patch_processor(processor, tokenizer, model_args)
        except Exception as e:
            logger.info_rank0(f"Failed to load processor: {e}.")
            processor = None

    else:
        if model_args.processor_name_or_path:
            processor_path = model_args.processor_name_or_path
            logger.info_rank0(f"Using processor from: {processor_path}")
        else:
            processor_path = model_args.model_name_or_path
        try:
            processor = AutoProcessor.from_pretrained(processor_path, **init_kwargs)
            patch_processor(processor, tokenizer, model_args)
        except Exception as e:
            logger.info_rank0(f"Failed to load processor: {e}.")
            processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments", custom: bool = False) -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args, custom=custom)
    
    # Handle custom LLaVA model
    if model_args.model_name_or_path == "custom_llava":
        from .custom.llava import CustomLLaVAModelConfig, CustomVisionConfig
        
        # Create text config (LLM config)
        text_config = AutoConfig.from_pretrained(
            model_args.llm_name_or_path or "meta-llama/Llama-2-7b-hf",
            trust_remote_code=model_args.trust_remote_code,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.hf_hub_token
        )
        
        # Create vision config using CustomVisionConfig
        vision_config = CustomVisionConfig(
            model_name=getattr(model_args, "vision_model_name", "gigapath"),
            local_weight_path=model_args.vision_tower,
            vision_mode=getattr(model_args, "vision_mode", "roi"),
            ROI_feature_dim=getattr(model_args, "ROI_feature_dim", None),
            slide_embed_dim=getattr(model_args, "slide_embed_dim", 768),
            MTL_token_num=getattr(model_args, "MTL_token_num", 0),
            heatmap=getattr(model_args, "heatmap", False),
            patches_per_wsi=getattr(model_args, "patches_per_wsi", 1024),
            hidden_size=getattr(model_args, "mm_hidden_size", 768),
        )
        
        # Get all LLaVA-specific arguments
        llava_kwargs = {
            "text_config": text_config,
            "vision_config": vision_config,
            "image_token_index": getattr(model_args, "image_token_index", 32000),
            "projector_hidden_act": getattr(model_args, "projector_hidden_act", "gelu"),
            "vision_feature_select_strategy": getattr(model_args, "vision_feature_select_strategy", "default"),
            "vision_feature_select_mode": getattr(model_args, "vision_feature_select_mode", "prumerge"),
            "vision_feature_layer": getattr(model_args, "vision_feature_layer", -2),
            "image_seq_length": getattr(model_args, "image_seq_length", 576),
            "multimodal_projector_bias": getattr(model_args, "multimodal_projector_bias", True),
            "projector_dropout": getattr(model_args, "projector_dropout", 0.1),
            "gamma_init": getattr(model_args, "gamma_init", 2.0),
            "gamma_min": getattr(model_args, "gamma_min", 0.8),
            "gamma_max": getattr(model_args, "gamma_max", 2.0),
            "llm_name_or_path": model_args.llm_name_or_path,
            "train_from_scratch": getattr(model_args, "train_from_scratch", False)
        }
        
        config = CustomLLaVAModelConfig(**llava_kwargs)
    else:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)
    
    return config


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args, custom=True)
    
    config = load_config(model_args, custom=True)
    if getattr(model_args, "image_seq_length", None):
        config.image_seq_length = model_args.image_seq_length

    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args)
            
    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            # --- START OF CRITICAL FIX ---
            # Prioritize loading our custom model architecture whenever the config matches.
            # This ensures the correct model structure (with projector) is built *before* any PEFT/LoRA logic is applied.
            if isinstance(config, CustomLLaVAModelConfig):
                logger.info_rank0("Initializing custom LLaVA model from config.")
                model = CustomLlavaForConditionalGeneration(config)
                load_class = None  # Explicitly prevent falling through to the generic loaders
            # --- END OF CRITICAL FIX ---
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
            elif (
                is_transformers_version_greater_than("4.46.0")
                and type(config) in AutoModelForImageTextToText._model_mapping.keys()
            ):  # image-text
                load_class = AutoModelForImageTextToText
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen2_5_omni
                load_class = AutoModelForTextToWaveform
            else:
                load_class = AutoModelForCausalLM

            if model is None:  # Only execute if our custom model hasn't been initialized
                if model_args.train_from_scratch:
                    model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
                else:
                    try:
                        model = load_class.from_pretrained(**init_kwargs)
                        if getattr(model.config, "model_type", None) == "qwen2_5_omni":
                            model = model.thinker  # use part of Omni model
                    except Exception as e:
                        logger.error_rank0(f"Failed to load model: {str(e)}")
                        if "model type" in str(e).lower() and "transformers" in str(e).lower():
                            logger.error_rank0(
                                "Please ensure you have the latest transformers version: "
                                "pip install --upgrade transformers or "
                                "pip install git+https://github.com/huggingface/transformers.git"
                            )
                        raise e

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)
        
    # --- START OF MODIFIED PROJECTOR LOADING ---
    # We inject the weights *before* PEFT/LoRA adapters are initialized.
    # The model object at this point should be our complete CustomLlavaForConditionalGeneration instance.
    if model_args.projector_path:
        logger.info(f"Attempting to load pretrained projector weights from {model_args.projector_path}")
        # Use a more direct and robust way to access the projector
        if isinstance(model, CustomLlavaForConditionalGeneration) and hasattr(model.model, 'multi_modal_projector'):
            projector_weights = torch.load(model_args.projector_path, map_location="cpu")
            model.model.multi_modal_projector.load_state_dict(projector_weights, strict=True)
            logger.info("Successfully loaded projector weights into the model.")
        else:
            logger.warning("Could not load projector weights. The model is not a CustomLlavaForConditionalGeneration instance or is missing the projector attribute.")
    # --- END OF MODIFIED PROJECTOR LOADING ---

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
