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

from llamafactory.train.tuner import run_exp
import yaml
import os
import sys
import argparse


def main():
 
    parser = argparse.ArgumentParser(description="Train LLaVA model")
    parser.add_argument("--config", type=str, default="custom_llava_gigapath_lora_sft.yaml", 
                       help="Configuration file name (default: custom_llava_gigapath_lora_sft.yaml)")
    parser.add_argument("--stage", type=str, choices=["stage2", "stage3"], 
                       help="Training stage (stage2: projector training, stage3: full training)")
    
    args = parser.parse_args()
    

    if args.stage == "stage2":
        config_folder = "train_full"
        config_file = "stage2_projector_full_training.yaml"
        print("🎯 Stage 2: Training projector with WSI-Caption pairs")
    elif args.stage == "stage3":
        config_folder = 'train_lora'
        config_file = "stage3_vqa_lora_sft.yaml"  # Stage3配置文件
        print("🎯 Stage 3: Training projector + LLM with VQA pairs")
    else:
        config_folder = 'train_lora'
        config_file = args.config
        print(f"🎯 Using custom config: {config_file}")
    
    # 构建配置文件路径
    config_path = os.path.join(
        os.path.dirname(__file__), 
        "..", 
        "examples", 
        config_folder, 
        config_file
    )
    config_path = os.path.abspath(config_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    print(f"📁 Loading config: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 显示配置信息
    print(f"📊 Dataset: {config.get('dataset')}")
    print(f"🏷️  Model: {config.get('model_name')}")
    print(f"📈 Epochs: {config.get('num_train_epochs')}")
    print(f"💾 Output: {config.get('output_dir')}")
    print("-" * 50)
    
    run_exp(config)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    run_exp()


if __name__ == "__main__":
    main()
