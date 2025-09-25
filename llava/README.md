# WSI-LLaVA: Whole Slide Image Large Language and Vision Assistant

This repository contains a sanitized version of our WSI-LLaVA implementation for whole slide image analysis using multimodal large language models. The code has been cleaned for public release while preserving the core architecture and innovations.

## Architecture Overview

WSI-LLaVA extends the LLaVA framework to handle whole slide images (WSIs) in pathology:

- **Vision Tower**: GigaPath-based encoder for WSI patch feature extraction
- **Multi-modal Projector**: Custom projector with gamma scaling for feature alignment
- **Language Model**: LLaMA-3.2-1B-Instruct for text generation
- **Visual Token Selection**: PruMerge and Prompt-based Top-K strategies
- **Dynamic H5 Loading**: Efficient processing of large WSI datasets

## 📁 Repository Structure

```
llava/
├── src/llamafactory/
│   ├── model/custom/
│   │   ├── llava.py                    # Custom LLaVA model implementation
│   │   └── prumerge.py                 # PruMerge visual token selection
│   └── data/
│       ├── unpuzzle_dataset.py         # WSI dataset loader (sanitized)
│       └── h5_collator.py              # Dynamic H5 data collator
│
├── examples/
│   ├── train_full/
│   │   └── stage2_projector_training.yaml    # Stage 2 training config
│   └── train_lora/
│       └── stage3_vqa_training.yaml          # Stage 3 training config
│
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch torchvision
pip install transformers llama-factory
pip install pandas h5py
pip install wandb  # Optional for logging
```

### 2. Data Preparation

Prepare your data in the following format:

**CSV File Structure:**
```csv
h5_path,caption,question,answer,slide_id,dataset
sample1.h5,This is a breast cancer tissue sample,,,slide_001,PathText
sample2.h5,Pathology report shows malignant cells,What is the diagnosis?,Malignant,slide_002,VQA
```

**H5 File Structure:**
- `features`: [N, 1536] array of patch features (GigaPath output)
- `coords_yx`: [N, 2] array of patch coordinates

### 3. Configuration

Update the configuration files:

**Stage 2 (Projector Training):**
```yaml
# Update examples/train_full/stage2_projector_training.yaml
dataset_dir: /path/to/your/dataset
vision_tower: /path/to/visual_tower.pt
```

**Stage 3 (VQA Training):**
```yaml
# Update examples/train_lora/stage3_vqa_training.yaml
dataset_dir: /path/to/your/dataset
projector_path: /path/to/stage2/checkpoint/mm_projector.bin
vision_tower: /path/to/visual_tower.pt
```

### 4. Training

```bash
# Stage 2: Projector Training
llamafactory-cli train examples/train_full/stage2_projector_training.yaml

# Stage 3: VQA Training
llamafactory-cli train examples/train_lora/stage3_vqa_training.yaml
```

## 🔧 Key Features

### Custom Multi-modal Projector

Our custom projector includes gamma scaling for better feature alignment:

```python
# Bounded gamma scaling for stable training
gamma = self.gamma_min + torch.sigmoid(self.gamma_logit) * (self.gamma_max - self.gamma_min)
hidden_states = gamma * hidden_states
```

### Visual Token Selection Strategies

1. **Random Sampling**: Randomly select patches for processing
2. **PruMerge**: CLS-attention guided pruning and merging
3. **Prompt-based Top-K**: Select tokens based on text query similarity

### Dynamic H5 Loading

Efficient processing of large WSI datasets:

- **Lazy Loading**: Load H5 features only when needed
- **Patch Sampling**: Configurable maximum patches per WSI
- **Memory Efficient**: Avoid loading all features into memory

### Two-Stage Training

**Stage 2: Projector Training**
- Freeze vision tower and language model
- Train only the multi-modal projector
- Focus on aligning visual and text features

**Stage 3: VQA Training**
- Inject trained projector from Stage 2
- LoRA fine-tuning of language model
- Instruction tuning for VQA tasks

## 📊 Model Configurations

### Available Models

- `custom_llava`: WSI-LLaVA with GigaPath backbone
- Supports both ROI and WSI vision modes

### Configuration Options

```yaml
# Vision configuration
vision_config:
  vision_mode: wsi
  model_name: gigapath
  hidden_size: 768
  patches_per_wsi: 2048

# Visual token selection
vision_feature_select_mode: random_sample  # or prumerge, prompt_topk
image_seq_length: 1536

# Projector configuration
mm_projector_type: mlp2x_gelu
gamma_min: 1.2
gamma_max: 1.8
```

## 🔬 Research Applications

This implementation has been used for:

- **Pathology VQA**: Question answering on whole slide images
- **Medical Report Generation**: Automated pathology report writing
- **Multimodal Medical AI**: Vision-language models for medical imaging
- **Visual Token Selection**: Efficient processing of large medical images

## ⚠️ Important Notes

### Data Privacy

This is a **sanitized version** for public release:

- All internal data paths have been removed
- Hardcoded paths replaced with placeholders
- Sensitive information removed from comments
- Example configurations provided instead of actual data

### Dependencies

Some components require additional setup:

- **GigaPath Model**: Requires proper model weights and implementation
- **LLaMA Factory**: Full LLaMA Factory installation required
- **H5 Files**: WSI features in the specified format
- **Vision Tower**: Pre-trained visual encoder from Stage 1

### Limitations

- WSI model loading requires internal implementation
- Some advanced features may need additional dependencies
- Dataset loading assumes specific H5 and CSV formats

## 🤝 Contributing

This repository is provided for research and educational purposes. For questions or contributions:

1. Check existing issues and documentation
2. Follow the code style and structure
3. Ensure all sensitive information is removed
4. Test with synthetic data before submission

## 📄 License

This project is released for research purposes. Please cite our work if you use this code:

```bibtex
@article{your_paper_2024,
  title={WSI-LLaVA: Whole Slide Image Large Language and Vision Assistant},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## Acknowledgments

- LLaMA Factory team for the base framework
- LLaVA team for the multimodal architecture
- GigaPath team for the vision encoder
- Medical imaging research community

---

**Note**: This is a sanitized version of our internal research code. Some functionality may require additional setup or implementation details not included in this public release.