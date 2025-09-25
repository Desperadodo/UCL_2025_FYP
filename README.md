# WSI-CLIP & WSI-LLaVA: Whole Slide Image Multimodal Learning

This repository contains sanitized implementations of our WSI-CLIP and WSI-LLaVA systems for whole slide image analysis in pathology. The code has been cleaned for public release while preserving core innovations and architectural designs.

## 🏗️ System Overview

Our system consists of two main components:

1. **WSI-CLIP**: Contrastive learning framework for WSI representation learning
2. **WSI-LLaVA**: Multimodal large language model for WSI question answering

## 📁 Repository Structure

```
fyp_code/
├── clip/                           # WSI-CLIP Training Framework
│   ├── factory.py                  # Model factory and configuration
│   ├── loss.py                     # Multi-positive CLIP loss
│   ├── model.py                    # WSI_CLIP model architecture
│   ├── custom_model_builder.py     # Custom model builders
│   ├── get_wsi_model.py           # Sanitized WSI model loader
│   └── model_configs/             # Model configurations
│       └── customize-clip-wsi-gigapath.json
│
├── open_clip_train/               # CLIP Training Framework
│   ├── main.py                    # Main training entry point
│   ├── params.py                  # Parameter definitions
│   ├── train.py                   # Training logic
│   ├── distributed.py             # Distributed training utilities
│   ├── scheduler.py               # Learning rate schedulers
│   ├── script/                    # Training scripts
│   │   └── train_clip_wsi.sh      # Example training script
│   └── src/                       # Source code
│       └── data/
│           └── unpuzzle_dataset.py # WSI dataset implementation
│
├── llava/                         # WSI-LLaVA Framework
│   ├── src/llamafactory/
│   │   ├── model/custom/
│   │   │   ├── llava.py           # Custom LLaVA model
│   │   │   └── prumerge.py        # PruMerge token selection
│   │   └── data/
│   │       ├── unpuzzle_dataset.py # WSI dataset loader
│   │       └── h5_collator.py     # Dynamic H5 collator
│   └── examples/                  # Training configurations
│       ├── train_full/
│       │   └── stage2_projector_training.yaml
│       └── train_lora/
│           └── stage3_vqa_training.yaml
│
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install torch torchvision
pip install transformers timm llama-factory
pip install pandas h5py
pip install wandb  # Optional for logging
```

### 2. WSI-CLIP Training

```bash
cd clip
# Update paths in open_clip_train/script/train_clip_wsi.sh
bash open_clip_train/script/train_clip_wsi.sh
```

### 3. WSI-LLaVA Training

```bash
cd llava
# Update paths in examples/train_full/stage2_projector_training.yaml
llamafactory-cli train examples/train_full/stage2_projector_training.yaml

# Then update paths in examples/train_lora/stage3_vqa_training.yaml
llamafactory-cli train examples/train_lora/stage3_vqa_training.yaml
```

## 🔧 Key Innovations

### WSI-CLIP Innovations

1. **Multi-Positive CLIP Loss**: Handles multiple images per case in medical datasets
2. **GigaPath Integration**: Efficient WSI feature extraction using dilated attention
3. **Dynamic Patch Sampling**: Memory-efficient processing of large WSIs

### WSI-LLaVA Innovations

1. **Custom Multi-modal Projector**: Gamma scaling for stable feature alignment
2. **Visual Token Selection**: PruMerge and Prompt-based Top-K strategies
3. **Dynamic H5 Loading**: Efficient processing without memory bottlenecks
4. **Two-Stage Training**: Projector training followed by LoRA fine-tuning

## 📊 Data Format Requirements

### H5 Files (WSI Features)
```
features: [N, 1536] array of patch features (GigaPath output)
coords_yx: [N, 2] array of patch coordinates
```

### CSV Files (Annotations)
```csv
h5_path,caption,question,answer,slide_id,dataset
sample1.h5,This is a breast cancer tissue sample,,,slide_001,PathText
sample2.h5,Pathology report shows malignant cells,What is the diagnosis?,Malignant,slide_002,VQA
```

## 🔬 Research Applications

This implementation has been used for:

- **Pathology Image Analysis**: Whole slide image classification and retrieval
- **Medical VQA**: Question answering on pathological images
- **Report Generation**: Automated pathology report writing
- **Multimodal Medical AI**: Vision-language models for medical imaging

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

## 📚 Documentation

- [CLIP Training Guide](clip/README.md) - WSI-CLIP training documentation
- [LLaVA Training Guide](llava/README.md) - WSI-LLaVA training documentation
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Complete deployment instructions

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
  title={WSI-CLIP and WSI-LLaVA: Multimodal Learning for Whole Slide Images},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 🙏 Acknowledgments

- GigaPath team for the vision encoder
- LLaMA Factory team for the base framework
- LLaVA team for the multimodal architecture
- OpenCLIP community for the base framework
- Medical imaging research community

---

**Note**: This is a sanitized version of our internal research code. Some functionality may require additional setup or implementation details not included in this public release.