import os
import pandas as pd
import h5py
import torch
import numpy as np
import random
import ast
import string # Import the string module
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from datasets import Dataset
from transformers import PreTrainedTokenizer
from PIL import Image

from ..extras import logging
from .converter import DatasetConverter
from .data_utils import Role

if TYPE_CHECKING:
    from ..hparams import DataArguments
    from .parser import DatasetAttr

logger = logging.get_logger(__name__)

# 🔧 Stage2简化：移除复杂的问题模板，专注于简单的image-caption对应
# 参考LLaVA官方Stage2使用CC-3M的简洁格式

# 🎯 Stage3 VQA问句模板 - 用于将caption转换为问答对
VQA_QUESTION_TEMPLATES = [
    "Please provide a comprehensive pathology report for this whole-slide image.",
    "Could you summarise the key microscopic findings shown here?",
    "Give a detailed diagnostic interpretation of the WSI.",
    "What is the formal pathologic diagnosis for this slide?",
    "Describe all notable histologic features present.",
    "Compose the full surgical pathology report based on this specimen.",
    "List the main tumour characteristics observed in the image.",
    "Provide a narrative pathologist's report for this case.",
    "Write the pathology impression section for this WSI.",
    "What conclusions should a pathologist draw from this slide?",
    "Draft the diagnostic comment you would add to the medical record.",
    "Summarise the lesion type, grade and any special findings visible.",
    "Give a concise but complete pathology synopsis.",
    "Create the final diagnostic statement for this whole-slide image.",
    "Please render the pathology diagnosis in standard terminology.",
    "Provide the histopathologic report as if for tumour board review.",
    "Offer a detailed description of tumour morphology and behaviour.",
    "Enumerate all relevant prognostic features seen in the WSI.",
    "Present the full pathologic assessment including margins and stage.",
    "Write the 'Microscopic Findings' section for this specimen.",
    "Summarise any invasive components and associated in-situ changes.",
    "State the tumour subtype and any ancillary findings.",
    "Provide a thorough histological evaluation suitable for the lab report.",
    "Compose an expert pathology note describing everything of importance.",
    "What should the attending pathologist report about this tissue?",
    "Deliver a structured pathology diagnosis for this case.",
    "Give an expert interpretation of the observed pathology.",
    "Summarise this WSI in the style of a formal pathology report.",
    "Please generate the diagnostic comment for this breast specimen.",
    "Describe the tumour and any associated features as seen under the microscope."
]

def make_vqa_question(slide_id: str, random_seed: Optional[int] = None) -> str:
    """
    为给定的slide_id生成一个VQA问句。
    使用slide_id作为随机种子确保同一个slide总是得到相同的问句。
    """
    if random_seed is None:
        # 使用slide_id的hash作为随机种子，确保可重现性
        random_seed = hash(slide_id) % (2**32)
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 随机选择一个模板
    selected_template = random.choice(VQA_QUESTION_TEMPLATES)
    
    # 重置随机种子避免影响其他随机操作
    random.seed(None)
    
    return selected_template

def get_unpuzzle_dataset(
    dataset_name: str,
    dataset_dir: str,
    tokenizer: "PreTrainedTokenizer",
    **kwargs
) -> Dataset:
    """Load and process the Unpuzzle WSI dataset."""
    try:
        # Get dataset attributes
        dataset_attr = kwargs.get("dataset_attr")
        data_args = kwargs.get("data_args")
        
        # 🔧 标准模式：根据dataset_name确定数据文件
        if dataset_name in ["pathtext_stage2_train", "pathtext_stage2_val"]:
            # 新的标准模式：直接从数据集名称确定文件
            if dataset_name == "pathtext_stage2_train":
                csv_filename = "PathText-BRCA-unpuzzle-train.csv"
                split_name = "训练集"
            else:  # pathtext_stage2_val
                csv_filename = "PathText-BRCA-unpuzzle-val.csv"
                split_name = "验证集"
            
            csv_path = f"/data/private_hdd/PathText-unpuzzle/{csv_filename}"
            logger.info(f"📊 使用 {split_name}: {csv_path}")
        elif dataset_name == "pathtext_stage2":
            # 兼容旧的方式（如果还有地方使用）
            split = dataset_attr.split if dataset_attr else 'train'
            csv_path = f"/data/private_hdd/PathText-unpuzzle/PathText-BRCA-unpuzzle-{split}.csv"
            logger.info(f"📊 兼容模式 - 使用 {split} 数据集: {csv_path}")
        else:
            # Construct CSV path for other datasets
            csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read CSV file - handle both comma and tab separators
        try:
            df = pd.read_csv(csv_path, sep=",", keep_default_na=False)  # Try comma first
        except:
            df = pd.read_csv(csv_path, sep="\t", keep_default_na=False)  # Fallback to tab
        
        # 关键修正：将所有NaN值替换为空字符串，以防止pyarrow在处理混合类型（str和float）时出错。
        df.fillna("", inplace=True)

        logger.info(f"Loaded {len(df)} samples from {csv_path}")
        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        # 🔍 如果使用预先划分的数据，记录划分信息
        if 'split' in df.columns:
            split_counts = df['split'].value_counts()
            logger.info(f"📊 数据划分统计: {split_counts.to_dict()}")
            # 只使用训练数据
            if 'train' in split_counts:
                df = df[df['split'] == 'train'].copy()
                logger.info(f"🎯 使用训练集数据: {len(df)} 个样本")
        
        # Convert DataFrame to list of dicts
        data = []
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        
        for idx, row in df.iterrows():
            # Handle different CSV formats
            if "filepath" in df.columns and "title" in df.columns:
                # PathText-unpuzzle format: filepath, subset, title
                h5_path = os.path.join("/data/private_hdd/PathText-unpuzzle", row["filepath"])
                caption = row["title"]
                question = ""
                answer = ""
                slide_id = os.path.basename(row["filepath"]).replace(".h5", "")
                dataset_name_field = row.get("subset", "PathText")
            else:
                # Standard format: h5_path, question, answer, caption, etc.
                h5_path = row.get("h5_path", "")
                if not os.path.isabs(h5_path):
                    h5_path = os.path.join(csv_dir, h5_path)
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()
                caption = row.get("caption", "")
                slide_id = row.get("slide_id", "")
                dataset_name_field = row.get("dataset", "")
            
            data.append({
                "h5_path": h5_path,
                "question": question,
                "answer": answer,
                "caption": caption,
                "slide_id": slide_id,
                "dataset": dataset_name_field,
            })
            
            # Log first few samples for debugging
            if idx < 3:
                logger.info(f"Sample {idx}: h5_path={h5_path}, caption_length={len(caption)}")
        
        # Create dataset
        dataset = Dataset.from_list(data)
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # 🔧 使用新的动态加载转换器
        converter = DynamicUnpuzzleDatasetConverter(
            dataset_attr=dataset_attr,
            data_args=data_args
        )
        return converter.convert(dataset, tokenizer)
        
    except Exception as e:
        logger.error(f"Error loading Unpuzzle dataset: {str(e)}")
        raise

def get_unpuzzle_vqa_dataset(
    dataset_name: str,
    dataset_dir: str,
    tokenizer: "PreTrainedTokenizer",
    **kwargs
) -> Dataset:
    """Load and process the Unpuzzle WSI dataset for VQA (Stage 3) training."""
    try:
        # Get dataset attributes
        dataset_attr = kwargs.get("dataset_attr")
        data_args = kwargs.get("data_args")
        
        # Stage3 VQA mode
        if dataset_name in ["pathtext_wsivqa_stage3_train", "pathtext_wsivqa_stage3_val", "pathtext_wsivqa_stage3_test"]:
            # 新的统一VQA数据集模式
            split = dataset_name.split('_')[-1]  # train, val, or test
            csv_filename = f"PathText-WSIVQA-BRCA-unpuzzle-{split}.csv"
            csv_path = f"/data/private_hdd/PathText-unpuzzle/{csv_filename}"
            split_name = f"Unified WSI-VQA {split} set"
            logger.info(f"{split_name}: {csv_path}")
        else:
            # Construct CSV path for other datasets
            csv_path = os.path.join(dataset_dir, f"{dataset_name}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Read CSV file - handle both comma and tab separators
        try:
            df = pd.read_csv(csv_path, sep=",", keep_default_na=False)  # Try comma first
        except:
            df = pd.read_csv(csv_path, sep="\t", keep_default_na=False)  # Fallback to tab
        
        # 关键修正：将所有NaN值替换为空字符串，以防止pyarrow在处理混合类型（str和float）时出错。
        df.fillna("", inplace=True)

        logger.info(f"Loaded {len(df)} samples from {csv_path}")

        # 🆕 添加数据源过滤逻辑
        if data_args and data_args.filter_by_data_source:
            logger.info(f"Filtering dataset by data_source: '{data_args.filter_by_data_source}'")
            original_count = len(df)
            df = df[df["data_source"] == data_args.filter_by_data_source]
            logger.info(f"Filtered {original_count} -> {len(df)} samples.")
            if len(df) == 0:
                logger.warning("Warning: The dataset is empty after filtering.")

        logger.info(f"CSV columns: {df.columns.tolist()}")
        
        # 🔍 如果使用预先划分的数据，记录划分信息
        if 'split' in df.columns:
            split_counts = df['split'].value_counts()
            logger.info(f"📊 数据划分统计: {split_counts.to_dict()}")
            # 只使用训练数据
            if 'train' in split_counts:
                df = df[df['split'] == 'train'].copy()
                logger.info(f"🎯 使用训练集数据: {len(df)} 个样本")
        
        # Convert DataFrame to list of dicts
        data = []
        discarded_samples_count = 0  # Initialize a counter for bad samples
        csv_dir = os.path.dirname(os.path.abspath(csv_path))
        
        for idx, row in df.iterrows():
            # Handle different CSV formats
            if "filepath" in df.columns and "title" in df.columns:
                # PathText-unpuzzle format: filepath, subset, title
                h5_path = os.path.join("/data/private_hdd/PathText-unpuzzle", row["filepath"])
                caption = row["title"]
                question = ""
                answer = ""
                slide_id = os.path.basename(row["filepath"]).replace(".h5", "")
                dataset_name_field = row.get("subset", "PathText")
            else:
                # Standard format: h5_path, question, answer, caption, etc.
                h5_path = row.get("h5_path", "")
                if not os.path.isabs(h5_path):
                    h5_path = os.path.join(csv_dir, h5_path)
                question = row.get("question", "").strip()
                answer = row.get("answer", "").strip()
                slide_id = row.get("slide_id", "") # Get slide_id early for logging
                
                # 🆕 Injected Prompt Engineering Logic for PathText samples (Case-insensitive)
                if row.get("data_source", "").lower() == "pathtext":
                    question = f"{question}\n\n**TASK:**\nStyle: Comprehensive Pathology Report"
                
                # 🔧 Enhance logic: if choices exist, format and append them to the question
                choices_str = str(row.get("choices", "")) # Ensure it's a string
                is_choice_question = False
                if choices_str and choices_str != '[]':
                    is_choice_question = True
                    try:
                        # Safely parse the string representation of the list
                        choices_list = ast.literal_eval(choices_str)
                        if isinstance(choices_list, list) and choices_list:
                            # Format options as A) B) C) ...
                            formatted_choices = [f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices_list)]
                            choices_text = "\n".join(formatted_choices)
                            question = f"{question}\n\nPlease select from the following options:\n{choices_text}"
                            
                            # Match and reformat the answer, or identify as a bad sample
                            answer_lower = answer.strip().lower()
                            choices_lower = [c.strip().lower() for c in choices_list]
                            
                            found_idx = -1

                            # Priority 1: Direct, exact match (case-insensitive)
                            try:
                                found_idx = choices_lower.index(answer_lower)
                            except ValueError:
                                # Priority 2: Unique substring match (for wrapper text like "the correct option(...)")
                                possible_matches_substring = [i for i, choice in enumerate(choices_lower) if choice in answer_lower]
                                if len(possible_matches_substring) == 1:
                                    found_idx = possible_matches_substring[0]
                                else:
                                    # Priority 3: Unique prefix match (for noise like "answer." or "answer4.")
                                    possible_matches_prefix = [i for i, choice in enumerate(choices_lower) if answer_lower.startswith(choice)]
                                    if len(possible_matches_prefix) == 1:
                                        found_idx = possible_matches_prefix[0]

                            if found_idx != -1:
                                correct_choice_text = choices_list[found_idx]
                                answer = f"{chr(65 + found_idx)}) {correct_choice_text}"
                            else:
                                # ‼️ This is a bad sample, log and discard it
                                discarded_samples_count += 1
                                logger.warning(
                                    f"DISCARDING sample. Answer not found in choices. "
                                    f"slide_id: {slide_id}, answer: '{answer}', choices: {choices_list}"
                                )
                                continue # Skip to the next row in the dataframe

                    except (ValueError, SyntaxError):
                        logger.warning(f"Could not parse choices string: {choices_str}, appending as is.")
                
                caption = row.get("caption", "")
                dataset_name_field = row.get("dataset", "")
            
            data.append({
                "h5_path": h5_path,
                "question": question,
                "answer": answer,
                "caption": caption,
                "slide_id": slide_id,
                "dataset": dataset_name_field,
            })
            
            # Log first few samples for debugging
            if idx < 3:
                logger.info(f"VQA Sample {idx}: h5_path={h5_path}, caption_length={len(caption)}, question_length={len(question)}, answer_length={len(answer)}")
        
        if discarded_samples_count > 0:
            logger.info(f"✅ Total discarded samples due to answer/choice mismatch: {discarded_samples_count}")

        # Create dataset
        dataset = Dataset.from_list(data)
        logger.info(f"Created VQA dataset with {len(dataset)} samples")
        
        # 🔧 使用新的VQA转换器
        converter = DynamicUnpuzzleVQADatasetConverter(
            dataset_attr=dataset_attr,
            data_args=data_args
        )
        return converter.convert(dataset, tokenizer)
        
    except Exception as e:
        logger.error(f"Error loading Unpuzzle VQA dataset: {str(e)}")
        raise

class DynamicUnpuzzleDatasetConverter(DatasetConverter):
    """动态加载WSI数据集转换器 - 避免预加载所有H5文件到内存"""
    
    def __init__(self, dataset_attr: "DatasetAttr", data_args: "DataArguments"):
        super().__init__(dataset_attr, data_args)
        self.tokenizer = None
        self.transforms = None
        
        # WSI processing parameters
        self.h5_key_feat = "features"
        self.h5_key_coord = "coords_yx"
        self.max_patches = getattr(data_args, "patches_per_wsi", 1024)
        
    def convert(self, dataset: Dataset, tokenizer: "PreTrainedTokenizer", transforms=None) -> Dataset:
        """Convert the dataset to LLaMA-Factory format - 延迟加载H5文件"""
        self.tokenizer = tokenizer
        self.transforms = transforms
        
        def convert_example(example: Dict) -> Dict:
            # 🔧 延迟加载：只存储H5路径和元数据，不加载特征数据
            h5_path = example.get("h5_path")
            
            # 只验证文件存在性，不加载特征数据
            if h5_path and not os.path.exists(h5_path):
                logger.warning(f"H5 file not found: {h5_path}")
                h5_path = None
            
            # 🎯 Stage2简化：只需要 WSI + Caption 对应关系 
            # 参考LLaVA官方Stage2使用CC-3M的简单image-caption格式
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()
            caption = example.get("caption", "").strip()
            
            if question and answer:
                # Stage3 VQA format: 保持复杂格式用于后续VQA训练
                user_content = f"<image>\n{question}"
                assistant_content = answer
            elif caption:
                # 🔧 Stage2修正格式: 明确的指令，而不是简单的续写
                # 参考LLaVA官方做法：图像标记 + 一个固定的、通用的问题
                question_text = "Please provide a comprehensive pathology report for this whole-slide image."
                user_content = f"<image>\n{question_text}"
                assistant_content = caption  # 使用caption作为response
            else:
                # Fallback: 使用基本描述
                slide_id = example.get("slide_id", "unknown")
                user_content = f"<image>"
                assistant_content = f"This is a histopathological slide from slide {slide_id}."
            
            # 简化的对话格式 - 去掉不必要的复杂性
            prompt = [{"role": "user", "content": user_content}]
            response = [{"role": "assistant", "content": assistant_content}]
            
            return {
                "_prompt": prompt,
                "_response": response,
                "_system": "",  # 🎯 Stage2简化：移除复杂的system message
                "_tools": "",
                "_images": [h5_path] if h5_path else [],  # 🔧 存储H5路径而不是特征数据
                "_videos": None,
                "_audios": None,
                # 保留关键元数据用于调试
                "_slide_id": example.get("slide_id", ""),
                "_dataset": example.get("dataset", ""),
                "_h5_path": h5_path,
                "_max_patches": self.max_patches,  # 传递patch数量参数
                # 🔧 简化：移除原始问答字段，专注于caption
                "_original_caption": caption,
            }
        
        return dataset.map(
            convert_example,
            remove_columns=dataset.column_names,
            desc="Converting Unpuzzle dataset to LLaMA-Factory format (Dynamic Loading)"
        )

class UnpuzzleDatasetConverter(DatasetConverter):
    """Convert Unpuzzle WSI dataset to LLaMA-Factory format."""
    
    def __init__(self, dataset_attr: "DatasetAttr", data_args: "DataArguments"):
        super().__init__(dataset_attr, data_args)
        self.tokenizer = None
        self.transforms = None
        
        # WSI processing parameters
        self.h5_key_feat = "features"
        self.h5_key_coord = "coords_yx"
        self.max_patches = getattr(data_args, "patches_per_wsi", 1024)
        
    def convert(self, dataset: Dataset, tokenizer: "PreTrainedTokenizer", transforms=None) -> Dataset:
        """Convert the dataset to LLaMA-Factory format."""
        self.tokenizer = tokenizer
        self.transforms = transforms
        
        def convert_example(example: Dict) -> Dict:
            # 🔧 延迟加载：只存储H5路径，不在此阶段加载特征
            h5_path = example.get("h5_path")
            
            # 只验证文件存在性，不加载特征数据
            if h5_path and not os.path.exists(h5_path):
                logger.warning(f"H5 file not found: {h5_path}")
                h5_path = None
            
            # 🔧 Stage2修正格式: 与DynamicUnpuzzleDatasetConverter保持一致
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()
            caption = example.get("caption", "").strip()
            
            if question and answer:
                # Stage3 VQA format: 保持完整格式
                user_content = f"<image>\n{question}"
                assistant_content = answer
            elif caption:
                # 🔧 Stage2修正格式: 图像标记 + caption内容
                user_content = f"<image>"
                assistant_content = caption  # 使用caption作为response
            else:
                # Fallback: 修正格式
                slide_id = example.get("slide_id", "unknown")
                user_content = f"<image>"
                assistant_content = f"This is a histopathological slide from slide {slide_id}."
            
            # 简化的对话格式
            prompt = [{"role": "user", "content": user_content}]
            response = [{"role": "assistant", "content": assistant_content}]
            
            return {
                "_prompt": prompt,
                "_response": response,
                "_system": "",  # 🎯 Stage2简化：移除复杂的system message
                "_tools": "",
                "_images": [h5_path] if h5_path else [],  # 🔧 存储H5路径而不是特征数据
                "_videos": None,
                "_audios": None,
                # 保留关键元数据
                "_slide_id": example.get("slide_id", ""),
                "_dataset": example.get("dataset", ""),
                "_h5_path": h5_path,
                # 🔧 简化：专注于caption
                "_original_caption": caption,
            }
        
        return dataset.map(
            convert_example,
            remove_columns=dataset.column_names,
            desc="Converting Unpuzzle dataset to LLaMA-Factory format"
        )

    def __call__(self, example: Dict) -> Dict:
        """Convert a single example to LLaMA-Factory format."""
        # 🔧 Stage2修正格式：与主要转换逻辑保持一致
        question = example.get("question", "").strip()
        answer = example.get("answer", "").strip()
        caption = example.get("caption", "").strip()
        
        if question and answer:
            user_content = f"<image>\n{question}"
            assistant_content = answer
        elif caption:
            user_content = f"<image>"  # 修正格式
            assistant_content = caption  # 使用caption作为response
        else:
            user_content = f"<image>"
            assistant_content = "This is a histopathological slide."
        
        return {
            "_prompt": [{"role": "user", "content": user_content}],
            "_response": [{"role": "assistant", "content": assistant_content}],
            "_system": "",  # 🎯 Stage2简化：移除复杂的system message
            "_tools": "",
            "_images": None,  # Will be processed during actual conversion
            "_videos": None,
            "_audios": None,
        } 

class DynamicUnpuzzleVQADatasetConverter(DatasetConverter):
    """动态加载WSI数据集转换器 - Stage3 VQA模式，将caption转换为问答对"""
    
    def __init__(self, dataset_attr: "DatasetAttr", data_args: "DataArguments"):
        super().__init__(dataset_attr, data_args)
        self.tokenizer = None
        self.transforms = None
        
        # WSI processing parameters
        self.h5_key_feat = "features"
        self.h5_key_coord = "coords_yx"
        self.max_patches = getattr(data_args, "patches_per_wsi", 1024)
        
    def convert(self, dataset: Dataset, tokenizer: "PreTrainedTokenizer", transforms=None) -> Dataset:
        """Convert the dataset to LLaMA-Factory format with VQA question generation"""
        self.tokenizer = tokenizer
        self.transforms = transforms
        
        def convert_example(example: Dict) -> Dict:
            # 🔧 延迟加载：只存储H5路径和元数据，不加载特征数据
            h5_path = example.get("h5_path")
            
            # 只验证文件存在性，不加载特征数据
            if h5_path and not os.path.exists(h5_path):
                logger.warning(f"H5 file not found: {h5_path}")
                h5_path = None
            
            # 🎯 Stage3 VQA模式：优先生成问答对
            question = example.get("question", "").strip()
            answer = example.get("answer", "").strip()
            caption = example.get("caption", "").strip()
            slide_id = example.get("slide_id", "unknown")
            
            generated_question = None  # 初始化变量
            
            if question and answer:
                # 已有问答对，直接使用
                user_content = f"<image>\n{question}"
                assistant_content = answer
                logger.debug(f"使用现有问答对: slide_id={slide_id}")
            elif caption:
                # 🔧 Stage3核心：自动生成问答对
                # 使用slide_id确保同一个slide总是得到相同的问句
                generated_question = make_vqa_question(slide_id)
                user_content = f"<image>\n{generated_question}"
                assistant_content = caption  # caption作为答案
                logger.debug(f"生成VQA问答对: slide_id={slide_id}, question={generated_question[:50]}...")
            else:
                # Fallback: 使用基本问答
                generated_question = "What does this whole-slide image show?"
                user_content = f"<image>\n{generated_question}"
                assistant_content = f"This is a histopathological slide from slide {slide_id}."
                logger.debug(f"使用fallback问答: slide_id={slide_id}")
            
            # 标准的对话格式
            prompt = [{"role": "user", "content": user_content}]
            response = [{"role": "assistant", "content": assistant_content}]
            
            return {
                "_prompt": prompt,
                "_response": response,
                "_system": "",  # 🎯 Stage3简化：移除复杂的system message
                "_tools": "",
                "_images": [h5_path] if h5_path else [],  # 🔧 存储H5路径而不是特征数据
                "_videos": None,
                "_audios": None,
                # 保留关键元数据用于调试
                "_slide_id": slide_id,
                "_dataset": example.get("dataset", ""),
                "_h5_path": h5_path,
                "_max_patches": self.max_patches,  # 传递patch数量参数
                # 🔧 VQA模式：保留原始数据
                "_original_caption": caption,
                "_original_question": question,
                "_original_answer": answer,
                "_generated_question": generated_question if not question else None,
            }
        
        return dataset.map(
            convert_example,
            remove_columns=dataset.column_names,
            desc="Converting Unpuzzle dataset to LLaMA-Factory VQA format (Dynamic Loading)"
        ) 