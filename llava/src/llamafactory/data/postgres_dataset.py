import os
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import psycopg2
from datasets import Dataset
from transformers import PreTrainedTokenizer
from PIL import Image
import torch
import numpy as np

# Disable PIL image size limit
Image.MAX_IMAGE_PIXELS = None

from ..extras import logging
from .base_converter import DatasetConverter
from .data_utils import Role

if TYPE_CHECKING:
    from ..hparams import DataArguments
    from .parser import DatasetAttr

logger = logging.get_logger(__name__)

def get_postgres_dataset(
    dataset_name: str,
    dataset_dir: str,
    tokenizer: PreTrainedTokenizer,
    **kwargs
) -> Dataset:
    """Load and process the PostgresDataset."""
    try:
        # Connect to PostgreSQL using environment variables or default values
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "vqa_demo"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "12345678")  # Default password from demo
        )
        
        # Create cursor
        cur = conn.cursor()
        
        # Execute query to get image data with text
        cur.execute("""
            WITH image_text AS (
                SELECT i.image_id, array_agg(t.text) as texts
                FROM image i
                LEFT JOIN "image-text" it ON i.image_id = it.image_id
                LEFT JOIN text t ON it.text_id = t.text_id
                GROUP BY i.image_id
            )
            SELECT i.*, it.texts
            FROM image i
            LEFT JOIN image_text it ON i.image_id = it.image_id
            ORDER BY i.image_id
        """)
        
        # Fetch all rows
        rows = cur.fetchall()
        
        # Get column names
        column_names = [desc[0] for desc in cur.description]
        # Convert to list of dicts
        data = []
        for row in rows:
            row_dict = dict(zip(column_names, row))
            # Convert texts array to list if it exists
            if row_dict.get('texts') is not None:
                row_dict['text'] = row_dict.pop('texts')
            else:
                row_dict['text'] = []
            data.append(row_dict)
        
        # Close cursor and connection
        cur.close()
        conn.close()
        
        # Create dataset
        dataset = Dataset.from_list(data)
        
        # Convert to LLaMA-Factory format
        converter = PostgresDatasetConverter(
            dataset_attr=kwargs.get("dataset_attr"),
            data_args=kwargs.get("data_args")
        )
        return converter.convert(dataset, tokenizer)
        
    except Exception as e:
        logger.error(f"Error loading PostgresDataset: {str(e)}")
        raise

class PostgresDatasetConverter(DatasetConverter):
    """Convert PostgresDataset to LLaMA-Factory format."""
    
    def __init__(self, dataset_attr: "DatasetAttr", data_args: "DataArguments"):
        super().__init__(dataset_attr, data_args)
        self.tokenizer = None  # Will be set during conversion
        self.transforms = None  # Will be set during conversion
        
    def convert(self, dataset: Dataset, tokenizer: PreTrainedTokenizer, transforms=None) -> Dataset:
        """Convert the dataset to LLaMA-Factory format."""
        self.tokenizer = tokenizer
        self.transforms = transforms
        
        def convert_example(example: Dict) -> Dict:
            # Load and transform image
            image_path = example.get("path")
            if image_path is not None:
                try:
                    image = Image.open(image_path)
                    if self.transforms is not None:
                        image = self.transforms(image)
                except Exception as e:
                    logger.error(f"Error loading image {image_path}: {str(e)}")
                    image = None
            else:
                image = None

            # Get text from the example
            text = example.get("text", [])
            if not text:
                # Use a unique default text based on the image path
                text = [f"Please analyze this medical image: {image_path}"]
            
            # Handle both VQA and captioning cases
            if len(text) >= 2:
                # VQA case: first text is question, second is answer
                question = text[0]
                answer = text[1]
                prompt = [
                    {
                        "role": "user",
                        "content": f"<image>\n{question}"
                    }
                ]
                response = [
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            else:
                # Captioning case: single text is both question and answer
                caption = text[0]
                prompt = [
                    {
                        "role": "user",
                        "content": f"<image>\nPlease describe this medical image."
                    }
                ]
                response = [
                    {
                        "role": "assistant",
                        "content": caption
                    }
                ]
            
            return {
                "_prompt": prompt,
                "_response": response,
                "_system": "You are a helpful assistant that analyzes medical images.",
                "_tools": "",
                "_images": [image] if image is not None else None,
                "_videos": None,
                "_audios": None
            }
        
        return dataset.map(
            convert_example,
            remove_columns=dataset.column_names,
            desc="Converting PostgresDataset to LLaMA-Factory format"
        )

    def __call__(self, example: Dict) -> Dict:
        """Convert a single example to LLaMA-Factory format."""
        # Load and transform image
        image_path = example.get("path")
        if image_path is not None:
            try:
                image = Image.open(image_path)
                if self.transforms is not None:
                    image = self.transforms(image)
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {str(e)}")
                image = None
        else:
            image = None

        # Get text from the example
        text = example.get("text", [])
        if not text:
            # Use a unique default text based on the image path
            text = [f"Please analyze this medical image: {image_path}"]
        
        # Handle both VQA and captioning cases
        if len(text) >= 2:
            # VQA case: first text is question, second is answer
            question = text[0]
            answer = text[1]
            prompt = [
                {
                    "role": "user",
                    "content": f"<image>\n{question}"
                }
            ]
            response = [
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
        else:
            # Captioning case: single text is both question and answer
            caption = text[0]
            prompt = [
                {
                    "role": "user",
                    "content": f"<image>\nPlease describe this medical image."
                }
            ]
            response = [
                {
                    "role": "assistant",
                    "content": caption
                }
            ]
        
        return {
            "_prompt": prompt,
            "_response": response,
            "_system": "You are a helpful assistant that analyzes medical images.",
            "_tools": "",
            "_images": [image] if image is not None else None,
            "_videos": None,
            "_audios": None
        } 