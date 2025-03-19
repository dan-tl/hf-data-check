#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from datasets import load_dataset
from PIL import Image
import io
from huggingface_hub import login
from loguru import logger

try:
    from config import HF_API_KEY
except ImportError:
    # Use default value when config.py file doesn't exist (user should provide actual value)
    HF_API_KEY = ""

def save_dataset_examples(dataset_name, sample_size=10, total_samples=100):
    """
    Load samples from the specified dataset in streaming mode and save them to a folder.
    
    Args:
        dataset_name (str): Hugging Face dataset name
        sample_size (int): Number of images to save
        total_samples (int): Total number of samples to load
    """
    # Extract folder name from dataset name (e.g., dandelin/cc12m -> cc12m)
    folder_name = dataset_name.split('/')[-1]
    
    # Create folder to save samples
    sample_dir = f"samples_{folder_name}"
    os.makedirs(sample_dir, exist_ok=True)
    logger.info(f"[{folder_name}] Created folder to save samples: {sample_dir}")
    
    # Load dataset in streaming mode
    logger.info(f"[{folder_name}] Loading dataset in streaming mode...")
    try:
        # Handle different split information per dataset
        try:
            dataset = load_dataset(dataset_name, split="train", streaming=True)
        except ValueError:
            try:
                # Use default split if 'train' split doesn't exist
                dataset = load_dataset(dataset_name, streaming=True)
                logger.warning(f"[{folder_name}] 'train' split not found, using default split")
            except Exception as e:
                logger.error(f"[{folder_name}] Failed to load dataset: {e}")
                return 0
    except Exception as e:
        logger.error(f"[{folder_name}] Error occurred while loading dataset: {e}")
        return 0
    
    # Extract samples from stream
    samples = []
    logger.info(f"[{folder_name}] Starting to collect {total_samples} samples...")
    dataset_iter = iter(dataset)
    for i in range(total_samples):
        try:
            example = next(dataset_iter)
            samples.append(example)
            if i % 20 == 0:
                logger.debug(f"[{folder_name}] Loaded {i} samples...")
        except StopIteration:
            logger.warning(f"[{folder_name}] No more samples available.")
            break
        except Exception as e:
            logger.error(f"[{folder_name}] Error processing sample {i}: {e}")
            continue
    
    logger.success(f"[{folder_name}] Successfully loaded {len(samples)} samples")
    
    if len(samples) == 0:
        logger.error(f"[{folder_name}] No samples available.")
        return 0
    
    # Randomly select samples to save
    sample_size = min(sample_size, len(samples))
    selected_samples = random.sample(samples, sample_size)
    logger.info(f"[{folder_name}] Randomly selected {sample_size} out of {len(samples)} samples")
    
    # Save images and text
    saved_count = 0
    for i, example in enumerate(selected_samples):
        try:
            # Image field name can vary by dataset
            image_field = None
            image_data = None
            
            # Check common image field names
            possible_image_fields = ['image', 'img', 'images', 'image_data', 'photo']
            
            # Find image field
            for field in possible_image_fields:
                if field in example:
                    image_field = field
                    image_data = example[field]
                    break
            
            if image_field is None:
                # Directly inspect keys to find image-like fields
                for key, value in example.items():
                    if isinstance(value, (bytes, dict)) and key not in ['text', 'caption', 'label']:
                        image_field = key
                        image_data = value
                        break
            
            if image_field is None:
                logger.warning(f"[{folder_name}] Cannot find image field in sample {i+1}.")
                continue
            
            # Process image
            if isinstance(image_data, dict) and 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes']))
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            else:
                logger.warning(f"[{folder_name}] Unsupported image format: {type(image_data)}")
                continue
            
            # Process caption - text field name can vary by dataset
            text_field = None
            text_data = None
            
            # Check common text field names
            possible_text_fields = ['text', 'caption', 'captions', 'description', 'descriptions', 'title']
            
            # Find text field
            for field in possible_text_fields:
                if field in example:
                    text_field = field
                    text_data = example[field]
                    break
            
            # Process text data
            if text_field is not None:
                if isinstance(text_data, list):
                    # If list, save each item on a separate line with index
                    caption_lines = [f"[{idx}] {item}" for idx, item in enumerate(text_data)]
                    caption = '\n'.join(caption_lines)
                elif isinstance(text_data, str):
                    caption = text_data
                else:
                    caption = str(text_data)
            else:
                caption = 'No caption'
            
            # Save image
            image_path = os.path.join(sample_dir, f"image_{i+1}.jpg")
            image.save(image_path)
            
            # Save caption as text file
            caption_path = os.path.join(sample_dir, f"caption_{i+1}.txt")
            with open(caption_path, 'w', encoding='utf-8') as f:
                f.write(caption)
            
            saved_count += 1
            logger.info(f"[{folder_name}] Sample {i+1} saved successfully")
            logger.debug(f"[{folder_name}] Paths: {image_path}, {caption_path}")
        except Exception as e:
            logger.error(f"[{folder_name}] Error processing sample {i+1}: {e}")
            continue
    
    logger.success(f"[{folder_name}] Saved {saved_count} images and captions to '{sample_dir}' folder.")
    return saved_count

def process_all_datasets():
    """
    Process sampling for all datasets.
    """
    # Login with Hugging Face token
    if not HF_API_KEY:
        logger.error("Hugging Face API key not set. Please check config.py file.")
        return
    
    login(token=HF_API_KEY)
    
    # List of datasets to process
    datasets = [
        "dandelin/redcaps",
        "dandelin/cc12m",
        "dandelin/wit",
        "dandelin/vg",
        "dandelin/sbu",
        "dandelin/cocokt",
        "dandelin/cc3m"
    ]
    
    total_success = 0
    
    # Process each dataset
    for dataset_name in datasets:
        logger.info(f"Starting to process dataset '{dataset_name}'...")
        saved_count = save_dataset_examples(dataset_name)
        if saved_count > 0:
            total_success += 1
            logger.success(f"Completed processing dataset '{dataset_name}'")
        else:
            logger.error(f"Failed to process dataset '{dataset_name}'")
    
    logger.success(f"Successfully processed {total_success} out of {len(datasets)} datasets")

if __name__ == "__main__":
    # Configure logger
    logger.remove()  # Remove default handler
    logger.add(
        "dataset_download.log",
        rotation="10 MB",
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    process_all_datasets()
