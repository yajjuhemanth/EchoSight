#!/usr/bin/env python3
"""
Utility script to download and cache models required for ECHOSIGHT.
This helps users prepare their environment without waiting for the first run.
"""

import os
import sys
import argparse
import logging
from transformers import BlipProcessor, BlipForConditionalGeneration

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_blip_model(model_name="Salesforce/blip-image-captioning-base"):
    """Download and cache the BLIP model."""
    try:
        logger.info(f"Downloading BLIP processor: {model_name}")
        processor = BlipProcessor.from_pretrained(model_name)
        logger.info(f"Downloading BLIP model: {model_name}")
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        logger.info("Model and processor successfully downloaded and cached.")
        return True
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download models for ECHOSIGHT")
    parser.add_argument('--model', type=str, default="Salesforce/blip-image-captioning-base",
                        help="Model name/path to download (default: Salesforce/blip-image-captioning-base)")
    args = parser.parse_args()
    
    logger.info("Starting model download process...")
    success = download_blip_model(args.model)
    
    if success:
        logger.info("All models successfully downloaded.")
        return 0
    else:
        logger.error("Failed to download required models.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 