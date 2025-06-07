import streamlit as st
from PIL import Image
import os
import tempfile
from configuration.csad import inference_openvino_modif, MVTecLOCODataset
from configuration.glass import GLASSInference
from configuration.patchcore import inference_patchcore, resnet_feature_extractor, transform
import torch
import logging
from configuration import config
from pathlib import Path # Pastikan Path diimpor

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Base directory for models (adjust this path as needed in your environment)
# Menggunakan Path() agar MODELS_BASE_DIR menjadi objek Path
MODELS_BASE_DIR = Path("models") 


class AnomalyDetector:
    def __init__(self, category_display):
        """
        Initialize detector with category (model is auto-selected)
        
        Args:
            category_display_name: Display name of category (e.g., "Bottle", "Hazelnut")
        """
        self.display_category = category_display
        self.device = torch.device("cpu") # User requested CPU
        
        # Convert display category to internal category
        self.category = category_display
        
        # Auto-select model based on category
        self.model_name = config.get_model_for_category(self.category)
        
        # Load thresholds and accuracy for this model from config.py
        self.threshold = config.get_threshold_for_category(self.category)
        self.accuracy = config.get_accuracy_for_category(self.category)
        
        LOGGER.info(f"Initialized detector: Category={self.display_category}, "
                   f"Internal={self.category}, Model={self.model_name}, "
                   f"Threshold={self.threshold}, Accuracy={self.accuracy}%")
        
    def get_threshold(self):
        """
        Get threshold for the current category.
        This will fetch the threshold from config.py, which now includes PatchCore thresholds.
        """
        return self.threshold
    
    def get_accuracy(self):
        """Get accuracy for the current category"""
        return self.accuracy
    
    def get_model_name(self):
        """Get the auto-selected model name"""
        return self.model_name
    
    def classify_anomaly(self, score):
        """
        Classify anomaly based on score and the pre-loaded threshold from config.py.
        
        Args:
            score (float): Anomaly score from model
            
        Returns:
            str: "NORMAL" or "ANOMALY"
        """
        threshold = self.get_threshold()
        
        if self.model_name == 'CSAD':
            # CSAD has different threshold structure
            if isinstance(threshold, dict):
                # For juice_bottle, use logical threshold, for others use structural
                if self.category == "juice_bottle":
                    limit = threshold.get("logical")
                else:
                    limit = threshold.get("structural")
                
                if limit is None:
                    raise ValueError(f"Appropriate threshold not found for category '{self.category}' in CSAD model")
                
                return "NORMAL" if score < limit else "ANOMALY"
            else:
                # If threshold is not a dict, use it directly (fallback, though CSAD usually has dict)
                return "NORMAL" if score < threshold else "ANOMALY"
        else:
            # For GLASS and PatchCore, threshold is a single value from config.py
            return "NORMAL" if score < threshold else "ANOMALY"

    def predict(self, image_bytes):
        """
        Run inference with the auto-selected model
        
        Args:
            image_bytes: Bytes of the uploaded image
            
        Returns:
            tuple: (score, classification)
        """
        tmp_path = None # Initialize tmp_path to None
        score = 0.0 # Initialize score
        
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                # Write bytes to temp file
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
                
            # Process based on auto-selected model type
            if self.model_name == "CSAD":
                # CSAD models are typically XML files for OpenVINO
                # model_dir di inference_openvino_modif expects a string, so convert Path to str
                score = inference_openvino_modif(tmp_path, self.category, device=self.device.type.upper())
                
            elif self.model_name == "GLASS":
                glass_model = GLASSInference(device=self.device, category=self.category)
                scores = glass_model.predict(tmp_path)
                score = scores[0]
                
            elif self.model_name == "PatchCore":
                # Construct the path to the memory_bank.pt file for PatchCore
                # Menggunakan Path() concatenation (/) agar memory_bank_path adalah objek Path
                memory_bank_filename = f"{self.category}_memory_bank.pt"
                memory_bank_path = MODELS_BASE_DIR / memory_bank_filename

                # Perform PatchCore inference
                score = inference_patchcore(
                    tmp_path, 
                    memory_bank_path, # memory_bank_path sekarang adalah objek Path
                    device_str=self.device.type
                )
                
            else:
                raise ValueError(f"Model '{self.model_name}' not supported")
                
            score = float(score)
            # Classify the result using the threshold pre-loaded from config.py
            classification = self.classify_anomaly(score)
            
            LOGGER.info(f"Model: {self.model_name}, Category: {self.category}, "
                       f"Score: {score:.4f}, Result: {classification}")
            
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            return score, classification
            
        except Exception as e:
            LOGGER.error(f"Error processing image with {self.model_name} model: {str(e)}")
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    
    def get_model_info(self):
        """
        Get model information including accuracy
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "category": self.display_category,
            "internal_category": self.category,
            "threshold": self.get_threshold(),
            "category_accuracy": self.get_accuracy(),
        }