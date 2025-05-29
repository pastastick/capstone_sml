import streamlit as st
from PIL import Image
import os
import tempfile
from csad import inference_openvino_modif, MVTecLOCODataset
from glass import GLASSInference
import torch
import logging
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, model_name, category):
        """
        Initialize detector with specific model and category
        
        Args:
            model_name: Name of model to use (GLASS, CSAD, etc.)
            category: Category to detect anomalies for (display name)
        """
        self.model_name = model_name
        self.display_category = category
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert display category to internal category
        self.category = self._get_internal_category(category)
        
        # Load thresholds and accuracy for this model
        self.thresholds = self._get_thresholds()
        self.accuracy = self._get_accuracy()
        
    def _get_internal_category(self, display_category):
        """Convert display category to internal category name"""
        if self.model_name == "CSAD":
            return config.COMPLEX_CATEGORIES.get(display_category, display_category.lower())
        else:
            return config.MODEL_MAP.get(display_category, display_category.lower())
    
    def _get_thresholds(self):
        """Get threshold configuration for the current model"""
        threshold_map = {
            "CSAD": config.thresholds_CSAD,
            "GLASS": config.thresholds_GLASS,
            "Autoencoder with L2 loss function": config.thresholds_Autoencoder,
            "ResNet50 with KNN": config.thresholds_ResNet50,
            "Res2Net": config.thresholds_Res2Net,
            "PatchCore": config.thresholds_PatchCore,
        }
        
        thresholds = threshold_map.get(self.model_name)
        if thresholds is None:
            raise ValueError(f"Model '{self.model_name}' not supported")
        
        return thresholds
    
    def _get_accuracy(self):
        """Get accuracy configuration for the current model"""
        accuracy_map = {
            "CSAD": config.accuracy_CSAD,
            "GLASS": config.accuracy_GLASS,
            "Autoencoder with L2 loss function": config.accuracy_Autoencoder,
            "ResNet50 with KNN": config.accuracy_ResNet50,
            "Res2Net": config.accuracy_Res2Net,
            "PatchCore": config.accuracy_PatchCore,
        }
        
        accuracy = accuracy_map.get(self.model_name)
        if accuracy is None:
            raise ValueError(f"Model '{self.model_name}' not supported")
        
        return accuracy
    
    def get_threshold(self):
        """Get appropriate threshold based on model and category"""
        threshold = self.thresholds.get(self.category)
        if threshold is None:
            raise ValueError(f"Category '{self.category}' not found in thresholds for model '{self.model_name}'")
        
        return threshold
    
    def get_accuracy(self):
        """Get accuracy for the current category"""
        accuracy = self.accuracy.get(self.category)
        if accuracy is None:
            LOGGER.warning(f"Accuracy not found for category '{self.category}' in model '{self.model_name}'")
            return 0
        
        return accuracy
    
    def get_overall_accuracy(self):
        """Get overall accuracy for the current model"""
        return self.accuracy.get("overall", 0)
    
    def classify_anomaly(self, score):
        """
        Classify anomaly based on score and threshold
        
        Args:
            score: Anomaly score from model
            
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
                # If threshold is not a dict, use it directly
                return "NORMAL" if score < threshold else "ANOMALY"
        else:
            # For other models, threshold is a single value
            return "NORMAL" if score < threshold else "ANOMALY"

    def predict(self, image_bytes):
        """
        Run inference with appropriate model
        
        Args:
            image_bytes: Bytes of the uploaded image
            
        Returns:
            tuple: (score, classification)
        """
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                # Write bytes to temp file
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
                
            # Process based on model type
            if self.model_name == "CSAD":
                score = inference_openvino_modif(tmp_path, self.category)
                
            elif self.model_name == "GLASS":
                glass_model = GLASSInference(device=self.device, category=self.category)
                scores = glass_model.predict(tmp_path)
                score = scores[0]
            
            ###### belum di inisiasi #################
            elif self.model_name == "Autoencoder with L2 loss function":
                # Placeholder for autoencoder implementation
                raise NotImplementedError(f"Model '{self.model_name}' not implemented yet")
                
            elif self.model_name == "ResNet50 with KNN":
                # Placeholder for ResNet50 implementation
                raise NotImplementedError(f"Model '{self.model_name}' not implemented yet")
                
            elif self.model_name == "Res2Net":
                # Placeholder for Res2Net implementation
                raise NotImplementedError(f"Model '{self.model_name}' not implemented yet")
                
            elif self.model_name == "PatchCore":
                # Placeholder for PatchCore implementation
                raise NotImplementedError(f"Model '{self.model_name}' not implemented yet")
                
            else:
                raise ValueError(f"Model '{self.model_name}' not supported")
                
            score = float(score)
            # Classify the result
            classification = self.classify_anomaly(score)
            
            LOGGER.info(f"Model: {self.model_name}, Category: {self.category}, Score: {score:.4f}, Result: {classification}")
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return score, classification
            
        except Exception as e:
            LOGGER.error(f"Error processing image with {self.model_name} model: {str(e)}")
            # Clean up temp file if it exists
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
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
            "overall_accuracy": self.get_overall_accuracy()
        }