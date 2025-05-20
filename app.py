import streamlit as st
from PIL import Image
import os
from csad import inference_openvino_modif, MVTecLOCODataset
from glass import GLASSInference
import torch
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Threshold per kategori
thresholds = {
    "breakfast_box": {"logical": 2.3232, "structural": -1.4716},
    "juice_bottle": {"logical": 13.4014, "structural": 14.5729},
    "pushpins": {"logical": 8.9723, "structural": 3.4282},
    "screw_bag": {"logical": 9.4575, "structural": 4.5171},
    "splicing_connectors": {"logical": 12.4772, "structural": 12.2652},
    "bottle": 0.9495,
    "cable": 0.6089,
    "capsule": 0.6297,
    "carpet": 0.9357,
    "grid": 0.7292,
    "hazelnut": 0.9702,
    "leather": 0.9311,
    "metal_nut": 0.7905,
    "pill": 0.3974,
    "screw": 0.4773,
    "tile": 0.9774,
    "toothbrush": 0.9055,
    "transistor":0.9818,
    "wood": 0.9737,
    "zipper": 0.8219,
}

# Mapping nama tampilan ke kategori internal
MODEL_MAP = {
    "Breakfast Box": "breakfast_box",
    "Juice Bottle": "juice_bottle",
    "Pushpins": "pushpins",
    "Screw Bag": "screw_bag",
    "Splicing Connectors": "splicing_connectors",
    "Bottle": "bottle",
    "Cable": "cable",
    "Capsule": "capsule",
    "Carpet": "carpet",
    "Grid": "grid",
    "Hazelnut": "hazelnut",
    "Leather": "leather",
    "Metal": "metal_nut",
    "Pill": "pill",
    "Screw": "screw",
    "Tile": "tile",
    "Toothbrush": "toothbrush",
    "Transistor": "transistor",
    "Wood": "wood",
    "Zipper": "zipper"
}

# Daftar kategori dengan dua threshold (logical & structural) dari CSAD
COMPLEX_CATEGORIES = {
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors"
}

class AnomalyDetector:
    def __init__(self, category):
        self.category = category
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def classify_anomaly(self, score):
        """Fungsi klasifikasi anomaly yang sama seperti sebelumnya"""
        th = thresholds.get(self.category)
        if th is None:
            raise ValueError(f"Kategori '{self.category}' tidak ditemukan di thresholds.")

        if self.category in COMPLEX_CATEGORIES:
            limit = th["logical"] if self.category == "juice_bottle" else th["structural"]
            return "NORMAL" if score < limit else "ANOMALY"
        else:
            return "NORMAL" if score < th else "ANOMALY"

    def predict(self, image_path):
        """Mengembalikan skor anomaly dan hasil klasifikasi"""
        try:
            if self.category in COMPLEX_CATEGORIES:
                score = inference_openvino_modif(image_path, self.category)
            else:
                glass_model = GLASSInference(device=self.device, category=self.category)
                scores = glass_model.predict(image_path)
                score = scores[0]
                LOGGER.info(f"Raw score: {score}")
            
            result = self.classify_anomaly(score)
            return score, result
            
        except Exception as e:
            LOGGER.error(f"Error processing image: {str(e)}")
            raise