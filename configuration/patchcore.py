import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import logging

# Setup logging
LOGGER = logging.getLogger(__name__)

# Define the ResNet feature extractor class based on your .ipynb
# This class needs to be defined here or imported from a common utility file
class resnet_feature_extractor(torch.nn.Module):
    def __init__(self):
        """This class extracts the feature maps from a pretrained Resnet model."""
        super(resnet_feature_extractor, self).__init__()
        # Import ResNet50 and its weights here to keep it self-contained
        from torchvision.models import resnet50, ResNet50_Weights
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Hook to extract feature maps
        def hook(module, input, output) -> None:
            """This hook saves the extracted feature map on self.features."""
            self.features.append(output)

        self.features = [] # Initialize features list
        self.model.layer2[-1].register_forward_hook(hook)            
        self.model.layer3[-1].register_forward_hook(hook) 

    def forward(self, input_tensor): # Renamed input to input_tensor to avoid conflict with Python keyword
        self.features = []
        with torch.no_grad():
            _ = self.model(input_tensor)

        self.avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]         # Feature map sizes h, w
        self.resize = torch.nn.AdaptiveAvgPool2d(fmap_size)

        resized_maps = [self.resize(self.avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)            # Merge the resized feature maps
        patch = patch.reshape(patch.shape[1], -1).T   # Create a column tensor
        
        return patch

# Define the transformation based on your .ipynb
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def inference_patchcore(image_path, memory_bank_path, device_str="cpu"):
    """
    Performs PatchCore inference for a single image, returning the anomaly score.
    Args:
        image_path (str): Path to the input image file.
        memory_bank_path (Path): Path to the saved memory_bank.pt file.
        device_str (str): 'cpu' or 'cuda'.
    Returns:
        float: The anomaly score (s_star).
    """
    device = torch.device(device_str)

    try:
        # Load the backbone model and move to device
        backbone = resnet_feature_extractor().to(device)
        backbone.eval() # Ensure backbone is in evaluation mode

        # Load memory_bank
        # Use map_location to ensure it loads correctly regardless of saved device
        if memory_bank_path.exists():
            memory_bank = torch.load(memory_bank_path, map_location=device)
        else:
            raise FileNotFoundError(f"Memory bank file not found: {memory_bank_path}")

        LOGGER.info(f"PatchCore inference: Device={device}, Memory bank shape={memory_bank.shape}")

        # Load and transform the image
        image = Image.open(image_path).convert('RGB')
        # Apply transformation and move to device
        test_image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = backbone(test_image)

        # Calculate distances to memory bank
        distances = torch.cdist(features, memory_bank, p=2.0)
        
        # Get min distances for each patch and then the max (s_star)
        dist_score, _ = torch.min(distances, dim=1) 
        s_star = torch.max(dist_score)
        
        # Return the anomaly score.
        return s_star.item() # .item() converts a 0-dim tensor to a Python number

    except Exception as e:
        LOGGER.error(f"Error during PatchCore inference for {image_path}: {str(e)}")
        raise
