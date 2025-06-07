# Updated Configuration - Auto Model Selection Based on Category

# All available categories
ALL_CATEGORIES = {
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
    "Metal nut": "metal_nut",
    "Pill": "pill",
    "Screw": "screw",
    "Tile": "tile",
    "Toothbrush": "toothbrush",
    "Transistor": "transistor",
    "Wood": "wood",
    "Zipper": "zipper",
}

# Model assignment rules based on category
CATEGORY_MODEL_MAP = {
    # PatchCore categories
    "bottle": "PatchCore",
    "hazelnut": "PatchCore", 
    "cable": "PatchCore",
    
    # CSAD categories
    "breakfast_box": "CSAD",
    "juice_bottle": "CSAD",
    "pushpins": "CSAD",
    "screw_bag": "CSAD",
    "splicing_connectors": "CSAD",
    
    # GLASS categories (all remaining)
    "capsule": "GLASS",
    "carpet": "GLASS",
    "grid": "GLASS",
    "leather": "GLASS",
    "metal_nut": "GLASS",
    "pill": "GLASS",
    "screw": "GLASS",
    "tile": "GLASS",
    "toothbrush": "GLASS",
    "transistor": "GLASS",
    "wood": "GLASS",
    "zipper": "GLASS",
}

# Function to get model based on category
def get_model_for_category(category_internal):
    """Returns the appropriate model for a given category"""
    return CATEGORY_MODEL_MAP.get(category_internal, "GLASS")

# Function to get accuracy for category
def get_accuracy_for_category(category_internal):
    """Returns the accuracy for a given category based on its assigned model"""
    model = get_model_for_category(category_internal)
    
    if model == "CSAD":
        return accuracy_CSAD.get(category_internal, 0)
    elif model == "GLASS":
        return accuracy_GLASS.get(category_internal, 0)
    elif model == "PatchCore":
        return accuracy_PatchCore.get(category_internal, 0)
    else:
        return 0

# Function to get threshold for category
def get_threshold_for_category(category_internal):
    """Returns the threshold for a given category based on its assigned model"""
    model = get_model_for_category(category_internal)
    
    if model == "CSAD":
        return thresholds_CSAD.get(category_internal, {"logical": 0, "structural": 0})
    elif model == "GLASS":
        return thresholds_GLASS.get(category_internal, 0)
    elif model == "PatchCore":
        return thresholds_PatchCore.get(category_internal, 0)
    else:
        return 0

# Threshold configurations for different models
thresholds_CSAD = {
    "breakfast_box": {"logical": 2.3232, "structural": -1.4716},
    "juice_bottle": {"logical": 13.4014, "structural": 14.5729},
    "pushpins": {"logical": 8.9723, "structural": 3.4282},
    "screw_bag": {"logical": 9.4575, "structural": 4.5171},
    "splicing_connectors": {"logical": 12.4772, "structural": 12.2652},
}

thresholds_GLASS = {
    "capsule": 0.6297,
    "carpet": 0.9357,
    "grid": 0.7292,
    "leather": 0.9311,
    "metal_nut": 0.7905,
    "pill": 0.3974,
    "screw": 0.4773,
    "tile": 0.9774,
    "toothbrush": 0.9055,
    "transistor": 0.9818,
    "wood": 0.9737,
    "zipper": 0.8219,
}

# Note: PatchCore thresholds need to be initialized with actual values
thresholds_PatchCore = {
    "bottle": 0.95,     # Placeholder - needs actual value
    "hazelnut": 0.97,   # Placeholder - needs actual value  
    "cable": 0.61,      # Placeholder - needs actual value
}

# Accuracy configurations
accuracy_CSAD = {
    "breakfast_box": 89,
    "juice_bottle": 92,
    "pushpins": 93,
    "screw_bag": 94,
    "splicing_connectors": 91,
}

accuracy_GLASS = {
    "capsule": 99,
    "carpet": 98,
    "grid": 100,
    "leather": 100,
    "metal_nut": 100,
    "pill": 97,
    "screw": 99,
    "tile": 100,
    "toothbrush": 100,
    "transistor": 99,
    "wood": 98,
    "zipper": 99,
}

# Note: PatchCore accuracies need to be initialized with actual values
accuracy_PatchCore = {
    "bottle": 95,      # Placeholder - needs actual value
    "hazelnut": 98,    # Placeholder - needs actual value
    "cable": 92,       # Placeholder - needs actual value
}