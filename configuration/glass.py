import os
import torch
import glob
import numpy as np
import PIL
import logging
from torchvision import transforms
import torch.nn.functional as F
import math
from complement import backbones, utils, common
from complement.model import Discriminator, Projection, PatchMaker

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Konstanta untuk normalisasi ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_paths, 
        classname, 
        rotate_degrees=0,
        translate=0,
        scale=0,
        resize=288, 
        imagesize=288,
        brightness_factor=0,
        contrast_factor=0,
        saturation_factor=0,
        gray_p=0,
        h_flip_p=0,
        v_flip_p=0,
        ):
        self.image_paths = image_paths
        self.classname = classname
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        
        if self.classname in ['toothbrush', 'wood']:
            self.resize = round(self.imgsize * 329 / 288)
        else:
            self.resize = resize
            
        self.transform = transforms.Compose([
            transforms.Resize(self.resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(gray_p),
            transforms.RandomAffine(
                rotate_degrees,
                translate=(translate, translate),
                scale=(1.0 - scale, 1.0 + scale),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __getitem__(self, idx):
        image = PIL.Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)

    def __len__(self):
        return len(self.image_paths)

class GLASSInference:
    def __init__(self, device, category):
        self.device = device
        self.category = category
        self.model = self._load_model()
        
    def _load_model(self):
        # Inisialisasi model dengan konfigurasi default
        backbone_name = "wideresnet50"
        try:
            backbone = backbones.load(backbone_name)
        except Exception as e:
            LOGGER.error(f"Failed to load backbone {backbone_name}: {str(e)}")
            raise
        
        input_shape = (3, 288, 288)
        
        glass = GLASS()
        glass.load(
            backbone=backbone,
            layers_to_extract_from=["layer2", "layer3"],
            device=self.device,
            input_shape=input_shape,
            pretrain_embed_dimension=1536,
            target_embed_dimension=1536
        )
        
        # Find and load trained weights
        model_dir = os.path.join("results", "models")
        ckpt_pattern = os.path.join(model_dir, "**", f"mvtec_{self.category}", "ckpt_best*.pth")
        ckpt_paths = glob.glob(ckpt_pattern, recursive=True)
        
        if not ckpt_paths:
            raise FileNotFoundError(
                f"No model checkpoint found for category {self.category} in {model_dir}. "
                f"Search pattern was: {ckpt_pattern}"
            )
        
        # Take the first match if multiple found
        ckpt_path = ckpt_paths[0]
        LOGGER.info(f"Loading model from: {ckpt_path}")
        
        # Di dalam metode _load_model pada GLASSInference:
        try:
            state_dict = torch.load(ckpt_path, map_location=self.device)
            missing_keys, unexpected_keys = glass.discriminator.load_state_dict(state_dict['discriminator'], strict=False)
            
            # Tambahkan pemuatan bobot pre_projection jika ada
            if 'pre_projection' in state_dict and hasattr(glass, 'pre_projection'):
                glass.pre_projection.load_state_dict(state_dict['pre_projection'])
                LOGGER.info("Loaded pre_projection weights")
                
            if missing_keys:
                LOGGER.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                LOGGER.warning(f"Unexpected keys: {unexpected_keys}")
        except Exception as e:
            LOGGER.error(f"Failed to load model weights: {str(e)}")
            raise

        return glass.to(self.device)

    def predict(self, input_path):
        LOGGER.info(f"Processing image(s) from: {input_path}")
        
        if os.path.isdir(input_path):
            image_paths = [
                os.path.join(input_path, f) 
                for f in os.listdir(input_path) 
                if f.lower().endswith(('png', 'jpg', 'jpeg'))
            ]
        else:
            image_paths = [input_path]

        dataset = SimpleDataset(image_paths, classname=self.category)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
        
        all_scores = []
        with torch.no_grad():
            for batch in loader:
                scores, _ = self.model._predict(batch.to(self.device))
                all_scores.extend(scores)

        return all_scores

class GLASS(torch.nn.Module):
    # Tetap pertahankan class GLASS dengan fungsi:
    # __init__, load, _embed, _predict, forward
    # Hapus semua fungsi terkait testing:
    # tester, _evaluate, predict (versi lama)
    # [Salin implementasi GLASS dari modif.py dan hapus fungsi yang tidak perlu]
    def __init__(self):
        super(GLASS, self).__init__()
        
    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            svd=0,
            step=20,
            limit=392,
            **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device  # Menetapkan device disini

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_margin = dsc_margin

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)

        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        if evaluation:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            _features = patch_features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, 3, 4, 5, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, 4, 5, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            patch_features[i] = _features

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes
    
    def forward(self, x):
        """Forward pass yang mengembalikan skor anomali"""
        # Pastikan x adalah tensor valid sebelum dilanjutkan
        if isinstance(x, torch.Tensor):
            return self._predict(x)[0]
        else:
           raise ValueError(f"Input must be a tensor, got {type(x)} instead")
    
    @torch.no_grad()
    def _predict(self, img):
        if not isinstance(img, torch.Tensor):
            raise ValueError(f"Input must be a tensor, got {type(img)} instead")
        
        if img.dim() != 4:
            img = img.unsqueeze(0)  # Auto-handle single image

        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():
            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            # Apply discriminator to get raw scores
            patch_scores = image_scores = self.discriminator(patch_features)
            
            # Process patch scores for segmentation
            unpatch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = unpatch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            # Process image scores for classification
            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
        