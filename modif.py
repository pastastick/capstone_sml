import os
import torch
import glob
import numpy as np
import PIL
import logging
import tqdm
from complement import backbones, utils, metrics, common
from torchvision import transforms
import torch.nn.functional as F
import math
from complement.model import Discriminator, Projection, PatchMaker
import pandas as pd    
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Konstanta untuk normalisasi ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Path dataset
datapath = 'C:/SEMESTER_6/capstone_sml/dataset/anomaly_detection'

# Pastikan path dataset ada
assert os.path.exists(datapath), "Dataset path invalid!"

# Kelas MVTecDataset untuk preprocessing data (sesuai paper GLASS)
class MVTecDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            source,
            anomaly_source_path='/root/dataset/dtd/images',
            dataset_name='mvtec',
            classname='leather',
            resize=288,
            imagesize=288,
            split="test",
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            downsampling=8,
            scale=0,
            batch_size=8,
            seed=0,  # Menambahkan parameter seed
            **kwargs,
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = fg
        self.rand_aug = rand_aug
        self.downsampling = downsampling
        self.resize = resize if self.distribution != 1 else [resize, resize]
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname
        self.dataset_name = dataset_name

        if self.distribution != 1 and (self.classname == 'toothbrush' or self.classname == 'wood'):
            self.resize = round(self.imgsize * 329 / 288)

        if self.fg == 1:
            self.class_fg = 1
        else:
            self.class_fg = 0

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = [
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
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)

    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.resize),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        transform_aug = transforms.Compose(transform_aug)
        return transform_aug

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        mask_fg = mask_s = aug_image = torch.tensor([1])

        if self.split == "test" and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "aug": aug_image,
            "mask_s": mask_s,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        classpath = os.path.join(self.source, self.classname, self.split)
        maskpath = os.path.join(self.source, self.classname, "ground_truth")
        anomaly_types = os.listdir(classpath)

        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_path, x) for x in anomaly_files]

            if self.split == "test" and anomaly != "good":
                anomaly_mask_path = os.path.join(maskpath, anomaly)
                anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                maskpaths_per_class[self.classname][anomaly] = [os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
            else:
                maskpaths_per_class[self.classname]["good"] = None

        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == "test" and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate

# Kelas GLASS (disesuaikan untuk testing saja)
class GLASS(torch.nn.Module):
    def __init__(self):
        super(GLASS, self).__init__()
        # Menghapus parameter device dari konstruktor
        # Device akan diatur saat memanggil metode load()

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
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1, #### untuk test
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
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

        self.meta_epochs = meta_epochs
        self.eval_epochs = eval_epochs
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
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        # self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        # os.makedirs(self.tb_dir, exist_ok=True)
        # self.logger = TBWrapper(self.tb_dir)

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

    def tester(self, test_data, name):
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            images, scores, segmentations, labels_gt, masks_gt, avg_time_per_batch, avg_time_per_image = self.predict(test_data)
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, pr, re, ac, f1, th = self._evaluate(
                images, scores, segmentations, labels_gt, masks_gt, name, path='eval'
            )
            epoch = int(ckpt_path[0].split('_')[-1].split('.')[0])
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.
            avg_time_per_batch, avg_time_per_image = 0.0, 0.0
            LOGGER.info("No ckpt file found!")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, pr, re, ac, f1, th, epoch, avg_time_per_batch, avg_time_per_image

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training'):
        scores = np.squeeze(np.array(scores))
        labels = np.array(labels_gt)
        
        # Simpan scores dan labels ke CSV
        save_dir = os.path.join("results", "scores")
        os.makedirs(save_dir, exist_ok=True)
        
        # Untuk image-level
        img_df = pd.DataFrame({
            'true_label': labels,
            'anomaly_score': scores
        })
        img_df.to_csv(os.path.join(save_dir, f"{name}_image_scores.csv"), index=False)
        
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]
        
        image_metrics = metrics.compute_best_pr_re(labels_gt, scores)
        image_presisi = image_metrics["precision"]
        image_recal = image_metrics["recall"]
        image_akurasi = image_metrics["accuracy"]
        image_f1 = image_metrics["f1_score"]
        image_threshold = image_metrics["threshold"]

        if len(masks_gt) > 0:
            segmentations = np.array(segmentations)
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(segmentations, masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(masks_gt)), segmentations)
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, image_presisi, image_recal, image_akurasi, image_f1, image_threshold



    # Modify the predict method to measure inference time
    def predict(self, test_dataloader):
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        
        # Initialize timing variables
        total_inference_time = 0.0
        num_batches = 0

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy().tolist())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy().tolist())
                    image = data["image"]
                    images.extend(image.numpy().tolist())
                    img_paths.extend(data["image_path"])
                
                # Start timing inference
                start_time = time.time()
                _scores, _masks = self._predict(image)
                end_time = time.time()
                
                # Calculate and accumulate inference time
                batch_inference_time = end_time - start_time
                total_inference_time += batch_inference_time
                num_batches += 1
                
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        # Calculate average inference time per batch and per image
        avg_time_per_batch = total_inference_time / num_batches if num_batches > 0 else 0
        avg_time_per_image = total_inference_time / len(images) if len(images) > 0 else 0
        
        return images, scores, masks, labels_gt, masks_gt, avg_time_per_batch, avg_time_per_image

    def forward(self, x):
        """Forward pass yang mengembalikan skor anomali"""
        # Pastikan x adalah tensor valid sebelum dilanjutkan
        if isinstance(x, torch.Tensor):
            return self._predict(x)[0]
        else:
            raise ValueError(f"Input harus berupa tensor, bukan {type(x)}")
    
    def _predict(self, img):
        if not isinstance(img, torch.Tensor):
            raise ValueError(f"Input harus berupa tensor, bukan {type(img)}")
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

            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)

# Inisialisasi model GLASS
layers_to_extract_from_coll = [["layer2", "layer3"]]
backbone_names = ["wideresnet50"]
pretrain_embed_dimension = 1536
target_embed_dimension = 1536

def get_glass(input_shape, device):
    glasses = []
    for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
        backbone = backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed

        # Perbaikan disini: Tidak mengirim device ke konstruktor
        glass_inst = GLASS()
        glass_inst.load(
            backbone=backbone,
            layers_to_extract_from=layers_to_extract_from,
            device=device,
            input_shape=input_shape,
            pretrain_embed_dimension=pretrain_embed_dimension,
            target_embed_dimension=target_embed_dimension,
        )
        glass_inst = glass_inst.to(device)
        glasses.append(glass_inst)
    return glasses

# Buat DataLoader untuk semua kelas MVTec
classes = ('carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper')
batch_size = 8
num_workers = 0  # Set ke 0 untuk menghindari masalah multiprocessing di Windows

def get_dataloader(seed, get_name="mvtec"):
    dataloaders = []
    for subclass in classes:
        test_dataset = MVTecDataset(
            source=datapath,
            classname=subclass,
            split="test",
            resize=288,
            imagesize=288,
            fg=0,
            seed=seed
        )
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        
        test_dataloader.name = f"{get_name}_{subclass}"
        test_dataloader.classname = subclass
        dataloaders.append(test_dataloader)
    
    return dataloaders

# Fungsi untuk mengatur device
def set_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        LOGGER.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        LOGGER.info("Using CPU")
    return device

# Jalankan proses testing
if __name__ == "__main__":
    seed = 0
    results_path = "results"
    models_dir = os.path.join(results_path, "models")
    
    # Buat direktori hasil jika belum ada
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    list_of_dataloaders = get_dataloader(seed)
    device = set_torch_device()

    result_collect = []
    timing_collect = []
    
    for dataloader in list_of_dataloaders:
        utils.fix_seeds(seed, device)
        dataset_name = dataloader.name
        imagesize = [3, 288, 288]
        glass_list = get_glass(imagesize, device)
        
        for i, glass in enumerate(glass_list):
            if glass.backbone.seed is not None:
                utils.fix_seeds(glass.backbone.seed, device)
                
            glass.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
            i_auroc, i_ap, p_auroc, p_ap, p_pro, pr, re, ac, f1, th, epoch, avg_time_batch, avg_time_img = glass.tester(dataloader, dataset_name)
            
            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "image_auroc": i_auroc,
                    "image_ap": i_ap,
                    "pixel_auroc": p_auroc,
                    "pixel_ap": p_ap,
                    "pixel_pro": p_pro,
                    "precision": pr,
                    "recall": re,
                    "accuracy": ac,
                    "f1_score": f1,
                    "threshold": th,
                    "best_epoch": epoch,
                }
            )
            
             # Collect timing metrics
            timing_collect.append({
                "dataset_name": dataset_name,
                "avg_time_per_batch": avg_time_batch,
                "avg_time_per_image": avg_time_img,
                "total_images": len(dataloader.dataset)
            })
            
            if epoch > -1:
                # Print performance metrics
                for key, item in result_collect[-1].items():
                    if isinstance(item, str):
                        continue
                    elif isinstance(item, int):
                        print(f"{key}:{item}")
                    else:
                        print(f"{key}:{round(item * 100, 2)} ", end="")
                
                # Print timing metrics
                print("\n\nInference Timing:")
                print(f"Average time per batch: {avg_time_batch:.4f} seconds")
                print(f"Average time per image: {avg_time_img:.4f} seconds")
                print(f"Total inference time: {avg_time_batch * len(dataloader):.4f} seconds")
                        
            print("\n")
            result_metric_names = list(result_collect[-1].keys())[1:]
            result_dataset_names = [results["dataset_name"] for results in result_collect]
            result_scores = [list(results.values())[1:] for results in result_collect]
            utils.compute_and_store_final_results(
                results_path,  # Simpan hasil di direktori "results", bukan notebook_path
                result_scores,
                result_metric_names,
                row_names=result_dataset_names,
            )
            
            # Save timing results to CSV
            timing_df = pd.DataFrame(timing_collect)
            timing_df.to_csv(os.path.join(results_path, "inference_timing.csv"), index=False)
        # Save timing results to CSV
        timing_df = pd.DataFrame(timing_collect)
        timing_df.to_csv(os.path.join(results_path, "inference_timing.csv"), index=False)