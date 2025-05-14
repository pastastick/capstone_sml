import torch
import torch.onnx
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import os
import tqdm
import argparse
import time
import pandas as pd
import cv2
import openvino as ov
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

class Padding2Resize():
    def __init__(self, pad_l, pad_t, pad_r, pad_b):
        self.pad_l = pad_l
        self.pad_t = pad_t
        self.pad_r = pad_r
        self.pad_b = pad_b

    def __call__(self,image,target_size,mode='nearest'):
        shape = len(image.shape)
        if shape == 3:
            image = image[None,:,:,:]
        elif shape == 2:
            image = image[None,None,:,:]
        # B,C,H,W
        if self.pad_b == 0:
            image = image[:,:,self.pad_t:]
        else:
            image = image[:,:,self.pad_t:-self.pad_b]
        if self.pad_r == 0:
            image = image[:,:,:,self.pad_l:]
        else:
            image = image[:,:,:,self.pad_l:-self.pad_r]
        
        if isinstance(image,np.ndarray):
            image = cv2.resize(image,(target_size,target_size),interpolation=cv2.INTER_NEAREST if mode == 'nearest' else cv2.INTER_LINEAR)
        elif isinstance(image,torch.Tensor):
            image = torch.nn.functional.interpolate(image, size=(target_size,target_size), mode=mode)


        if shape == 3:
            return image[0]
        elif shape == 2:
            return image[0,0]
        return image
    
####################################################
# Fungsi untuk padding gambar
def get_padding_functions(orig_size,target_size=256,resize_target_size=None,mode='nearest',fill=0):
    """
        padding_func, inverse_padding_func = get_padding_functions(image.size,target_size=256)
        image2 = padding_func(image) # image2.size = (256,256) with padding
        image2.show()
        image3 = inverse_padding_func(image2) # image3.size = (256,256) without padding
        image3.show()
    """
    resize_target_size = target_size if resize_target_size is None else resize_target_size
    imsize = orig_size
    long_size = max(imsize)
    scale = target_size / long_size
    new_h = int(imsize[1] * scale + 0.5)
    new_w = int(imsize[0] * scale + 0.5)

    if (target_size - new_w) % 2 == 0:
        pad_l = pad_r = (target_size - new_w) // 2
    else:
        pad_l,pad_r = (target_size - new_w) // 2,(target_size - new_w) // 2 + 1
    if (target_size - new_h) % 2 == 0:
        pad_t = pad_b = (target_size - new_h) // 2
    else:
        pad_t,pad_b = (target_size - new_h) // 2,(target_size - new_h) // 2 + 1
    inter =  Image.NEAREST if mode == 'nearest' else Image.BILINEAR

    padding_func = transforms.Compose([
        transforms.Resize((new_h,new_w),interpolation=inter),
        transforms.Pad((pad_l, pad_t, pad_r, pad_b), fill=fill, padding_mode='constant')
    ])
    return padding_func, Padding2Resize(pad_l,pad_t,pad_r,pad_b)

###################################################################
class MVTecLOCODataset(Dataset):
    def __init__(self, image_folder, image_size=256, use_pad=True, to_gpu=False):
        """Inisialisasi dataset untuk inferensi dengan path fleksibel."""
        self.image_folder = image_folder
        self.image_size = image_size
        self.use_pad = use_pad
        self.build_transform()
        # Mengambil semua file PNG dari folder, mendukung subfolder seperti datasets/breakfast_box/11111
        # self.img_paths = glob.glob(os.path.join(image_folder, "**", "*.png"), recursive=True)
        self.load_images(to_gpu=to_gpu)

    def build_transform(self):
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.resize_norm_transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size),interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.aug_tranform = transforms.RandomChoice([
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
        ])
        self.transform_gt = transforms.Compose([
            transforms.ToTensor(),
        ])

    def load_images(self, to_gpu=False):
        """Memuat gambar dari path yang diberikan."""
        # Jika input adalah file, baca hanya satu gambar
        if os.path.isfile(self.image_folder):
            self.img_paths = [self.image_folder]
        # Jika input adalah folder, ambil semua file PNG
        elif os.path.isdir(self.image_folder):
            self.img_paths = glob.glob(os.path.join(self.image_folder, "**", "*.png"), recursive=True)
        else:
            raise ValueError("Path tidak valid atau bukan file/folder.")

        if not self.img_paths:
            raise ValueError("Tidak ada gambar PNG ditemukan di folder yang diberikan.")

        # Menggunakan gambar pertama untuk menentukan fungsi padding
        self.pad_func, self.pad2resize = get_padding_functions(
            Image.open(self.img_paths[0]).size,
            target_size=self.image_size,
            mode='bilinear'
        )
        
        self.samples = list()
        for img_path in self.img_paths:
            img = Image.open(img_path).convert('RGB')
            resize_img = self.resize_norm_transform(img)
            pad_img = self.norm_transform(self.pad_func(img))
            if to_gpu:
                resize_img = resize_img.cuda()
                pad_img = pad_img.cuda()
            self.samples.append({
                'image': resize_img,
                'pad_image': pad_img,
                'path': img_path
            })

    def __len__(self):
        """Mengembalikan jumlah gambar dalam dataset."""
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Mengembalikan sample untuk inferensi."""
        return self.samples[idx]
    
###############################################################
def inference_openvino_modif(image_path, category):
    """Melakukan inferensi pada gambar dari path tertentu dan mengklasifikasikan langsung."""
    # Inisialisasi OpenVINO
    core = ov.Core()
    # root = os.getcwd()
    # print(f"path model= ckpt/openvino_models/{category}.xml")
    # print(f"path image= {image_path}")
    compiled_model = core.compile_model(f"ckpt/openvino_models/{category}.xml", "CPU")
    infer_request = compiled_model.create_infer_request()

    # Fungsi bantu untuk konversi tensor ke numpy
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # Membuat dataset dan dataloader
    test_set = MVTecLOCODataset(
        image_folder=image_path,
        image_size=256,
        to_gpu=False
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    results = []
    
    # Proses inferensi
    for sample in tqdm.tqdm(test_loader, desc="Mengklasifikasikan gambar"):
        image = sample['image']
        path = sample['path']  # Batch size 1, ambil path pertama
        
        # Persiapan input untuk OpenVINO
        input_tensor = ov.Tensor(array=to_numpy(image), shared_memory=False)
        infer_request.set_input_tensor(input_tensor)

        # Jalankan inferensi
        infer_request.start_async()
        infer_request.wait()

        # Ambil skor dari output
        output = infer_request.get_output_tensor()
        score = output.data[0]  # Skor anomali
        # print(f"{path}")
        # print(f"image score: {score}")
        
        # results.append((path, score))
        
    return score