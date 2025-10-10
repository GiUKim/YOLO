import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from glob import glob
from config import *
from torchvision import transforms
import random

def load_label_txt(file_path):
    try:
        label = np.loadtxt(file_path, dtype=np.float32)
        if label.ndim == 1:
            label = label.reshape(1, -1)
        return label
    except (ValueError, OSError):
        return np.array([]).reshape(0, 5)

class YOLOv1Augmentation:
    
    def __init__(self, augmentation_config):
        self.horizontal_flip = augmentation_config.get('horizontal_flip', False)
        self.vertical_flip = augmentation_config.get('vertical_flip', False)
        self.brightness = augmentation_config.get('brightness', -1)
        self.contrast = augmentation_config.get('contrast', -1)
    
    def apply_flip_augmentation(self, image, label):
        applied_transforms = []
        
        if self.horizontal_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # cx = 1 - cx
            if label.shape[0] > 0:
                label[:, 1] = 1.0 - label[:, 1]
            applied_transforms.append('horizontal_flip')
        
        if self.vertical_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            # cy = 1 - cy
            if label.shape[0] > 0:
                label[:, 2] = 1.0 - label[:, 2]
            applied_transforms.append('vertical_flip')
        
        return image, label, applied_transforms
    
    def apply_brightness_contrast(self, image):
        if self.brightness >= 0:
            brightness_factor = random.uniform(0.0, self.brightness)
            enhancer = transforms.ColorJitter(brightness=brightness_factor)
            image = enhancer(image)
        
        if self.contrast >= 0:
            contrast_factor = random.uniform(0.0, self.contrast)
            enhancer = transforms.ColorJitter(contrast=contrast_factor)
            image = enhancer(image)
        
        return image
    
    def __call__(self, image, label):
        image, label, applied_transforms = self.apply_flip_augmentation(image, label)
        
        image = self.apply_brightness_contrast(image)
        
        return image, label

# input label shape: (?, 5) cls, cx, cy, w, h
# output label shape: (grid_size, grid_size, 5*2 + num_classes)
def YOLOv1_convert_label_to_grid(label, grid_size=7, num_classes=80):
    grid = torch.zeros(grid_size, grid_size, 5*2 + num_classes)
    if label.shape[0] == 0: # background img
        return grid

    for i in range(label.shape[0]):
        cls, cx, cy, w, h = label[i]
        grid_x = int(cx * grid_size)
        grid_y = int(cy * grid_size)
        cell_x = cx * grid_size - grid_x
        cell_y = cy * grid_size - grid_y
        
        if grid[grid_y, grid_x, 4] == 0:
            grid[grid_y, grid_x, :5] = torch.clamp(torch.tensor([cell_x, cell_y, w, h, 1]), 0.0, 1.0)
        
        grid[grid_y, grid_x, 10 + int(cls)] = 1.0
    return grid

class YOLOv1Dataset(Dataset):
    def __init__(self, data_dir, transform=None, use_augmentation=False):
        self.data_dir = data_dir
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.image_files = glob(os.path.join(data_dir, '*.jpg'))
        
        if self.use_augmentation:
            self.augmentation = YOLOv1Augmentation(AUGMENTATION_CONFIG)
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = image_path.replace('.jpg', '.txt')
        image = Image.open(image_path).convert('L' if DATASET_CONFIG['input_channels'] == 1 else 'RGB')
        label = load_label_txt(label_path)
        
        if self.augmentation:
            image, label = self.augmentation(image, label)
        
        gt = YOLOv1_convert_label_to_grid(label, grid_size=DATASET_CONFIG['grid_size'], num_classes=DATASET_CONFIG['num_classes'])
        if self.transform:
            image = self.transform(image)
        
        return image, gt

def create_dataloader(data_dir, batch_size=32, shuffle=True, use_augmentation=False, num_workers=4):
    
    transform = transforms.Compose([
        transforms.Resize((DATASET_CONFIG['image_height'], DATASET_CONFIG['image_width'])),
        transforms.ToTensor(),
    ])
    
    dataset = YOLOv1Dataset(
        data_dir=data_dir,
        transform=transform,
        use_augmentation=use_augmentation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader