import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from glob import glob
from config import *
from torchvision import transforms

def load_label_txt(file_path):
    try:
        label = np.loadtxt(file_path, dtype=np.float32)
        if label.ndim == 1:
            label = label.reshape(1, -1)
        return label
    except (ValueError, OSError):
        return np.array([]).reshape(0, 5)

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
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = glob(os.path.join(data_dir, '*.jpg'))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = image_path.replace('.jpg', '.txt')
        image = Image.open(image_path).convert('L' if DATASET_CONFIG['input_channels'] == 1 else 'RGB')
        label = load_label_txt(label_path)
        gt = YOLOv1_convert_label_to_grid(label, grid_size=DATASET_CONFIG['grid_size'], num_classes=DATASET_CONFIG['num_classes'])
        if self.transform:
            image = self.transform(image)
        return image, gt