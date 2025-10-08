"""
YOLO 프로젝트 설정 파일
"""
import torch

DATASET_CONFIG = {
    'input_channels': 3,  # 3: RGB, 1: Grayscale
    'image_width': 448,
    'image_height': 448,
    'grid_size': 7,
    'num_classes': 2,
    'class_names': ['truck', 'bus'],
}

MODEL_CONFIG = {
    'backbone': 'YOLOv1',
    'pretrained': False,
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.00001,
    'num_epochs': 100,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'lambda_coord': 5.0,
    'lambda_noobj': 0.5,
}

SAVE_CONFIG = {
    'save_path': './checkpoints',
    'project_name': 'vehicle_detection',
    'save_period': 1,
}

AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation': 10,  # degrees
    'brightness': 0.2,
    'contrast': 0.2,
}

PATH_CONFIG = {
    'data_dir': 'vehicle_dataset',
    'train_dir': 'vehicle_dataset/train',
    'val_dir': 'vehicle_dataset/validation',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
}

PREDICT_CONFIG = {
    'model_path': './checkpoints/vehicle_detection54/best.pt',
    'image_path': 'vehicle_dataset/validation',
    'save_path': './results',
    'confidence_threshold': 0.2,
    'iou_threshold': 0.5,
}

DEVICE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
}
