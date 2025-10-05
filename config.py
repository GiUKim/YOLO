
import torch

DATASET_CONFIG = {
    'input_channels': 3,  # 3: RGB, 1: Grayscale
    'image_width': 448,
    'image_height': 448,
    'grid_size': 7,
    'num_classes': 2,
    'class_names': [
        'bus',
        'truck'
    ],
    'num_boxes': 2,
}

MODEL_CONFIG = {
    'backbone': 'YOLOv1',
    'pretrained': False,
    'freeze_backbone': False,
}

TRAIN_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'lambda_coord': 5.0,
    'lambda_noobj': 0.5,
    'scheduler_type': 'cosine',  # 'step', 'cosine', 'none'
    'scheduler_params': {
        'step': {'step_size': 7, 'gamma': 0.1},
        'cosine': {'T_max': 100, 'eta_min': 1e-6},
        'none': {}
    }
}

AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation': 10,  # degrees
    'brightness': 0.2,
    'contrast': 0.2,
}

SAVE_CONFIG = {
    'save_dir': "./checkpoints",
    'project_name': "vehicle_detection",
    'save_period': 1,
}

PREDICT_CONFIG = {
    'model_path': "./checkpoints/vehicle_detection6/best.pt",
    'image_path': "./vehicle_dataset/validation",
    'save_path': "./results",
    'confidence_threshold': 0.2,
    'iou_threshold': 0.5,
}

PATH_CONFIG = {
    'data_dir': 'vehicle_dataset',
    'train_dir': 'vehicle_dataset/train',
    'val_dir': 'vehicle_dataset/validation',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
}

DEVICE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
}