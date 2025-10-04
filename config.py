"""
YOLO 프로젝트 설정 파일
"""
import torch

# 데이터셋 설정
DATASET_CONFIG = {
    'input_channels': 3,  # 3: RGB, 1: Grayscale
    'image_width': 448,
    'image_height': 448,
    'grid_size': 7,
    'num_classes': 80,
    'num_boxes': 2,
}

# 모델 설정
MODEL_CONFIG = {
    'backbone': 'YOLOv1',
    'pretrained': False,
    'freeze_backbone': False,
}

# 훈련 설정
TRAIN_CONFIG = {
    'batch_size': 32,
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'lambda_coord': 5.0,
    'lambda_noobj': 0.5,
}

# 데이터 증강 설정
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'vertical_flip': False,
    'rotation': 10,  # degrees
    'brightness': 0.2,
    'contrast': 0.2,
}

# 경로 설정
PATH_CONFIG = {
    'data_dir': 'yolo_dataset',
    'train_dir': 'yolo_dataset/train',
    'val_dir': 'yolo_dataset/validation',
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
}

# 디바이스 설정
DEVICE_CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
}

# 유틸리티 함수
def get_input_channels():
    """입력 채널 수 반환"""
    return DATASET_CONFIG['input_channels']

def is_rgb():
    """RGB 모드인지 확인"""
    return DATASET_CONFIG['input_channels'] == 3

def is_grayscale():
    """그레이스케일 모드인지 확인"""
    return DATASET_CONFIG['input_channels'] == 1

def get_image_mode():
    """PIL 이미지 모드 반환"""
    return 'RGB' if is_rgb() else 'L'
