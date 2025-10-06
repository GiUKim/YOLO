from util import YOLOv1Loss, calculate_metrics
from model import YOLOv1
from dataset import YOLOv1Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
from config import *

def get_optimizer(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_CONFIG['learning_rate'])
    return optimizer

def get_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return scheduler

def get_YOLOv1_criterion(num_classes, lambda_coord=5.0, lambda_noobj=0.5):
    criterion = YOLOv1Loss(num_classes, lambda_coord, lambda_noobj)
    return criterion

def get_model(ch=3, num_classes=80):
    model = YOLOv1(ch, num_classes)
    model = model.to(DEVICE_CONFIG['device'])
    return model

def get_transform():
    train_transform = transforms.Compose([
        transforms.Resize((DATASET_CONFIG['image_height'], DATASET_CONFIG['image_width'])),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((DATASET_CONFIG['image_height'], DATASET_CONFIG['image_width'])),
        transforms.ToTensor(),
    ])
    return train_transform, val_transform

def get_dataloader():
    train_transform, val_transform = get_transform()
    dataset_train = YOLOv1Dataset(PATH_CONFIG['train_dir'], transform=train_transform)
    dataset_val = YOLOv1Dataset(PATH_CONFIG['val_dir'], transform=val_transform)
    dataloader_train = DataLoader(dataset_train, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True, num_workers=DEVICE_CONFIG['num_workers'])
    dataloader_val = DataLoader(dataset_val, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False, num_workers=DEVICE_CONFIG['num_workers'])
    return dataloader_train, dataloader_val

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, targets in pbar:
            images = images.to(DEVICE_CONFIG['device'])
            targets = targets.to(DEVICE_CONFIG['device'])
            
            outputs = model(images)
            calculate_metrics(outputs, targets)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def train(model, optimizer, scheduler, criterion, dataloader, val_dataloader):
    print("🚀 Starting YOLOv1 Training")
    print("=" * 60)
    
    for epoch in range(TRAIN_CONFIG['num_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['num_epochs']}")
        for images, targets in pbar:
            images = images.to(DEVICE_CONFIG['device'])
            targets = targets.to(DEVICE_CONFIG['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            calculate_metrics(outputs, targets)
            loss = criterion(outputs, targets)
            
            # NaN 체크
            if torch.isnan(loss):
                print(f"⚠️  NaN loss detected at epoch {epoch+1}, batch {num_batches+1}")
                print(f"   Output range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"   Target range: [{targets.min().item():.4f}, {targets.max().item():.4f}]")
                break
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{train_loss/num_batches:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Validation
        if not torch.isnan(loss):
            val_loss = evaluate(model, criterion, val_dataloader)
            print(f"📊 Epoch {epoch+1} - Train Loss: {train_loss/num_batches:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"❌ Epoch {epoch+1} failed due to NaN loss")
            break
        
        scheduler.step()
    
    print("✅ Training completed!")

def main():
    model = get_model()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = get_YOLOv1_criterion(DATASET_CONFIG['num_classes'])
    train_dataloader, val_dataloader = get_dataloader()
    train(model, optimizer, scheduler, criterion, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()