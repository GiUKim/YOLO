from util import YOLOv1Loss, calculate_yolov1_metrics
from model import YOLOv1
from dataset import YOLOv1Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import numpy as np
import os
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

def get_model(ch, num_classes):
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

def create_project_dir():
    save_path = SAVE_CONFIG['save_path']
    project_name = SAVE_CONFIG['project_name']
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    project_dir = None
    for i in range(1000):
        candidate_dir = os.path.join(save_path, f"{project_name}{i}")
        if not os.path.exists(candidate_dir):
            project_dir = candidate_dir
            break
    
    if project_dir is None:
        raise ValueError(f"Could not find available project directory in {save_path}")
    
    os.makedirs(project_dir)
    print(f"📁 Created project directory: {project_dir}")
    
    return project_dir

def save_model(model, optimizer, epoch, val_loss, project_dir, is_best=False):
    save_period = SAVE_CONFIG['save_period']
    
    if is_best:
        best_path = os.path.join(project_dir, 'best.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, best_path)
        print(f"💾 Saved best model: {best_path}")
    
    if save_period > 0 and (epoch + 1) % save_period == 0:
        checkpoint_path = os.path.join(project_dir, f'epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"💾 Saved checkpoint: {checkpoint_path}")

def count_instances_in_dataset(dataloader):
    instance_counts = [0] * DATASET_CONFIG['num_classes']
    
    for images, targets in dataloader:
        targets = targets.detach().cpu().numpy()
        batch_size = targets.shape[0]
        
        for i in range(batch_size):
            for cell_y in range(DATASET_CONFIG['grid_size']):
                for cell_x in range(DATASET_CONFIG['grid_size']):
                    cell_target = targets[i, cell_y, cell_x]
                    target_box1 = cell_target[:5]
                    target_has_obj = (target_box1[-1] == 1)
                    
                    if target_has_obj:
                        target_cls = np.argmax(cell_target[10:])
                        instance_counts[target_cls] += 1
    
    return instance_counts

def print_evaluation_table(tp, fp, fn, instance_counts):
    class_names = DATASET_CONFIG['class_names']
    num_classes = DATASET_CONFIG['num_classes']
    
    precisions = []
    recalls = []
    f1_scores = []
    
    for i in range(num_classes):
        precision = tp[i] / (tp[i] + fp[i] + 1e-6)
        recall = tp[i] / (tp[i] + fn[i] + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)
    
    total_tp = sum(tp)
    total_fp = sum(fp)
    total_fn = sum(fn)
    total_instances = sum(instance_counts)
    
    total_precision = total_tp / (total_tp + total_fp + 1e-6)
    total_recall = total_tp / (total_tp + total_fn + 1e-6)
    total_f1_score = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-6)
    
    print("\n" + "="*120)
    print("VALIDATION EVALUATION RESULTS")
    print("="*120)
    print(f"{'Class':<15} {'Instance':<10} {'TP':<8} {'FP':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*120)
    
    for i in range(num_classes):
        print(f"{i}: {class_names[i]:<12} {int(instance_counts[i]):<10} {int(tp[i]):<8} {int(fp[i]):<8} {int(fn[i]):<8} "
              f"{precisions[i]:<12.4f} {recalls[i]:<12.4f} {f1_scores[i]:<12.4f}")
    
    print("-"*120)
    print(f"{'TOTAL':<15} {int(total_instances):<10} {int(total_tp):<8} {int(total_fp):<8} {int(total_fn):<8} "
          f"{total_precision:<12.4f} {total_recall:<12.4f} {total_f1_score:<12.4f}")
    print("="*120)

def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    tp = np.zeros(DATASET_CONFIG['num_classes'])
    fp = np.zeros(DATASET_CONFIG['num_classes'])
    fn = np.zeros(DATASET_CONFIG['num_classes'])
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for images, targets in pbar:
            images = images.to(DEVICE_CONFIG['device'])
            targets = targets.to(DEVICE_CONFIG['device'])
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            _tp, _fp, _fn = calculate_yolov1_metrics(outputs, targets)
            tp += _tp
            fp += _fp
            fn += _fn
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    instance_counts = count_instances_in_dataset(dataloader)
    
    print_evaluation_table(tp, fp, fn, instance_counts)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

def train(model, optimizer, scheduler, criterion, dataloader, val_dataloader):
    print("🚀 Starting YOLOv1 Training")
    print("=" * 60)
    
    project_dir = create_project_dir()
    
    best_val_loss = float('inf')
    
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
            
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"🎯 New best validation loss: {val_loss:.4f}")
            
            save_model(model, optimizer, epoch, val_loss, project_dir, is_best=is_best)
        else:
            print(f"❌ Epoch {epoch+1} failed due to NaN loss")
            break
        
        scheduler.step()
    
    print("✅ Training completed!")

def main():
    model = get_model(ch=DATASET_CONFIG['input_channels'], num_classes=DATASET_CONFIG['num_classes'])
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = get_YOLOv1_criterion(DATASET_CONFIG['num_classes'])
    train_dataloader, val_dataloader = get_dataloader()
    train(model, optimizer, scheduler, criterion, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()