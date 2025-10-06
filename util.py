import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DATASET_CONFIG
import numpy as np

class YOLOv1Loss(nn.Module):
    def __init__(self, num_classes=80, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv1Loss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, pred, target):
        batch_size = pred.size(0)
        
        if torch.isnan(pred).any():
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        if torch.isnan(target).any():
            return torch.tensor(0.0, requires_grad=True, device=target.device)
        
        pred_box1 = pred[:, :, :, :5]
        target_box1 = target[:, :, :, :5]
        
        pred_box2 = pred[:, :, :, 5:10]
        
        pred_class = pred[:, :, :, 10:]
        target_class = target[:, :, :, 10:]
        
        obj_mask1 = target_box1[:, :, :, 4] == 1
        noobj_mask1 = target_box1[:, :, :, 4] == 0
        noobj_mask2 = torch.ones_like(target_box1[:, :, :, 4], dtype=torch.bool)
        
        obj_mask_class = obj_mask1
        
        coord_loss = 0
        if obj_mask1.sum() > 0:
            obj_indices = torch.where(obj_mask1)  # (batch_indices, cell_y_indices, cell_x_indices)
            
            iou1_values = []
            iou2_values = []
            responsible_mask1 = []
            responsible_mask2 = []
            for i in range(len(obj_indices[0])):
                batch_idx = obj_indices[0][i]
                cell_y = obj_indices[1][i]
                cell_x = obj_indices[2][i]
                
                pred_box1_cell = pred_box1[batch_idx, cell_y, cell_x, :4]
                pred_box2_cell = pred_box2[batch_idx, cell_y, cell_x, :4]
                target_box_cell = target_box1[batch_idx, cell_y, cell_x, :4]
                
                pred_box1_img = convert_coordinate_cell_to_image_tensor(pred_box1_cell, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                pred_box2_img = convert_coordinate_cell_to_image_tensor(pred_box2_cell, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                target_box_img = convert_coordinate_cell_to_image_tensor(target_box_cell, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                
                iou1 = iou_tensor(pred_box1_img, target_box_img)
                iou2 = iou_tensor(pred_box2_img, target_box_img)
                
                iou1_values.append(iou1)
                iou2_values.append(iou2)
                
                # Responsibility assignment
                if iou1 >= iou2:
                    responsible_mask1.append(True)
                    responsible_mask2.append(False)
                else:
                    responsible_mask1.append(False)
                    responsible_mask2.append(True)
            
            if any(responsible_mask1):
                responsible_indices1 = [i for i, mask in enumerate(responsible_mask1) if mask]
                
                xy_pred1 = torch.stack([pred_box1[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i], :2] 
                                      for i in responsible_indices1])
                xy_target1 = torch.stack([target_box1[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i], :2] 
                                        for i in responsible_indices1])
                xy_loss1 = F.mse_loss(xy_pred1, xy_target1, reduction='sum')
                
                wh_pred1 = torch.stack([pred_box1[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i], 2:4] 
                                      for i in responsible_indices1])
                wh_target1 = torch.stack([target_box1[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i], 2:4] 
                                        for i in responsible_indices1])
                
                wh_pred1_clamped = torch.clamp(wh_pred1, min=1e-8)
                wh_target1_clamped = torch.clamp(wh_target1, min=1e-8)
                
                wh_pred1_sqrt = torch.sqrt(wh_pred1_clamped)
                wh_target1_sqrt = torch.sqrt(wh_target1_clamped)
                
                wh_loss1 = F.mse_loss(wh_pred1_sqrt, wh_target1_sqrt, reduction='sum')
                
                coord_loss += xy_loss1 + wh_loss1
            
            if any(responsible_mask2):
                responsible_indices2 = [i for i, mask in enumerate(responsible_mask2) if mask]
                
                xy_pred2 = torch.stack([pred_box2[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i], :2] 
                                      for i in responsible_indices2])
                xy_target2 = torch.stack([target_box1[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i], :2] 
                                        for i in responsible_indices2])
                xy_loss2 = F.mse_loss(xy_pred2, xy_target2, reduction='sum')
                
                wh_pred2 = torch.stack([pred_box2[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i], 2:4] 
                                      for i in responsible_indices2])
                wh_target2 = torch.stack([target_box1[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i], 2:4] 
                                        for i in responsible_indices2])
                
                wh_pred2_clamped = torch.clamp(wh_pred2, min=1e-8)
                wh_target2_clamped = torch.clamp(wh_target2, min=1e-8)
                
                wh_pred2_sqrt = torch.sqrt(wh_pred2_clamped)
                wh_target2_sqrt = torch.sqrt(wh_target2_clamped)
                
                wh_loss2 = F.mse_loss(wh_pred2_sqrt, wh_target2_sqrt, reduction='sum')
                
                coord_loss += (xy_loss2 + wh_loss2)
        
        coord_loss *= self.lambda_coord
        
        conf_loss = 0
        if obj_mask1.sum() > 0:
            for i in range(len(obj_indices[0])):
                batch_idx = obj_indices[0][i]
                cell_y = obj_indices[1][i] 
                cell_x = obj_indices[2][i]
                
                iou1 = iou1_values[i]
                iou2 = iou2_values[i]
                
                if iou1 >= iou2:
                    conf_pred1 = pred_box1[batch_idx, cell_y, cell_x, 4]
                    conf_target1 = iou1.clone().detach()
                    conf_loss += F.mse_loss(conf_pred1, conf_target1)
                    
                    conf_pred2 = pred_box2[batch_idx, cell_y, cell_x, 4]
                    conf_target2 = torch.zeros_like(conf_pred2)
                    conf_loss += F.mse_loss(conf_pred2, conf_target2)
                else:
                    conf_pred2 = pred_box2[batch_idx, cell_y, cell_x, 4]
                    conf_target2 = iou2.clone().detach()
                    conf_loss += F.mse_loss(conf_pred2, conf_target2)
                    
                    conf_pred1 = pred_box1[batch_idx, cell_y, cell_x, 4]
                    conf_target1 = torch.zeros_like(conf_pred1)
                    conf_loss += F.mse_loss(conf_pred1, conf_target1)

        if noobj_mask1.sum() > 0:
            conf_pred1 = pred_box1[noobj_mask1][:, 4]
            conf_target1 = torch.zeros_like(conf_pred1)
            conf_loss += self.lambda_noobj * F.mse_loss(conf_pred1, conf_target1)
            
            conf_pred2 = pred_box2[noobj_mask2][:, 4]
            conf_target2 = torch.zeros_like(conf_pred2)
            conf_loss += self.lambda_noobj * F.mse_loss(conf_pred2, conf_target2)
        
        class_loss = 0
        if obj_mask1.sum() > 0:
            if any(responsible_mask1):
                responsible_indices1 = [i for i, mask in enumerate(responsible_mask1) if mask]
                
                class_pred1 = torch.stack([pred_class[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i]] 
                                        for i in responsible_indices1])
                class_target1 = torch.stack([target_class[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i]] 
                                        for i in responsible_indices1])
                class_loss += F.mse_loss(class_pred1, class_target1)
            
            if any(responsible_mask2):
                responsible_indices2 = [i for i, mask in enumerate(responsible_mask2) if mask]
                
                class_pred2 = torch.stack([pred_class[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i]] 
                                        for i in responsible_indices2])
                class_target2 = torch.stack([target_class[obj_indices[0][i], obj_indices[1][i], obj_indices[2][i]] 
                                        for i in responsible_indices2])
                class_loss += F.mse_loss(class_pred2, class_target2)
        
        total_loss = coord_loss + conf_loss + class_loss
        
        return total_loss / batch_size

def convert_coordinate_cell_to_image(box, cell_y, cell_x, grid_size):
    cx, cy = box[:2]
    new_cx = (1 / grid_size) * (cell_x + cx)
    new_cy = (1 / grid_size) * (cell_y + cy)
    new_box = box.copy()
    new_box[0] = new_cx
    new_box[1] = new_cy
    return new_box

def convert_coordinate_cell_to_image_tensor(box, cell_y, cell_x, grid_size):
    cx, cy = box[0], box[1]
    new_cx = (1 / grid_size) * (cell_x + cx)
    new_cy = (1 / grid_size) * (cell_y + cy)
    
    new_box = box.clone()
    new_box[0] = new_cx
    new_box[1] = new_cy
    return new_box

def iou_tensor(box1, box2):
    x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
    x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
    
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    inter_x_min = torch.max(x1_min, x2_min)
    inter_y_min = torch.max(y1_min, y2_min)
    inter_x_max = torch.min(x1_max, x2_max)
    inter_y_max = torch.min(y1_max, y2_max)
    
    inter_width = torch.clamp(inter_x_max - inter_x_min, min=0)
    inter_height = torch.clamp(inter_y_max - inter_y_min, min=0)
    inter_area = inter_width * inter_height
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-6)

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x1_min, y1_min = x1 - w1/2, y1 - h1/2
    x1_max, y1_max = x1 + w1/2, y1 + h1/2
    x2_min, y2_min = x2 - w2/2, y2 - h2/2
    x2_max, y2_max = x2 + w2/2, y2 + h2/2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_yolov1_metrics(pred, target, conf_threshold=0.5, iou_threshold=0.5):
    pred = pred.detach().cpu().numpy() # (b, grid, grid, 5*2 + num_classes)
    target = target.detach().cpu().numpy() # (b, grid, grid, 5*2 + num_classes)
    num_classes = DATASET_CONFIG['num_classes']
    batch_size = pred.shape[0]

    tp = np.zeros(DATASET_CONFIG['num_classes'])
    fp = np.zeros(DATASET_CONFIG['num_classes'])
    fn = np.zeros(DATASET_CONFIG['num_classes'])
    for i in range(batch_size):
        for cell_y in range(DATASET_CONFIG['grid_size']):
            for cell_x in range(DATASET_CONFIG['grid_size']):
                cell_pred = pred[i, cell_y, cell_x]
                cell_target = target[i, cell_y, cell_x]
                
                pred_box1, pred_box2 = cell_pred[:5], cell_pred[5:10]
                target_box1 = cell_target[:5]
                                
                pred_conf1, pred_conf2 = pred_box1[-1], pred_box2[-1]
                pred_box1 = convert_coordinate_cell_to_image(pred_box1, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                pred_box2 = convert_coordinate_cell_to_image(pred_box2, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                target_box1 = convert_coordinate_cell_to_image(target_box1, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                pred_has_obj1 = (pred_conf1 > conf_threshold)
                pred_has_obj2 = (pred_conf2 > conf_threshold)

                target_has_obj1 = (target_box1[-1] == 1)
                num_gt_obj = target_has_obj1
                
                pred_cls = np.argmax(cell_pred[10:])
                target_cls = np.argmax(cell_target[10:])

                # case1 : pred #?, gt #0
                if (num_gt_obj == 0):
                    if (pred_has_obj1):
                        fp[pred_cls] += 1
                    if (pred_has_obj2):
                        fp[pred_cls] += 1
                    continue

                # case2 : pred #0, gt #1
                # case3 : pred #1, gt #1
                # case4 : pred #2, gt #1
                if (not pred_has_obj1) and (not pred_has_obj2): # case 2
                    fn[target_cls] += 1
                if ((pred_has_obj1 and not pred_has_obj2) or (not pred_has_obj1 and pred_has_obj2)): # case 3
                    if (pred_has_obj1):
                        _iou = iou(pred_box1[:-1], target_box1[:-1])
                        if (_iou > iou_threshold) and (pred_cls == target_cls):
                            tp[target_cls] += 1
                        else:
                            fp[pred_cls] += 1
                    else:
                        _iou = iou(pred_box2[:-1], target_box1[:-1])
                        if (_iou > iou_threshold) and (pred_cls == target_cls):
                            tp[target_cls] += 1
                        else:
                            fp[pred_cls] += 1
                if (pred_has_obj1) and (pred_has_obj2): # case 4
                    if (pred_cls == target_cls):
                        _iou1 = iou(pred_box1[:-1], target_box1[:-1])
                        _iou2 = iou(pred_box2[:-1], target_box1[:-1])
                        max_iou = max(_iou1, _iou2)
                        if (max_iou > iou_threshold):
                            tp[target_cls] += 1
                            fp[pred_cls] += 1
                        else:
                            fp[pred_cls] += 2
                    else:
                        fp[pred_cls] += 2
    return tp, fp, fn