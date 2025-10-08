import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DATASET_CONFIG, TRAIN_CONFIG
import numpy as np

class YOLOv1Loss(nn.Module):
    def __init__(self, num_classes=80, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv1Loss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, pred, target):
        batch_size, grid_size = pred.size(0), DATASET_CONFIG['grid_size']
        device = pred.device
        
        pred_box1 = pred[:, :, :, :5]  # (b, g, g, 5)
        pred_box2 = pred[:, :, :, 5:10]  # (b, g, g, 5)
        pred_classes = pred[:, :, :, 10:]  # (b, g, g, cls)
        
        target_box = target[:, :, :, :5]  # (b, g, g, 5)
        target_classes = target[:, :, :, 10:]  # (b, g, g, cls)
        
        obj_mask = target[:, :, :, 4] == 1  # (b, g, g)
        noobj_mask = ~obj_mask  # (b, g, g)
        
        pred_box1_img = self.convert_cell_coords_to_image_coords(pred_box1, grid_size)
        pred_box2_img = self.convert_cell_coords_to_image_coords(pred_box2, grid_size)
        target_box_img = self.convert_cell_coords_to_image_coords(target_box, grid_size)
        
        iou1 = self.iou_vectorized(pred_box1_img, target_box_img)  # (b, g, g)
        iou2 = self.iou_vectorized(pred_box2_img, target_box_img)  # (b, g, g)
        
        box1_responsible = iou1 > iou2  # (b, g, g)
        box2_responsible = ~box1_responsible  # (b, g, g)
        
        box1_responsible = box1_responsible & obj_mask
        box2_responsible = box2_responsible & obj_mask
        
        # 1. Coordinate Loss
        coord_loss = self.yolov1_coord_loss(
            pred_box1, pred_box2, target_box, 
            box1_responsible, box2_responsible
        )
        
        # 2. Confidence Loss
        confidence_loss = self.yolov1_conf_loss(
            pred_box1, pred_box2, iou1, iou2,
            box1_responsible, box2_responsible, noobj_mask
        )
        
        # 3. Class Loss
        class_loss = self.yolov1_class_loss(
            pred_classes, target_classes, obj_mask
        )
        
        # Total loss
        total_loss = coord_loss + confidence_loss + class_loss
        return total_loss / batch_size
    
    def convert_cell_coords_to_image_coords(self, boxes, grid_size):
        batch_size, grid_h, grid_w = boxes.shape[:3]
        device = boxes.device
        
        cell_x = torch.arange(grid_w, device=device).float().view(1, 1, grid_w, 1)
        cell_y = torch.arange(grid_h, device=device).float().view(1, grid_h, 1, 1)
        
        cx, cy = boxes[:, :, :, 0:1], boxes[:, :, :, 1:2]
        w, h, conf = boxes[:, :, :, 2:3], boxes[:, :, :, 3:4], boxes[:, :, :, 4:5]
        
        new_cx = (1 / grid_size) * (cell_x + cx)
        new_cy = (1 / grid_size) * (cell_y + cy)
        
        return torch.cat([new_cx, new_cy, w, h, conf], dim=-1)
    
    def iou_vectorized(self, box1, box2):
        x1, y1, w1, h1 = box1[:, :, :, 0], box1[:, :, :, 1], box1[:, :, :, 2], box1[:, :, :, 3]
        x2, y2, w2, h2 = box2[:, :, :, 0], box2[:, :, :, 1], box2[:, :, :, 2], box2[:, :, :, 3]
        
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
    
    def yolov1_coord_loss(self, pred_box1, pred_box2, target_box, 
                                     box1_responsible, box2_responsible):
        # YOLOv1 coordinate loss: λ_coord * [(x_pred - x_gt)² + (y_pred - y_gt)² + (√w_pred - √w_gt)² + (√h_pred - √h_gt)²]
        device = pred_box1.device
        coord_loss_box1 = torch.tensor(0.0, device=device)
        coord_loss_box2 = torch.tensor(0.0, device=device)
        
        if box1_responsible.any():
            # x, y
            xy_loss_box1 = ((pred_box1[:, :, :, 0] - target_box[:, :, :, 0])**2 + 
                           (pred_box1[:, :, :, 1] - target_box[:, :, :, 1])**2) * box1_responsible.float()
            
            # w, h (with sqrt)
            w_loss_box1 = ((torch.sqrt(pred_box1[:, :, :, 2].clamp(min=1e-6)) - 
                           torch.sqrt(target_box[:, :, :, 2].clamp(min=1e-6)))**2) * box1_responsible.float()
            h_loss_box1 = ((torch.sqrt(pred_box1[:, :, :, 3].clamp(min=1e-6)) - 
                           torch.sqrt(target_box[:, :, :, 3].clamp(min=1e-6)))**2) * box1_responsible.float()
            
            coord_loss_box1 = (xy_loss_box1 + w_loss_box1 + h_loss_box1).sum()
        
        if box2_responsible.any():
            # x, y
            xy_loss_box2 = ((pred_box2[:, :, :, 0] - target_box[:, :, :, 0])**2 + 
                           (pred_box2[:, :, :, 1] - target_box[:, :, :, 1])**2) * box2_responsible.float()
            
            # w, h (with sqrt)
            w_loss_box2 = ((torch.sqrt(pred_box2[:, :, :, 2].clamp(min=1e-6)) - 
                           torch.sqrt(target_box[:, :, :, 2].clamp(min=1e-6)))**2) * box2_responsible.float()
            h_loss_box2 = ((torch.sqrt(pred_box2[:, :, :, 3].clamp(min=1e-6)) - 
                           torch.sqrt(target_box[:, :, :, 3].clamp(min=1e-6)))**2) * box2_responsible.float()
            
            coord_loss_box2 = (xy_loss_box2 + w_loss_box2 + h_loss_box2).sum()
        
        return self.lambda_coord * (coord_loss_box1 + coord_loss_box2)
    
    def yolov1_conf_loss(self, pred_box1, pred_box2, iou1, iou2,
                                          box1_responsible, box2_responsible, noobj_mask):
        device = pred_box1.device
        confidence_loss = torch.tensor(0.0, device=device)
        
        if box1_responsible.any():
            confidence_loss += ((pred_box1[:, :, :, 4] - iou1)**2 * box1_responsible.float()).sum()
        
        if box2_responsible.any():
            confidence_loss += ((pred_box2[:, :, :, 4] - iou2)**2 * box2_responsible.float()).sum()
        
        # No obj
        if noobj_mask.any():
            confidence_loss += self.lambda_noobj * ((pred_box1[:, :, :, 4]**2 * noobj_mask.float()).sum() + 
                                                   (pred_box2[:, :, :, 4]**2 * noobj_mask.float()).sum())
        
        return confidence_loss
    
    def yolov1_class_loss(self, pred_classes, target_classes, obj_mask):
        if not obj_mask.any():
            return torch.tensor(0.0, device=pred_classes.device)
        
        # mse loss
        class_diff = (pred_classes - target_classes)**2
        class_loss = (class_diff * obj_mask.unsqueeze(-1).float()).sum()
        
        return class_loss

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

def calculate_yolov1_metrics(pred, target, conf_threshold=0.15, iou_threshold=0.5):
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

def NMS(box_list, iou_threshold):
    # cls, cx, cy, w, h, conf
    box_list = sorted(box_list, key=lambda x: x[-1], reverse=True)
    rm_list = []
    for i in range(len(box_list)):
        for j in range(i + 1, len(box_list)):
            if (box_list[i][0] == box_list[j][0]) and (iou(box_list[i][1:-1], box_list[j][1:-1]) > iou_threshold):
                rm_list.append(j)
    new_box_list = []
    for i in range(len(box_list)):
        if i not in rm_list:
            new_box_list.append(box_list[i])
    return new_box_list
