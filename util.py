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
        obj_mask1 = target[:, :, :, 4] == 1

        confidence_loss = 0.0
        coord_loss = 0.0
        class_loss = 0.0
        confidence_loss_count = 0
        coord_loss_count = 0
        class_loss_count = 0
        # select responsible box
        # Apply lambda_coord and lambda_noobj weights as in YOLOv1 paper
        for b in range(batch_size):
            for cell_y in range(DATASET_CONFIG['grid_size']):
                for cell_x in range(DATASET_CONFIG['grid_size']):
                    if obj_mask1[b, cell_y, cell_x]:
                        # GT object exists in this cell
                        # GT object exist -> competition of box1, box2 (responsible box)
                        pred_box1 = pred[b, cell_y, cell_x, :5]
                        pred_box2 = pred[b, cell_y, cell_x, 5:10]
                        target_box = target[b, cell_y, cell_x, :5]
                        pred_converted_box1 = convert_coordinate_cell_to_image_tensor(pred_box1, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                        pred_converted_box2 = convert_coordinate_cell_to_image_tensor(pred_box2, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                        target_converted_box = convert_coordinate_cell_to_image_tensor(target_box, cell_y, cell_x, DATASET_CONFIG['grid_size'])
                        iou1 = iou_tensor(pred_converted_box1, target_converted_box)
                        iou2 = iou_tensor(pred_converted_box2, target_converted_box)
                        if iou1 > iou2:
                            # calc classes, coord loss box1 only
                            coord_loss += F.mse_loss(pred_box1[:2], target_box[:2])
                            coord_loss += F.mse_loss(torch.sqrt(pred_box1[2:4].clamp(min=1e-6)), torch.sqrt(target_box[2:4].clamp(min=1e-6)))
                            confidence_loss += F.mse_loss(pred_box1[4], torch.ones_like(pred_box1[4]))
                        else:
                            # box2 is responsible
                            coord_loss += F.mse_loss(pred_box2[:2], target_box[:2])
                            coord_loss += F.mse_loss(torch.sqrt(pred_box2[2:4].clamp(min=1e-6)), torch.sqrt(target_box[2:4].clamp(min=1e-6)))
                            confidence_loss += F.mse_loss(pred_box2[4], torch.ones_like(pred_box2[4]))
                        confidence_loss_count += 1 # for average loss
                        coord_loss_count += 1 # for average loss
                        # class loss
                        class_loss += F.mse_loss(pred[b, cell_y, cell_x, 10:], target[b, cell_y, cell_x, 10:])
                        class_loss_count += 1 # for average loss
                    else:
                        # only calculate confidence negative loss (no object)
                        # Both box1 and box2 confidence losses for noobj
                        pred_box1 = pred[b, cell_y, cell_x, :5]
                        pred_box2 = pred[b, cell_y, cell_x, 5:10]
                        confidence_loss += self.lambda_noobj * F.mse_loss(pred_box1[4], torch.zeros_like(pred_box1[4]))
                        confidence_loss += self.lambda_noobj * F.mse_loss(pred_box2[4], torch.zeros_like(pred_box2[4]))
                        confidence_loss_count += 2 # for average loss

        # Average the losses
        if confidence_loss_count > 0:
            confidence_loss = confidence_loss / confidence_loss_count
        if coord_loss_count > 0:
            coord_loss = coord_loss / coord_loss_count
        if class_loss_count > 0:
            class_loss = class_loss / class_loss_count

        # Apply lambda_coord and lambda_noobj as in YOLOv1 paper
        total_loss = self.lambda_coord * coord_loss + confidence_loss + class_loss
        return total_loss


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