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
        
        # 1. 입력 검증
        if torch.isnan(pred).any():
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        if torch.isnan(target).any():
            return torch.tensor(0.0, requires_grad=True, device=target.device)
        
        # 2. 텐서 분할
        pred_box1 = pred[:, :, :, :5]
        target_box1 = target[:, :, :, :5]
        
        pred_box2 = pred[:, :, :, 5:10]
        target_box2 = target[:, :, :, 5:10]
        
        pred_class = pred[:, :, :, 10:]
        target_class = target[:, :, :, 10:]
        
        # 3. 마스크 생성
        obj_mask1 = target_box1[:, :, :, 4] == 1
        obj_mask2 = target_box2[:, :, :, 4] == 1
        noobj_mask1 = target_box1[:, :, :, 4] == 0
        noobj_mask2 = target_box2[:, :, :, 4] == 0
        
        obj_mask_class = (obj_mask1 | obj_mask2)
        
        # 4. 좌표 손실 계산 (Box 1)
        coord_loss = 0
        if obj_mask1.sum() > 0:
            # XY 손실
            xy_pred = pred_box1[:, :, :, :2][obj_mask1]
            xy_target = target_box1[:, :, :, :2][obj_mask1]
            xy_loss1 = F.mse_loss(xy_pred, xy_target, reduction='sum')
            
            if torch.isnan(xy_loss1):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
            # WH 손실 (sqrt 안전성 확보)
            wh_pred = pred_box1[:, :, :, 2:4][obj_mask1]
            wh_target = target_box1[:, :, :, 2:4][obj_mask1]
            
            # 음수 값 클램핑
            wh_pred_clamped = torch.clamp(wh_pred, min=1e-8)
            wh_target_clamped = torch.clamp(wh_target, min=1e-8)
            
            wh_pred_sqrt = torch.sqrt(wh_pred_clamped)
            wh_target_sqrt = torch.sqrt(wh_target_clamped)
            
            wh_loss1 = F.mse_loss(wh_pred_sqrt, wh_target_sqrt, reduction='sum')
            
            if torch.isnan(wh_loss1):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
            coord_loss += xy_loss1 + wh_loss1
        
        # 5. 좌표 손실 계산 (Box 2)
        if obj_mask2.sum() > 0:
            # XY 손실
            xy_pred = pred_box2[:, :, :, :2][obj_mask2]
            xy_target = target_box2[:, :, :, :2][obj_mask2]
            xy_loss2 = F.mse_loss(xy_pred, xy_target, reduction='sum')
            
            if torch.isnan(xy_loss2):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
            # WH 손실
            wh_pred = pred_box2[:, :, :, 2:4][obj_mask2]
            wh_target = target_box2[:, :, :, 2:4][obj_mask2]
            
            wh_pred_clamped = torch.clamp(wh_pred, min=1e-8)
            wh_target_clamped = torch.clamp(wh_target, min=1e-8)
            
            wh_pred_sqrt = torch.sqrt(wh_pred_clamped)
            wh_target_sqrt = torch.sqrt(wh_target_clamped)
            
            wh_loss2 = F.mse_loss(wh_pred_sqrt, wh_target_sqrt, reduction='sum')
            
            if torch.isnan(wh_loss2):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
            coord_loss += xy_loss2 + wh_loss2
        
        # 6. 좌표 손실에 가중치 적용
        coord_loss *= self.lambda_coord
        
        # 7. 신뢰도 손실 계산
        conf_loss = 0
        
        # Object가 있는 경우
        if obj_mask1.sum() > 0:
            conf_pred = pred_box1[:, :, :, 4][obj_mask1]
            conf_target = target_box1[:, :, :, 4][obj_mask1]
            conf_loss += F.mse_loss(conf_pred, conf_target, reduction='sum')
            
            if torch.isnan(conf_loss):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        if obj_mask2.sum() > 0:
            conf_pred = pred_box2[:, :, :, 4][obj_mask2]
            conf_target = target_box2[:, :, :, 4][obj_mask2]
            conf_loss += F.mse_loss(conf_pred, conf_target, reduction='sum')
            
            if torch.isnan(conf_loss):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        # No-object가 있는 경우
        if noobj_mask1.sum() > 0:
            conf_pred = pred_box1[:, :, :, 4][noobj_mask1]
            conf_target = target_box1[:, :, :, 4][noobj_mask1]
            noobj_loss1 = F.mse_loss(conf_pred, conf_target, reduction='sum')
            conf_loss += self.lambda_noobj * noobj_loss1
            
            if torch.isnan(noobj_loss1):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        if noobj_mask2.sum() > 0:
            conf_pred = pred_box2[:, :, :, 4][noobj_mask2]
            conf_target = target_box2[:, :, :, 4][noobj_mask2]
            noobj_loss2 = F.mse_loss(conf_pred, conf_target, reduction='sum')
            conf_loss += self.lambda_noobj * noobj_loss2
            
            if torch.isnan(noobj_loss2):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        # 8. 클래스 손실 계산
        class_loss = 0
        if obj_mask_class.sum() > 0:
            class_pred = pred_class[obj_mask_class]
            class_target = target_class[obj_mask_class]
            class_loss = F.mse_loss(class_pred, class_target, reduction='sum')
            
            if torch.isnan(class_loss):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        # 9. 총 손실 계산
        total_loss = coord_loss + conf_loss + class_loss
        
        if torch.isnan(total_loss):
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        return total_loss / batch_size

def convert_coordinate_cell_to_image(box, cell_y, cell_x, grid_size):
    cx, cy = box[:2]
    new_cx = (1 / grid_size) * (cell_x + cx)
    new_cy = (1 / grid_size) * (cell_y + cy)
    new_box = box.copy()
    new_box[0] = new_cx
    new_box[1] = new_cy
    return new_box

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
                pred_box1[:-1] = convert_coordinate_cell_to_image(pred_box1[:-1], cell_y, cell_x, DATASET_CONFIG['grid_size'])
                pred_box2[:-1] = convert_coordinate_cell_to_image(pred_box2[:-1], cell_y, cell_x, DATASET_CONFIG['grid_size'])
                target_box1[:-1] = convert_coordinate_cell_to_image(target_box1[:-1], cell_y, cell_x, DATASET_CONFIG['grid_size'])
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

                # case2 : pred #0, gt #1
                # case3 : pred #1, gt #1
                # case4 : pred #2, gt #1
                if (not pred_has_obj1) and (not pred_has_obj2): # case 2
                    fn[target_cls] += 1
                if (pred_has_obj1 and not pred_has_obj2) or (not pred_has_obj1 and pred_has_obj2): # case 3
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
                        fp[pred_cls] += 1
                        _iou1 = iou(pred_box1[:-1], target_box1[:-1])
                        _iou2 = iou(pred_box2[:-1], target_box1[:-1])
                        max_iou = max(_iou1, _iou2)
                        if (max_iou > iou_threshold):
                            tp[target_cls] += 1
                        else:
                            fp[pred_cls] += 1
                    else:
                        fp[pred_cls] += 2
    return tp, fp, fn