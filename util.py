import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DATASET_CONFIG

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
        target_box2 = target[:, :, :, 5:10]
        
        pred_class = pred[:, :, :, 10:]
        target_class = target[:, :, :, 10:]
        
        obj_mask1 = target_box1[:, :, :, 4] == 1
        obj_mask2 = target_box2[:, :, :, 4] == 1
        noobj_mask1 = target_box1[:, :, :, 4] == 0
        noobj_mask2 = target_box2[:, :, :, 4] == 0
        
        obj_mask_class = (obj_mask1 | obj_mask2)
        
        coord_loss = 0
        if obj_mask1.sum() > 0:
            xy_pred = pred_box1[:, :, :, :2][obj_mask1]
            xy_target = target_box1[:, :, :, :2][obj_mask1]
            xy_loss1 = F.mse_loss(xy_pred, xy_target, reduction='sum')
            
            if torch.isnan(xy_loss1):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
            wh_pred = pred_box1[:, :, :, 2:4][obj_mask1]
            wh_target = target_box1[:, :, :, 2:4][obj_mask1]
            
            wh_pred_clamped = torch.clamp(wh_pred, min=1e-8)
            wh_target_clamped = torch.clamp(wh_target, min=1e-8)
            
            wh_pred_sqrt = torch.sqrt(wh_pred_clamped)
            wh_target_sqrt = torch.sqrt(wh_target_clamped)
            
            wh_loss1 = F.mse_loss(wh_pred_sqrt, wh_target_sqrt, reduction='sum')
            
            if torch.isnan(wh_loss1):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
            coord_loss += xy_loss1 + wh_loss1
        
        if obj_mask2.sum() > 0:
            xy_pred = pred_box2[:, :, :, :2][obj_mask2]
            xy_target = target_box2[:, :, :, :2][obj_mask2]
            xy_loss2 = F.mse_loss(xy_pred, xy_target, reduction='sum')
            
            if torch.isnan(xy_loss2):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
            
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
        
        coord_loss *= self.lambda_coord

        conf_loss = 0
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
        
        class_loss = 0
        if obj_mask_class.sum() > 0:
            class_pred = pred_class[obj_mask_class]
            class_target = target_class[obj_mask_class]
            class_loss = F.mse_loss(class_pred, class_target, reduction='sum')
            
            if torch.isnan(class_loss):
                return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        total_loss = coord_loss + conf_loss + class_loss
        
        if torch.isnan(total_loss):
            return torch.tensor(0.0, requires_grad=True, device=pred.device)
        
        return total_loss / batch_size
    
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

    tp = np.zeros((len(arange(0.5, 0.95, 0.05)), num_classes), dtype=np.float32)
    fp = np.zeros((len(arange(0.5, 0.95, 0.05)), num_classes), dtype=np.float32)
    fn = np.zeros((len(arange(0.5, 0.95, 0.05)), num_classes), dtype=np.float32)
    precision = np.zeros((len(arange(0.5, 0.95, 0.05)), num_classes), dtype=np.float32)
    recall = np.zeros((len(arange(0.5, 0.95, 0.05)), num_classes), dtype=np.float32)
    f1_score = np.zeros((len(arange(0.5, 0.95, 0.05)), num_classes), dtype=np.float32)
    for i in range(batch_size):
        for cell_y in range(DATASET_CONFIG['grid_size']):
            for cell_x in range(DATASET_CONFIG['grid_size']):
                cell_pred = pred[i, cell_y, cell_x]
                cell_target = target[i, cell_y, cell_x]
                
                pred_box1, pred_box2 = cell_pred[:5], cell_pred[5:10]
                target_box1, target_box2 = cell_target[:5], cell_target[5:10]
                                
                pred_conf1, pred_conf2 = pred_box1[-1], pred_box2[-1]
                pred_has_obj1 = (pred_conf1 > conf_threshold)
                pred_has_obj2 = (pred_conf2 > conf_threshold)

                target_has_obj1 = (target_box1[-1] == 1)
                target_has_obj2 = (target_box2[-1] == 1)
                num_gt_obj = target_has_obj1 + target_has_obj2
                
                pred_cls = torch.argmax(cell_pred[10:])
                target_cls = torch.argmax(cell_target[10:])

                # case1 : pred #any, gt #0
                if (num_gt_obj == 0):
                    if (pred_has_obj1):
                        fp[:, pred_cls] += 1
                    if (pred_has_obj2):
                        fp[:, pred_cls] += 1

                # case2 : pred #0, gt #1
                # case3 : pred #1, gt #1
                # case4 : pred #2, gt #1
                if (not pred_has_obj1) and (not pred_has_obj2): # case 2
                    fn[:, target_cls] += 1
                if (pred_has_obj1 and not pred_has_obj2) or (not pred_has_obj1 and pred_has_obj2): # case 3
                    if (pred_has_obj1):
                        _iou = iou(pred_box1[:-1], target_box1[:-1])
                        if (_iou > iou_threshold) and (pred_cls == target_cls):
                            correct_max_idx = int((_iou - iou_threshold) / 0.05)
                            tp[:correct_max_idx + 1, target_cls] += 1
                            fp[correct_max_idx + 1:, target_cls] += 1
                        else:
                            fp[:, pred_cls] += 1
                    else:
                        _iou = iou(pred_box2[:-1], target_box2[:-1])
                        if (_iou > iou_threshold) and (pred_cls == target_cls):
                            correct_max_idx = int((_iou - iou_threshold) / 0.05)
                            tp[:correct_max_idx + 1, target_cls] += 1
                            fp[correct_max_idx + 1:, target_cls] += 1
                        else:
                            fp[:, pred_cls] += 1
                if (pred_has_obj1) and (pred_has_obj2): # case 4
                    if (pred_cls == target_cls):
                        _iou1 = iou(pred_box1[:-1], target_box1[:-1])
                        _iou2 = iou(pred_box2[:-1], target_box2[:-1])
                        if (_iou1 > iou_threshold) or (_iou2 > iou_threshold):
                            max_iou = max(_iou1, _iou2)
                            correct_max_idx = int((max_iou - iou_threshold) / 0.05)
                            tp[:correct_max_idx + 1, pred_cls] += 1
                            fp[correct_max_idx + 1:, pred_cls] += 1
                            if (_iou1 <= iou_threshold) or (_iou2 <= iou_threshold):
                                fp[:, pred_cls] += 1
                        else:
                            fp[:, pred_cls] += 2
                    else:
                        fp[:, pred_cls] += 2
    # tp fp fn done
    denominator_p = tp + fp
    precision = np.where(denominator_p > 0, tp / denominator_p, 0)
    denominator_r = tp + fn
    recall = np.where(denominator_r > 0, tp / denominator_r, 0)
    denominator_f = precision + recall
    f1_score = np.where(denominator_f > 0, 2 * precision * recall / denominator_f, 0)

    return precision, recall, f1_score

def calculate_map_metrics(pred, target, conf_threshold=0.5, iou_thresholds=[0.5]):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    num_classes = DATASET_CONFIG['num_classes']
    batch_size = pred.shape[0]
    
    predictions = {i: [] for i in range(num_classes)}
    ground_truths = {i: [] for i in range(num_classes)}
    
    for i in range(batch_size):
        for cell_y in range(DATASET_CONFIG['grid_size']):
            for cell_x in range(DATASET_CONFIG['grid_size']):
                cell_target = target[i, cell_y, cell_x]
                target_box1 = cell_target[:5]
                target_box2 = cell_target[5:10]
                
                if target_box1[4] == 1:  # GT box1
                    target_cls = np.argmax(cell_target[10:])
                    ground_truths[target_cls].append(target_box1[:4])
                
                if target_box2[4] == 1:  # GT box2
                    target_cls = np.argmax(cell_target[10:])
                    ground_truths[target_cls].append(target_box2[:4])
    
    for i in range(batch_size):
        for cell_y in range(DATASET_CONFIG['grid_size']):
            for cell_x in range(DATASET_CONFIG['grid_size']):
                cell_pred = pred[i, cell_y, cell_x]
                pred_box1 = cell_pred[:5]
                pred_box2 = cell_pred[5:10]
                
                if pred_box1[4] > conf_threshold:  # confidence threshold
                    pred_cls = np.argmax(cell_pred[10:])
                    predictions[pred_cls].append((pred_box1[4], pred_box1[:4]))
                
                if pred_box2[4] > conf_threshold:
                    pred_cls = np.argmax(cell_pred[10:])
                    predictions[pred_cls].append((pred_box2[4], pred_box2[:4]))
    
    maps = []
    for iou_threshold in iou_thresholds:
        class_aps = []
        
        for class_id in range(num_classes):
            if not predictions[class_id] or not ground_truths[class_id]:
                continue
            
            class_preds = sorted(predictions[class_id], key=lambda x: x[0], reverse=True)
            gt_boxes = ground_truths[class_id]
            
            tp = np.zeros(len(class_preds))
            fp = np.zeros(len(class_preds))
            gt_used = set()
            
            for idx, (conf, pred_box) in enumerate(class_preds):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in gt_used:
                        continue
                    iou_val = iou(pred_box, gt_box)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_gt_idx = gt_idx
                
                if best_iou > iou_threshold:
                    tp[idx] = 1
                    gt_used.add(best_gt_idx)
                else:
                    fp[idx] = 1
            
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(gt_boxes)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
            
            recalls = np.concatenate(([0.], recalls, [1.]))
            precisions = np.concatenate(([0.], precisions, [0.]))
            
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            
            i = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
            class_aps.append(ap)
        
        maps.append(np.mean(class_aps) if class_aps else 0)
    
    return maps[0] if len(maps) == 1 else maps
    