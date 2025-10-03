import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv1Loss(nn.Module):
    """
    YOLOv1 Loss Function
    
    Args:
        pred: (batch_size, 7, 7, 5*2 + num_classes)
        target: (batch_size, 7, 7, 5*2 + num_classes)
    """
    def __init__(self, num_classes=80, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv1Loss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, pred, target):
        batch_size = pred.size(0)
        
        # Box 1: [x1, y1, w1, h1, conf1]
        pred_box1 = pred[:, :, :, :5]  # (batch, 7, 7, 5)
        target_box1 = target[:, :, :, :5]
        
        # Box 2: [x2, y2, w2, h2, conf2]
        pred_box2 = pred[:, :, :, 5:10]  # (batch, 7, 7, 5)
        target_box2 = target[:, :, :, 5:10]
        
        # Class prob
        pred_class = pred[:, :, :, 10:]  # (batch, 7, 7, num_classes)
        target_class = target[:, :, :, 10:]
        
        # Object mask
        obj_mask1 = target_box1[:, :, :, 4] == 1  # (batch, 7, 7)
        obj_mask2 = target_box2[:, :, :, 4] == 1
        noobj_mask1 = target_box1[:, :, :, 4] == 0
        noobj_mask2 = target_box2[:, :, :, 4] == 0
        
        # any object in this grid cell
        obj_mask_class = (obj_mask1 | obj_mask2)  # (batch, 7, 7)
        
        # 1. Coordinate Loss (only for boxes with objects)
        coord_loss = 0
        # Box 1 coordinate loss
        if obj_mask1.sum() > 0:
            # x, y loss
            xy_loss1 = F.mse_loss(pred_box1[:, :, :, :2][obj_mask1], 
                                 target_box1[:, :, :, :2][obj_mask1], reduction='sum')
            # w, h loss -[> sqrt]
            wh_loss1 = F.mse_loss(torch.sqrt(pred_box1[:, :, :, 2:4][obj_mask1] + 1e-8), 
                                 torch.sqrt(target_box1[:, :, :, 2:4][obj_mask1] + 1e-8), reduction='sum')
            coord_loss += xy_loss1 + wh_loss1
        
        # Box 2 coordinate loss
        if obj_mask2.sum() > 0:
            xy_loss2 = F.mse_loss(pred_box2[:, :, :, :2][obj_mask2], 
                                 target_box2[:, :, :, :2][obj_mask2], reduction='sum')
            
            wh_loss2 = F.mse_loss(torch.sqrt(pred_box2[:, :, :, 2:4][obj_mask2] + 1e-8), 
                                 torch.sqrt(target_box2[:, :, :, 2:4][obj_mask2] + 1e-8), reduction='sum')
            
            coord_loss += xy_loss2 + wh_loss2

        coord_loss *= self.lambda_coord
        
        # 2. Confidence Loss
        conf_loss = 0        
        # Object confidence loss
        if obj_mask1.sum() > 0:
            conf_loss += F.mse_loss(pred_box1[:, :, :, 4][obj_mask1], 
                                   target_box1[:, :, :, 4][obj_mask1], reduction='sum')
        
        if obj_mask2.sum() > 0:
            conf_loss += F.mse_loss(pred_box2[:, :, :, 4][obj_mask2], 
                                   target_box2[:, :, :, 4][obj_mask2], reduction='sum')
        
        # No-object confidence loss
        if noobj_mask1.sum() > 0:
            conf_loss += self.lambda_noobj * F.mse_loss(pred_box1[:, :, :, 4][noobj_mask1], 
                                                       target_box1[:, :, :, 4][noobj_mask1], reduction='sum')
        
        if noobj_mask2.sum() > 0:
            conf_loss += self.lambda_noobj * F.mse_loss(pred_box2[:, :, :, 4][noobj_mask2], 
                                                       target_box2[:, :, :, 4][noobj_mask2], reduction='sum')
        
        # 3. Class Loss
        class_loss = 0
        if obj_mask_class.sum() > 0:
            class_loss = F.mse_loss(pred_class[obj_mask_class], target_class[obj_mask_class], reduction='sum')
        
        # Total loss
        total_loss = coord_loss + conf_loss + class_loss
        
        return total_loss / batch_size  # Average over batch
