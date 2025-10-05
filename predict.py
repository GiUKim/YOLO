import numpy as np
from model import YOLOv1
import torch
from torchvision import transforms
from config import *
import os
from glob import glob
from tqdm import tqdm
import time
import cv2
from PIL import Image, ImageDraw, ImageFont
from train import get_model
from util import convert_coordinate_cell_to_image, NMS
import random

def generate_colors(num_classes):
    colors = []
    for _ in range(num_classes):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors

def draw_bbox(image, box_list, colors, class_names):
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for box in box_list:
        cls, cx, cy, w, h, conf = box
        cls = int(cls)
        
        img_width, img_height = image.size
        x1 = int((cx - w/2) * img_width)
        y1 = int((cy - h/2) * img_height)
        x2 = int((cx + w/2) * img_width)
        y2 = int((cy + h/2) * img_height)
        
        x1 = max(0, min(x1, img_width-1))
        y1 = max(0, min(y1, img_height-1))
        x2 = max(0, min(x2, img_width-1))
        y2 = max(0, min(y2, img_height-1))
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            continue
        
        color = colors[cls]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        text = f"{cls}: {class_name} ({conf:.2f})"
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if y1 - text_height - 5 < 0:
            text_y = y2 + 5
        else:
            text_y = y1 - text_height - 5
        
        draw.rectangle([x1, text_y, x1 + text_width + 10, text_y + text_height + 5], 
                      fill=color, outline=color)
        
        draw.text((x1 + 5, text_y + 2), text, fill=(255, 255, 255), font=font)
    
    return image

def save_results(image, box_list, image_path, save_path):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'labels'), exist_ok=True)
    
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    image.save(os.path.join(save_path, filename))
    
    label_path = os.path.join(save_path, 'labels', f"{name_without_ext}.txt")
    with open(label_path, 'w') as f:
        for box in box_list:
            cls, cx, cy, w, h, conf = box
            f.write(f"{int(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

def main():
    model = get_model(DATASET_CONFIG['input_channels'], DATASET_CONFIG['num_classes'])
    model.load_state_dict(torch.load(PREDICT_CONFIG['model_path'])['model_state_dict'])
    model.eval()
    model.to(DEVICE_CONFIG['device'])

    colors = generate_colors(DATASET_CONFIG['num_classes'])
    class_names = DATASET_CONFIG.get('class_names', [f'Class {i}' for i in range(DATASET_CONFIG['num_classes'])])
    
    print(f"🎨 Generated {len(colors)} colors for {len(class_names)} classes")
    print(f"📁 Results will be saved to: {PREDICT_CONFIG['save_path']}")

    images = glob(os.path.join(PREDICT_CONFIG['image_path'], "*.jpg"))
    print(f"🔍 Processing {len(images)} images...")
    
    for image_path in tqdm(images, desc="Processing images"):
        original_image = Image.open(image_path)
        
        processed_image = original_image.resize((DATASET_CONFIG['image_width'], DATASET_CONFIG['image_height']))
        image_array = np.array(processed_image)
        image_array = image_array.transpose(2, 0, 1)
        image_array = image_array.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE_CONFIG['device'])
        
        with torch.no_grad():
            outputs = model(image_tensor)[0]
            outputs = outputs.cpu().numpy()
            
            box_list = []
            for i in range(7):
                for j in range(7):
                    cls = np.argmax(outputs[i, j, 10:])
                    if (outputs[i, j, 4] > PREDICT_CONFIG['confidence_threshold']):
                        box1 = outputs[i, j, :5]
                        box1 = convert_coordinate_cell_to_image(box1, i, j, DATASET_CONFIG['grid_size'])
                        box1 = np.insert(box1, 0, cls)
                        box_list.append(box1)
                    if (outputs[i, j, 9] > PREDICT_CONFIG['confidence_threshold']):
                        box2 = outputs[i, j, 5:10]
                        box2 = convert_coordinate_cell_to_image(box2, i, j, DATASET_CONFIG['grid_size'])
                        box2 = np.insert(box2, 0, cls)
                        box_list.append(box2)
            
            box_list = NMS(box_list, PREDICT_CONFIG['iou_threshold'])
            
            if len(box_list) > 0:
                result_image = draw_bbox(original_image, box_list, colors, class_names)
            else:
                result_image = original_image
            
            save_results(result_image, box_list, image_path, PREDICT_CONFIG['save_path'])
    
    print(f"✅ Processing completed! Results saved to {PREDICT_CONFIG['save_path']}")
    print(f"📊 Images: {PREDICT_CONFIG['save_path']}")
    print(f"📝 Labels: {PREDICT_CONFIG['save_path']}/labels/")



if __name__ == "__main__":
    main()
