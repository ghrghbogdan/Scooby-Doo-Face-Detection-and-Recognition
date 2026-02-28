import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.nn.functional as F
from CNNTrainer import *

CLASS_MAP = {
    'daphne': 0, 'fred': 1, 'shaggy': 2, 'velma': 3, 'unknown': 4
}
REV_CLASS_MAP = {v: k for k, v in CLASS_MAP.items()}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_ground_truth(annotations_dir):
    """Încarcă GT pentru validare."""
    gt_data = {} 
    characters = ['daphne', 'fred', 'shaggy', 'velma']
    for char_name in characters:
        filename = f"task2_{char_name}_gt_validare.txt"
        path = os.path.join(annotations_dir, filename)
        
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    img_name = parts[0]
                    if img_name not in gt_data:
                        gt_data[img_name] = set()
                    gt_data[img_name].add(CLASS_MAP[char_name])
    return gt_data

# def evaluate_detections(model, detections_path, file_names_path, scores_path, images_root, gt_data):
#     """Calculează scorurile pentru grafice."""
#     detections = np.load(detections_path)
#     file_names = np.load(file_names_path)
    
#     print(f"--- EVALUARE GRAFICE: Procesez {len(detections)} detecții ---")
    
#     results = {
#         'daphne': {'true': [], 'scores': []},
#         'fred':   {'true': [], 'scores': []},
#         'shaggy': {'true': [], 'scores': []},
#         'velma':  {'true': [], 'scores': []}
#     }
    
#     model.eval()
#     with torch.no_grad():
#         for i, (bbox, fname) in enumerate(zip(detections, file_names)):
#             fname = str(fname) 
#             img_path = os.path.join(images_root, fname)
#             if not os.path.exists(img_path): continue
                
    
#             image = Image.open(img_path).convert('RGB')
#             x1, y1, x2, y2 = map(int, bbox)
#             w, h = image.size
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)
            
#             if x2 <= x1 or y2 <= y1: continue
            
#             face_crop = image.crop((x1, y1, x2, y2))
#             input_tensor = transform(face_crop).unsqueeze(0).to(DEVICE)
#             outputs = model(input_tensor)
#             probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
#             true_labels_in_img = gt_data.get(fname, set())
            
#             for char_name, char_idx in CLASS_MAP.items():
#                 if char_name == 'unknown': continue
#                 score = probs[char_idx]
#                 label = 1 if char_idx in true_labels_in_img else 0
#                 results[char_name]['scores'].append(score)
#                 results[char_name]['true'].append(label)
                    

#             if i % 500 == 0: print(f"\rProgres Eval: {i}/{len(detections)}", end="")
#     print("\n")
#     return results

def generate_and_save_files(model, detections_path, file_names_path, output_dir, images_root):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    detections = np.load(detections_path)
    file_names = np.load(file_names_path)
    
    solution_data = {
        'daphne': {'det': [], 'fname': [], 'score': []},
        'fred':   {'det': [], 'fname': [], 'score': []},
        'shaggy': {'det': [], 'fname': [], 'score': []},
        'velma':  {'det': [], 'fname': [], 'score': []}
    }

    model.eval()
    with torch.no_grad():
        for i, (bbox, fname) in enumerate(zip(detections, file_names)):
            fname = str(fname)
            img_path = os.path.join(images_root, fname)
            if not os.path.exists(img_path): continue
            
            image = Image.open(img_path).convert('RGB')
            x1, y1, x2, y2 = map(int, bbox)
            w, h = image.size
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1: continue

            face_crop = image.crop((x1, y1, x2, y2))
            input_tensor = transform(face_crop).unsqueeze(0).to(DEVICE)
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            
            pred_idx = np.argmax(probs)
            pred_name = REV_CLASS_MAP.get(pred_idx)
            pred_score = probs[pred_idx]

            if pred_name in solution_data:
                solution_data[pred_name]['det'].append(bbox)
                solution_data[pred_name]['fname'].append(fname)
                solution_data[pred_name]['score'].append(pred_score)

            
            if i % 500 == 0: print(f"\r{i}/{len(detections)}", end="")
    for char_name, data in solution_data.items():
        np.save(os.path.join(output_dir, f"detections_{char_name}.npy"), np.array(data['det']))
        np.save(os.path.join(output_dir, f"file_names_{char_name}.npy"), np.array(data['fname']))
        np.save(os.path.join(output_dir, f"scores_{char_name}.npy"), np.array(data['score']))
