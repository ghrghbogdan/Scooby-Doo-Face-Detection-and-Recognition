import os
from PIL import Image

CLASS_MAP = {
    'daphne': 0,
    'fred': 1,
    'shaggy': 2,
    'velma': 3,
    'unknown': 4
}

def convert_pascal_to_yolo(x1, y1, x2, y2, img_width, img_height):
    center_x = (x1 + x2) / 2 / img_width
    center_y = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return center_x, center_y, width, height

def create_yolo_dataset():
    base_path = os.path.join('..', 'antrenare')
    
    for char_name in ['daphne', 'fred', 'shaggy', 'velma']:
        char_dir = os.path.join(base_path, char_name)
        labels_dir = os.path.join(char_dir, 'labels')
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)

    annotation_file = os.path.join(base_path, 'daphne_annotations.txt')
    
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        image_annotations = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:
                img_name = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:5])
                label_str = parts[5].lower()
                
                if label_str in CLASS_MAP:
                    if img_name not in image_annotations:
                        image_annotations[img_name] = []
                    image_annotations[img_name].append((x1, y1, x2, y2, CLASS_MAP[label_str]))
        
        for img_name, annotations in image_annotations.items():
            img_path = None
            for char_name in ['daphne', 'fred', 'shaggy', 'velma']:
                potential_path = os.path.join(base_path, char_name, img_name)
                if os.path.exists(potential_path):
                    img_path = potential_path
                    break
            
            if img_path:

                img = Image.open(img_path)
                img_width, img_height = img.size
                
                label_name = os.path.splitext(img_name)[0] + '.txt'
                label_path = os.path.join(base_path, 'daphne', 'labels', label_name)
                
                with open(label_path, 'w') as lf:
                    for x1, y1, x2, y2, class_id in annotations:
                        cx, cy, w, h = convert_pascal_to_yolo(x1, y1, x2, y2, img_width, img_height)
                        lf.write(f"{class_id} {cx} {cy} {w} {h}\n")
                    
    
    print("Conversion complete")

if __name__ == "__main__":
    create_yolo_dataset()
