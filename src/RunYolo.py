import numpy as np
import os
from ultralytics import YOLO

IMAGES_DIR = os.path.join('..', 'testare')
MODEL_PATH = "../save_files/modelYoLo/weights/best.pt" 
OUTPUT_DIR_1 = "../evaluare/fisiere_solutie/343_Gheorghe_Bogdan/bonus/task1"
OUTPUT_DIR_2 = "../evaluare/fisiere_solutie/343_Gheorghe_Bogdan/bonus/task2"

ID_TO_NAME = {
    0: 'daphne', 1: 'fred', 2: 'shaggy', 3: 'velma', 4: 'unknown'
}

def generate_yolo_solution():
    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=IMAGES_DIR, 
        conf=0.25,        
        save=False,       
        stream=True,      
        verbose=False
    )

    task1 = {
        'det': [], 'fname': [], 'score': []
    }
    task2 = {
        'daphne': {'det': [], 'fname': [], 'score': []},
        'fred':   {'det': [], 'fname': [], 'score': []},
        'shaggy': {'det': [], 'fname': [], 'score': []},
        'velma':  {'det': [], 'fname': [], 'score': []}
    }

    count = 0
    for r in results:
        img_name = os.path.basename(r.path)
        boxes = r.boxes
        
        if len(boxes) == 0: continue

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = ID_TO_NAME.get(cls_id, 'unknown')

            task1['det'].append([x1, y1, x2, y2])
            task1['fname'].append(img_name)
            task1['score'].append(conf)

            if class_name in task2:
                task2[class_name]['det'].append([x1, y1, x2, y2])
                task2[class_name]['fname'].append(img_name)
                task2[class_name]['score'].append(conf)
        
        count += 1
        if count % 200 == 0:
            print(f"\rProcesat {count} imagini...", end="")

    print("\n\nSalvare fișiere pe disk...")
    if not os.path.exists(OUTPUT_DIR_1):
        os.makedirs(OUTPUT_DIR_1)
    if not os.path.exists(OUTPUT_DIR_2):
        os.makedirs(OUTPUT_DIR_2)

    np.save(os.path.join(OUTPUT_DIR_1, "detections_all_faces.npy"), np.array(task1['det']))
    np.save(os.path.join(OUTPUT_DIR_1, "file_names_all_faces.npy"), np.array(task1['fname']))
    np.save(os.path.join(OUTPUT_DIR_1, "scores_all_faces.npy"), np.array(task1['score']))

    for char_name, data in task2.items():
        np.save(os.path.join(OUTPUT_DIR_2, f"detections_{char_name}.npy"), np.array(data['det']))
        np.save(os.path.join(OUTPUT_DIR_2, f"file_names_{char_name}.npy"), np.array(data['fname']))
        np.save(os.path.join(OUTPUT_DIR_2, f"scores_{char_name}.npy"), np.array(data['score']))

