import cv2 as cv
import random
import os
os.makedirs('../positive_examples', exist_ok=True)
os.makedirs('../negative_examples', exist_ok=True)
def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

i=0
j=0
folders = ['daphne', 'fred', 'shaggy', 'velma']

for folder in folders:
    print(f'Processing folder: {folder}')
    with open(f'../antrenare/{folder}_annotations.txt') as file:
        lines = [line.strip().split() for line in file.readlines()]
    for line in lines:
        print(f'Processing line: {line}')
        image_name = line[0]
        xmin, ymin, xmax, ymax = map(int, line[1:5])
        image = cv.imread(f'../antrenare/{folder}/{image_name}')
        face = image[ymin:ymax, xmin:xmax]
        face = cv.resize(face, (32,40))
        cv.imwrite(f'../positive_examples/face_{i}.jpg', face)
        i += 1
        for _ in range(5):
            random_nr = random.randint(0,64)
            random_nr= random_nr - 32
            random_x = random.randint(0, max(0, image.shape[1] - (xmax - xmin) + random_nr))
            random_x = max(0, min(random_x, image.shape[1] - (xmax - xmin)))
            random_nr = random.randint(0,80)
            random_nr= random_nr - 40
            random_y = random.randint(0, max(0, image.shape[0] - (ymax - ymin) + random_nr))
            random_y = max(0, min(random_y, image.shape[0] - (ymax - ymin)))
            non_face = image[random_y:random_y + (ymax - ymin), random_x:random_x + (xmax - xmin)]

            while intersection_over_union((xmin, ymin, xmax, ymax), (random_x, random_y, random_x + (xmax - xmin), random_y + (ymax - ymin))) > 0.1:
                random_nr = random.randint(0,64)
                random_nr= random_nr - 32
                random_x = random.randint(0, max(0, image.shape[1] - (xmax - xmin) + random_nr))
                random_x = max(0, min(random_x, image.shape[1] - (xmax - xmin)))
                random_nr = random.randint(0,80)
                random_nr= random_nr - 40
                random_y = random.randint(0, max(0, image.shape[0] - (ymax - ymin) + random_nr))
                random_y = max(0, min(random_y, image.shape[0] - (ymax - ymin)))
                non_face = image[random_y:random_y + (ymax - ymin), random_x:random_x + (xmax - xmin)]
            non_face = cv.resize(non_face, (32,40))
            cv.imwrite(f'../negative_examples/non_face_{j}.jpg', non_face)
            j += 1
