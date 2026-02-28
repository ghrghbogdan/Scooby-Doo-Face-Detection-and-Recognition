from sklearn.preprocessing import normalize
from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog


class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None

    def get_positive_descriptors(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []
        print('Calculam descriptorii pt %d imagini pozitive...' % num_images)
        for i in range(num_images):
            print('Procesam exemplul pozitiv numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)

            positive_descriptors.append(features)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_descriptors(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in range(num_images):
            print('Procesam exemplul negativ numarul %d...' % i)
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            # TODO: completati codul functiei in continuare

            descr = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                        cells_per_block=(2, 2), feature_vector=True)
            negative_descriptors.append(descr)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return
        # from sklearn.preprocessing import normalize
        training_examples = normalize(training_examples, norm='l2')
        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 2*10 ** -1, 4*10 ** -1, 6*10 ** -1,8*10 ** -1, 10 ** 0]
        for c in Cs:
            print('Antrenam un clasificator pentru c=%f' % c)
            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)
            acc = model.score(training_examples, train_labels)
            print(acc)
            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(negative_scores) + 20))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = np.maximum(bbox_a[0], bbox_b[0])
        y_a = np.maximum(bbox_a[1], bbox_b[1])
        x_b = np.minimum(bbox_a[2], bbox_b[2])
        y_b = np.minimum(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

    def run(self):
        """
        Aceasta functie returneaza toate detectiile ( = ferestre) pentru toate imaginile din self.params.dir_test_examples
        Directorul cu numele self.params.dir_test_examples contine imagini ce
        pot sau nu contine fete. Aceasta functie ar trebui sa detecteze fete atat pe setul de
        date MIT+CMU dar si pentru alte imagini
        Functia 'non_maximal_suppression' suprimeaza detectii care se suprapun (protocolul de evaluare considera o detectie duplicata ca fiind falsa)
        Suprimarea non-maximelor se realizeaza pe pentru fiecare imagine.
        :return:
        detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
        detections[i, :] = [x_min, y_min, x_max, y_max]
        scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
        file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
        (doar numele, nu toata calea).
        """

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        detections = []
        scores = []
        file_names = []
        w = self.best_model.coef_[0]
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)
        for i in range(num_test_images):
        # for i in range(3):

            start_time = timeit.default_timer()
            print('Procesam imaginea de testare %d/%d..' % (i, num_test_images))
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            # TODO: completati codul functiei in continuare
            # Variabile locale pentru imaginea curenta
            original_img = img.copy()
            image_detections = []
            image_scores = []
            
            # Parametri Sliding Window
            scale = 1.0
            scale_factor = 0.9 
            current_img = original_img.copy()
            min_size = 32

            # piramida de imagini
            while current_img.shape[0] >= min_size and current_img.shape[1] >= min_size:
                H, W = current_img.shape
                step_size = self.params.dim_hog_cell 
                
                for win_h, win_w in self.params.dim_window:
                    if H < win_h or W < win_w:
                      continue

                    for y in range(0, H - win_h, step_size):
                        for x in range(0, W - win_w, step_size):
                            
                            patch = current_img[y : y + win_h, 
                                                x : x + win_w]

                            features = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                            cells_per_block=(2, 2), feature_vector=True)
                            features = normalize(features.reshape(1, -1), norm='l2')
                            features = features.flatten()
                            
                            score = np.dot(features, w) + bias
                            
                            if score > self.params.threshold:
                                x_min = int(x / scale)
                                y_min = int(y / scale)
                                x_max = int((x + win_w) / scale)
                                y_max = int((y + win_h) / scale)
                                
                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)

                current_img = cv.resize(current_img, (0,0), fx=scale_factor, fy=scale_factor)
                scale *= scale_factor
            
            if len(image_detections) > 0:
                det_array = np.array(image_detections)
                sc_array = np.array(image_scores)
                
                final_boxes, final_scores = self.non_maximal_suppression(det_array, 
                                                                            sc_array, 
                                                                            original_img.shape)
                
                for box, sc in zip(final_boxes, final_scores):
                    detections.append(box)
                    scores.append(sc)
                    file_names.append(os.path.basename(test_files[i]))

            end_time = timeit.default_timer()
            print('Timpul de procesarea al imaginii de testare %d/%d este %f sec.'
                  % (i, num_test_images, end_time - start_time))

        return np.array(detections), np.array(scores), np.array(file_names)

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:5], np.int32)

        num_gt_detections = len(ground_truth_detections)
        gt_exists_detection = np.zeros(num_gt_detections)
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.dir_save_files, 'precizie_medie.png'))
        plt.show()


    def mine_hard_negatives_from_training(self):
        characters = ['daphne', 'fred', 'shaggy', 'velma']
        hard_negative_descriptors = []
        
        w = self.best_model.coef_[0]
        bias = self.best_model.intercept_[0]
        
        for character in characters:
            
            annotations_path = os.path.join(self.params.dir_train_images, 
                                        f'{character}_annotations.txt')
            gt_data = np.loadtxt(annotations_path, dtype='str')
            gt_file_names = gt_data[:, 0]
            gt_boxes = np.array(gt_data[:, 1:5], dtype=int)
            
            # Procesează imaginile
            images_path = os.path.join(self.params.dir_train_images, character, '*.jpg')
            files = glob.glob(images_path)
            
            print(f'Mining hard negatives din {character}')
            i=0
            for file in files:
                img_name = os.path.basename(file)
                img = cv.imread(file, cv.IMREAD_GRAYSCALE)
                
                gt_indices = np.where(gt_file_names == img_name)[0]
                gt_boxes_img = gt_boxes[gt_indices] if len(gt_indices) > 0 else []
                
                # Sliding window
                for win_h, win_w in self.params.dim_window:
                    for y in range(0, img.shape[0] - win_h, self.params.dim_hog_cell):
                        for x in range(0, img.shape[1] - win_w, self.params.dim_hog_cell):
                            patch = img[y:y+win_h, x:x+win_w]
                            
                            features = hog(patch,
                                        pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                        cells_per_block=(2, 2),
                                        feature_vector=True)
                            features = normalize(features.reshape(1, -1), norm='l2')
                            features = features.flatten()
                            
                            score = np.dot(features, w) + bias
                            
                            if score > 0:
                                bbox = [x, y, x + win_w, y + win_h]
                                
                                is_false_positive = True
                                for gt_box in gt_boxes_img:
                                    if self.intersection_over_union(bbox, gt_box) > 0.3:
                                        is_false_positive = False
                                        break
                                
                                if is_false_positive:
                                    hard_negative_descriptors.append(features)
                i+=1
                print(f'Am gasit {len(hard_negative_descriptors)} hard negatives in {i} files')
        return np.array(hard_negative_descriptors)

