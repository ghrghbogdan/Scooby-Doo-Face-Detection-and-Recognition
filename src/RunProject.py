from Parameters import *
from FacialDetector import *
import pdb
from Visualize import *
from FacialRecogniser import *
from RunYolo import generate_yolo_solution

params: Parameters = Parameters()
if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)

os.makedirs('../evaluare/fisiere_solutie/343_Gheorghe_Bogdan', exist_ok=True)
os.makedirs('../evaluare/fisiere_solutie/343_Gheorghe_Bogdan/task1', exist_ok=True)
os.makedirs('../evaluare/fisiere_solutie/343_Gheorghe_Bogdan/task2', exist_ok=True)
os.makedirs('../evaluare/fisiere_solutie/343_Gheorghe_Bogdan/bonus', exist_ok=True)

npy_path = '../evaluare/fisiere_solutie/343_Gheorghe_Bogdan/task1' 
output_solution_path = '../evaluare/fisiere_solutie/343_Gheorghe_Bogdan/task2'
val_img_root = os.path.join('..', 'validare', 'validare')
val_ann_root = os.path.join('..', 'validare')


# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente
positive_features_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples) + '.npy')
if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Am incarcat descriptorii pentru exemplele pozitive')
else:
    print('Construim descriptorii pentru exemplele pozitive:')
    positive_features = facial_detector.get_positive_descriptors()
    np.save(positive_features_path, positive_features)
    print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_path)

# exemple negative
negative_features_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples) + '.npy')
if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Am incarcat descriptorii pentru exemplele negative')
else:
    print('Construim descriptorii pentru exemplele negative:')
    negative_features = facial_detector.get_negative_descriptors()
    np.save(negative_features_path, negative_features)
    print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_path)

# Pasul 4. Invatam clasificatorul liniar
print(positive_features.shape)
print(negative_features.shape)
training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
num_pos_real = positive_features.shape[0]
num_neg_real = negative_features.shape[0]
train_labels = np.concatenate((np.ones(num_pos_real), np.zeros(num_neg_real)))
facial_detector.train_classifier(training_examples, train_labels)

# Pasul 5. (optional) Antrenare cu exemple puternic negative (detectii cu scor >0 din cele 274 de imagini negative)
# Daca implementati acest pas ar trebui sa modificati functia FacialDetector.run()
# astfel incat sa va returneze descriptorii detectiilor cu scor > 0 din cele 274 imagini negative
# completati codul in continuare
# TODO:  (optional)  completeaza codul in continuare
if params.use_hard_mining:
    print('HARD NEGATIVE MINING')
    
    hard_negatives = facial_detector.mine_hard_negatives_from_training()
    
    if len(hard_negatives) > 0:
        all_negatives = np.concatenate([negative_features, hard_negatives], axis=0)
        
        # reantrenare
        training_examples = np.concatenate([positive_features, all_negatives], axis=0)
        train_labels = np.concatenate([
            np.ones(positive_features.shape[0]), 
            np.zeros(all_negatives.shape[0])
        ])
        
        old_model_path = os.path.join(params.dir_save_files, 'best_model_*')
        import glob as g
        for f in g.glob(old_model_path):
            os.remove(f)
        
        print('Re-antrenare cu hard negatives...')
        facial_detector.train_classifier(training_examples, train_labels)

detections, scores, file_names = facial_detector.run()
os.makedirs(params.output_dir, exist_ok=True)

np.save(os.path.join(params.output_dir, 'detections_all_faces.npy'), detections)
np.save(os.path.join(params.output_dir, 'scores_all_faces.npy'), scores)
np.save(os.path.join(params.output_dir, 'file_names_all_faces.npy'), file_names)
# if params.has_annotations:
#     facial_detector.eval_detections(detections, scores, file_names)
#     show_detections_with_ground_truth(detections, scores, file_names, params)
# else:
#     show_detections_without_ground_truth(detections, scores, file_names, params)


#incarcare model
model = CNN_v1().to(DEVICE)
if os.path.exists('../save_files/model1.pth'):
    model.load_state_dict(torch.load('../save_files/model1.pth'))
    print("Model încărcat cu succes!")
else:
    print("model1.pth nu exista")

# # 3. EVALUARE (GRAFICE)
# gt_data = load_ground_truth(val_ann_root)
# res = evaluate_detections(
#     model, 
#     os.path.join(npy_path, 'detections_all_faces.npy'),
#     os.path.join(npy_path, 'file_names_all_faces.npy'),
#     os.path.join(npy_path, 'scores_all_faces.npy'),
#     val_img_root,
#     gt_data
# )

generate_and_save_files(
    model,
    os.path.join(npy_path, 'detections_all_faces.npy'),
    os.path.join(npy_path, 'file_names_all_faces.npy'),
    output_solution_path,
    val_img_root
)

generate_yolo_solution()