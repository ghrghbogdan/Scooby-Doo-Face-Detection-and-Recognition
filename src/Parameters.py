import os

class Parameters:
    def __init__(self):
        self.base_dir = '../'
        self.dir_pos_examples = os.path.join(self.base_dir, 'positive_examples')
        self.dir_neg_examples = os.path.join(self.base_dir, 'negative_examples')
        self.dir_test_examples = os.path.join(self.base_dir,'testare')
        self.path_annotations = os.path.join(self.base_dir, 'validare/task1_gt_validare.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'save_files')
        self.dir_train_images = os.path.join(self.base_dir, 'antrenare')
        self.output_dir = os.path.join(self.base_dir, 'evaluare/fisiere_solutie/343_Gheorghe_Bogdan/task1')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        self.dim_window = [(40, 32)]  # exemplele pozitive 
        self.dim_hog_cell = 4  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 6546  # numarul exemplelor pozitive
        self.number_negative_examples = 32734  # numarul exemplelor negative
        self.has_annotations = False
        self.threshold = 0.5
        self.use_flip_images = True
        self.use_hard_mining = False # parametrn pentru hard negative mining 
