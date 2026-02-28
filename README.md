Install the following libraries with their specified versions:
================================================================================
numpy==1.24.3
opencv_python==4.8.0.74
scikit_image==0.21.0
scikit_learn==1.3.0
Pillow==10.0.0
matplotlib==3.7.2
torch==2.0.1
torchvision==0.15.2
ultralytics==8.0.100

Installation command:
    pip install -r requirements.txt

Or install manually:
    pip install numpy==1.24.3 opencv_python==4.8.0.74 scikit_image==0.21.0 \
                scikit_learn==1.3.0 Pillow==10.0.0 matplotlib==3.7.2 \
                torch==2.0.1 torchvision==0.15.2 ultralytics==8.0.100

HOW TO RUN THE PROJECT
================================================================================

Script: RunProject.py
Location: src/RunProject.py

Command:
    cd src/
    python RunProject.py

Output locations:

    - Task 1 (All faces detected): evaluare/fisiere_solutie/343_Gheorghe_Bogdan/task1/          
    
    - Task 2 (Per-character predictions): evaluare/fisiere_solutie/343_Gheorghe_Bogdan/task2/

    - Bonus Task 1 (All detections): evaluare/fisiere_solutie/343_Gheorghe_Bogdan/bonus/task1/        
    
    - Bonus Task 2 (Per-character): evaluare/fisiere_solutie/343_Gheorghe_Bogdan/bonus/task2/




PROJECT STRUCTURE
================================================================================


    - antrenare/              
    - evaluare/              
    - save_files/             
    - src/                   
      ├── RunProject.py    
      ├── RunYolo.py       
      ├── yolo.py          
      ├── CNNTrainer.py     
      ├── FacialDetector.py 
      ├── FacialRecogniser.py 
      ├── Parameters.py     
      ├── Visualize.py
      ├── script.py
      ├── convert_to_yolo_format.py      
      └── scooby_config.yaml
    - testare/
    - validare/              


NOTES
================================================================================

- Make sure you keep the same project structure as presented
- By executing RunProject.py you will get all the solution files in evaluare/fisiere_solutie/343_Gheorghe_Bogdan
- I already modified the path to the test folder, but you can find the path for the first 2 tasks in Parameters.py, line 8, self.dir_test_examples = os.path.join(self.base_dir,'testare') and for the bonus task RunYolo.py, line 6, MODEL_PATH = "../save_files/modelYoLo/weights/best.pt" 
- task 1 should take about 35 seconds per image on my PC, so it should take about 2 hours for the full test, the other tasks run fast and smooth
# Scooby-Doo-Face-Detection-and-Recognition
