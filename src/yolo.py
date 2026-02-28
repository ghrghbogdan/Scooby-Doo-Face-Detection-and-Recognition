from ultralytics import YOLO


model = YOLO("yolov8n.pt") 
model.train(
    data="scooby_config.yaml",
    epochs=100,       
    imgsz=640,       
    batch=64,        
    project="ScoobyYolo",
    name="run_test",
    verbose=True,
    plots=True,
    workers=0
)
print("Training complete.")