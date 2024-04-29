from ultralytics import YOLO

# Load a model
#model = YOLO('yolov8s.yaml')  # build a new model from YAML
#model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='datasets/data.yaml', epochs=500, imgsz=128, seed=1234)