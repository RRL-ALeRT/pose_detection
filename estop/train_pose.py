import os
from ultralytics import YOLO

# 1. Load the pre-trained YOLO11 pose model 
model = YOLO('yolo11n-pose.pt') # Start with a nano model for speedy training

yaml_path = os.path.expanduser("~/pose_detection/estop/data.yaml")

# 2. Train the model!
results = model.train(
    data=yaml_path,        # path to your dataset config
    epochs=100,            # how many times to look over the dataset
    imgsz=640,             # image size to train at (matches blender output)
    batch=16,              # images per batch (lower this if your GPU runs out of memory)
    device=0,              # train on GPU 0
    project='estop/runs',  # where to save the weights
    name='pose_training'
)
