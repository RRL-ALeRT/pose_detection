import os
from ultralytics import YOLO

# 1. Load the pre-trained YOLO11 pose model 
model = YOLO('yolo11n-pose.pt') # Start with a nano model for speedy training

yaml_path = os.path.expanduser("~/pose_detection/estop/data.yaml")

# 2. Train the model!
results = model.train(
    data=yaml_path,        # path to your dataset config
    epochs=200,            # how many times to look over the dataset
    imgsz=640,             # image size to train at (matches blender output)
    batch=16,              # images per batch (lower this if your GPU runs out of memory)
    device=0,              # train on GPU 0
    
    # --- AUGMENTATIONS FOR STRICTLY FULL OBJECTS ---
    degrees=180.0,         # Simulate mounting at any angle (walls, sideways, upside-down)
    translate=0.0,         # explicitly 0.0 - prevents YOLO from artificially pushing the E-stop off the edge of the image
    scale=0.3,             # +/- 30% scaling to simulate zoom/distance without cutting the object off
    erasing=0.0,           # explicitly 0.0 - prevents YOLO from randomly drawing black boxes over parts of the E-stop
    perspective=0.0005,    # Gentle perspective skew (simulates cheap camera lenses or odd viewing angles)
    fliplr=0.5,            # 50% horizontal flip (E-stops are generally symmetric, so this is safe and helps generalized learning)
    mosaic=0.0,            # explicitly 0.0 - disables YOLO's default 4-image stitching which forcibly generates cut-off "partial" objects at the borders
    
    project='estop/runs',  # where to save the weights
    name='pose_training'
)

# 3. Automatically export to OpenVINO after training finishes!
print("Training finished! Exporting best model to OpenVINO...")
# The YOLO object retains the best weights after training finishes.
model.export(format='openvino')
print("Export complete! You will find the OpenVINO model folder next to your .pt weights.")
