from ultralytics import YOLO
import os

# Find the path to your best weights. 
# If you ran training multiple times, this might be 'pose_training2', etc.
weights_path = '/home/fh-aachen/pose_detection/runs/pose/estop/runs/pose_training2/weights/best.pt'

if not os.path.exists(weights_path):
    print(f"Error: Could not find weights at {weights_path}")
    print("Please check your estop/runs/ directory for the correct 'pose_training' folder.")
else:
    # 1. Load the trained PyTorch model
    model = YOLO(weights_path)

    # 2. Export the model to OpenVINO format
    print("Exporting model to OpenVINO...")
    model.export(format='openvino')
    
    print("Export complete! You will find the OpenVINO model folder next to your .pt weights.")
