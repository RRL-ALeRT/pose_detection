import os
from ultralytics import YOLO

# 1. Standard Object Detection Model (not -pose)
model = YOLO('yolo11s.pt')

yaml_path = os.path.expanduser("~/pose_detection/estop/data_detect.yaml")

results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    
    # --- AUGMENTATIONS FOR STRICTLY FULL OBJECTS ---
    degrees=180.0,
    translate=0.0,
    scale=0.3,
    erasing=0.0,
    perspective=0.0005,
    fliplr=0.5,
    mosaic=0.0,
    hsv_h=0.015,  # Add color augmentations to generalize texture better
    hsv_s=0.7,    
    hsv_v=0.4,    
    
    project='estop/runs_detect',
    name='bbox_training'
)

print("Training finished! Exporting best model to OpenVINO...")
model.export(format='openvino')
print("Export complete!")
