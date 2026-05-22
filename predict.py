import pyrealsense2 as rs
import numpy as np
import cv2
import time
from ultralytics import YOLO

# 1. Load your best model
model = YOLO("/home/fh-aachen/yolo_dataset/linear_sense/runs/detect/train/weights/best.pt")
# model = YOLO("/home/fh-aachen/oliver_ws/estop_keypoints_dataset_generation_multi/runs/detect/train/weights/best.pt")

# 2. Configure RealSense
pipeline = rs.pipeline()
config = rs.config()

# RealSense D435/D455 usually uses 640x480 for the RGB stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Verify that at least one RealSense device is connected before starting.
ctx = rs.context()
if len(ctx.query_devices()) == 0:
    raise RuntimeError("No RealSense device detected. Check USB connection and power.")

print("Starting RealSense pipeline...")
pipeline.start(config)
time.sleep(1.0)

consecutive_timeouts = 0

try:
    while True:
        # Wait for frames with a shorter timeout so we can recover if the stream stalls.
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
        except RuntimeError as exc:
            if "Frame didn't arrive within" in str(exc):
                consecutive_timeouts += 1
                print(
                    f"Warning: frame timeout ({consecutive_timeouts}/5). Retrying..."
                )

                if consecutive_timeouts >= 5:
                    print("Restarting RealSense pipeline after repeated timeouts...")
                    pipeline.stop()
                    time.sleep(0.5)
                    pipeline.start(config)
                    time.sleep(1.0)
                    consecutive_timeouts = 0
                continue

            raise

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        consecutive_timeouts = 0

        # Convert to numpy array (OpenCV format)
        img = np.asanyarray(color_frame.get_data())

        # 3. Run YOLO inference
        # We use stream=True for real-time performance
        results = model(img, stream=True, conf=0.5)  # Adjust confidence threshold as needed

        # 4. Show results
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("RealSense + YOLOv8", annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
