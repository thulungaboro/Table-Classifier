"""
Quick Demo Script
-----------------
Runs inference on a target image using the trained YOLOv8 model and prints detections.

Usage:
    python demo.py --image path/to/document.jpg
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Quick Demo for Table Classifier YOLOv8")
    parser.add_argument("--image", default="table_test.jpg", help="Path to document image")
    parser.add_argument("--weights", default="runs/detect/train/weights/best.pt", help="Path to model weights")
    args = parser.parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)

    if not weights_path.exists():
        print(f"Weights file not found at '{weights_path}'. Please verify the model path.")
        return

    if not image_path.exists():
        print(f"Image not found at '{image_path}'. Provide a valid --image path.")
        return

    print(f"Loading YOLOv8 Table Classifier from '{weights_path}'...")
    model = YOLO(str(weights_path))

    print(f"Running inference on '{image_path}'...")
    results = model.predict(source=str(image_path), conf=0.25, iou=0.45, imgsz=1024, save=True)

    for r in results:
        print(f"\nFound {len(r.boxes)} table(s):")
        for idx, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            conf = float(box.conf[0])
            coords = [round(c, 2) for c in box.xyxy[0].tolist()]
            print(f"  [{idx+1}] {label.upper()} | Confidence: {conf:.2%} | Box [x1, y1, x2, y2]: {coords}")

    print("\nAnnotated prediction output saved in 'runs/detect/predict/'.")

if __name__ == "__main__":
    main()
