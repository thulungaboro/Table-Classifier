"""
demo.py
-------
Quick single-image inference demo for the Table Classifier (YOLOv8).

Runs the trained model on one image, draws bounding boxes + class labels
(bordered / borderless) on it, and saves the annotated result.

Usage:
    python demo.py --image table_test.jpg --weights runs/detect/train/weights/best.pt --out docs/example_detection.jpg
    python demo.py --image table_test.jpg   # uses defaults below
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# Class id -> name mapping (matching dataset/data.yaml)
CLASS_NAMES = {0: "bordered", 1: "borderless"}

DEFAULT_WEIGHTS = "runs/detect/train/weights/best.pt"
DEFAULT_OUT = "demo_output.jpg"
DEFAULT_CONF = 0.25


def parse_args():
    parser = argparse.ArgumentParser(description="Run table detection on a single image.")
    parser.add_argument("--image", default="table_test.jpg", help="Path to input image.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to trained model weights (.pt).")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path to save the annotated output image.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Confidence threshold (0-1).")
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    out_path = Path(args.out)

    if not image_path.exists():
        print(f"Input image not found: {image_path}. Please provide a valid --image path.")
        return
    if not weights_path.exists():
        print(
            f"Weights not found: {weights_path}\n"
            "Point --weights at your trained checkpoint, e.g. runs/detect/train/weights/best.pt"
        )
        return

    print(f"Loading model from {weights_path} ...")
    model = YOLO(str(weights_path))

    print(f"Running inference on {image_path} (conf >= {args.conf}) ...")
    results = model.predict(source=str(image_path), conf=args.conf, verbose=False)

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        print("No tables detected.")
    else:
        print(f"Detected {len(boxes)} table(s):")
        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
            xyxy = box.xyxy.tolist()[0]
            print(f"  - {label} (conf={conf:.2f}) bbox={[round(v, 1) for v in xyxy]}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated = result.plot()  # numpy array (BGR) with boxes/labels drawn
    import cv2
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved annotated image to {out_path}")


if __name__ == "__main__":
    main()
