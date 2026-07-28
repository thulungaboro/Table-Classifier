"""
Model Testing Script
--------------------
This script tests the trained YOLO model on new images.
You can test on:
1. Single image
2. Multiple images
3. Validation set (to see metrics)
4. Custom image directory
"""

from pathlib import Path
from ultralytics import YOLO
import cv2
import os

def load_best_model():
    """Load the best-performing trained model from the runs folder."""
    import csv
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent.parent
    run_dir = base_dir / "runs" / "detect"

    candidates = []
    for train_dir in run_dir.glob("train*"):
        weight_path = train_dir / "weights" / "best.pt"
        results_path = train_dir / "results.csv"

        if not weight_path.exists() or not results_path.exists():
            continue

        best_map50 = -1.0
        best_map50_95 = -1.0
        with results_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    map50 = float(row["metrics/mAP50(B)"])
                    map50_95 = float(row["metrics/mAP50-95(B)"])
                except (KeyError, TypeError, ValueError):
                    continue

                if map50 > best_map50 or (map50 == best_map50 and map50_95 > best_map50_95):
                    best_map50 = map50
                    best_map50_95 = map50_95

        candidates.append((best_map50, best_map50_95, weight_path))

    if not candidates:
        raise FileNotFoundError(" No training runs found. Run training first with: python training_1.py")

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_weight_path = candidates[0][2]
    print(f" Loading model from: {best_weight_path}")
    from ultralytics import YOLO
    return YOLO(str(best_weight_path))


def test_single_image(model, image_path):
    """Test model on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing single image: {image_path}")
    print(f"{'='*60}")

    if not os.path.exists(image_path):
        print(f" Image not found: {image_path}")
        return

    # Run inference with lower NMS IoU threshold to remove overlapping duplicates
    results = model.predict(image_path, conf=0.25, iou=0.45, imgsz=1024, save=True, project="runs/detect", name="predict")

    # Print results
    for result in results:
        print(f"\nDetections found: {len(result.boxes)}")
        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            print(f"  [{idx+1}] {class_name} - Confidence: {confidence:.2%}")


def test_validation_set(model):
    """Test model on validation set and get metrics"""
    print(f"\n{'='*60}")
    print("Testing on Validation Set (with metrics)")
    print(f"{'='*60}")

    base_dir = Path(__file__).resolve().parent.parent
    dataset_yaml = base_dir / "dataset" / "data.yaml"

    if not dataset_yaml.exists():
        print(f" Dataset config not found: {dataset_yaml}")
        return

    # Run validation
    metrics = model.val(data=str(dataset_yaml))

    print("\n Validation Metrics:")
    print(f"  mAP50: {metrics.box.map50:.3f}")
    print(f"  mAP50-95: {metrics.box.map:.3f}")


def test_directory(model, image_dir, output_dir="test_results"):
    """Test model on all images in a directory"""
    print(f"\n{'='*60}")
    print(f"Testing all images in: {image_dir}")
    print(f"{'='*60}")

    image_path = Path(image_dir)
    if not image_path.exists():
        print(f" Directory not found: {image_dir}")
        return

    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []

    for ext in extensions:
        image_files.extend(image_path.glob(f"*{ext}"))
        image_files.extend(image_path.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f" No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Run predictions on all images with lower NMS IoU threshold
    results = model.predict(
        source=[str(f) for f in image_files],
        conf=0.25,
        iou=0.45,
        imgsz=1024,
        save=True,
        project="runs/detect",
        name=output_dir
    )

    # Summarize results
    print(f"\n Summary:")
    total_detections = 0
    for idx, result in enumerate(results):
        total_detections += len(result.boxes)
        print(f"  Image {idx+1}: {len(result.boxes)} detections")

    print(f"Total detections: {total_detections}")
    print(f"Results saved to: runs/detect/{output_dir}")


def test_sample_images(model):
    """Test on sample images in project root"""
    print(f"\n{'='*60}")
    print("Testing Sample Images from Project Root")
    print(f"{'='*60}")

    base_dir = Path(__file__).resolve().parent.parent
    sample_images = [
        base_dir / "table_test.jpg",
        base_dir / "table2.png"
    ]

    for img in sample_images:
        if img.exists():
            test_single_image(model, str(img))
        else:
            print(f" Sample image not found: {img}")


def main():
    """Main testing menu"""
    print("\n" + "="*60)
    print(" TABLE CLASSIFIER - MODEL TESTING")
    print("="*60)

    # Load model
    try:
        model = load_best_model()
    except FileNotFoundError as e:
        print(f" {e}")
        return

    print("\n Select Test Option:")
    print("  1. Test on sample images (table_test.jpg, table2.png)")
    print("  2. Test on validation set (with metrics)")
    print("  3. Test on custom image file")
    print("  4. Test on all images in a directory")
    print("  5. Test on dataset validation folder")
    print("  0. Exit")

    choice = input("\nEnter choice (0-5): ").strip()

    if choice == "1":
        test_sample_images(model)

    elif choice == "2":
        test_validation_set(model)

    elif choice == "3":
        image_path = input("Enter image path: ").strip()
        test_single_image(model, image_path)

    elif choice == "4":
        directory = input("Enter directory path: ").strip()
        output_name = input("Enter output folder name (default: predict): ").strip() or "predict"
        test_directory(model, directory, output_name)

    elif choice == "5":
        base_dir = Path(__file__).resolve().parent.parent
        val_images = base_dir / "dataset" / "images" / "val"
        test_directory(model, str(val_images), "validate_test")

    elif choice == "0":
        print("Exiting...")
        return

    else:
        print(" Invalid choice")
        return

    print(f"\n{'='*60}")
    print(" Testing complete!")
    print(f"Results saved to: runs/detect/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
