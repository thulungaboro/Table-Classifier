from ultralytics import YOLO
import csv
from pathlib import Path

def main():
    print("=" * 60)
    print("  EVALUATION STEP 1: POOLED TEST SET (Real + Synthetic)")
    print("=" * 60)
    
    # Load your newly trained model
    model = YOLO(r"D:\Project-1-Tb-DTC\runs\detect\train\weights\best.pt")
    
    # 1. Run evaluation on the full pooled test set
    results_pooled = model.val(
        data=r"D:\Project-1-Tb-DTC\dataset_combined\data.yaml", 
        split="test",
        name="eval_test_pooled"
    )
    
    print("\n" + "=" * 60)
    print("  EVALUATION STEP 2: REAL-ONLY TEST SET")
    print("=" * 60)
    print("Filtering test set using manifest.csv to isolate real-world data...\n")
    
    # 2. Setup the real-only test data
    dataset_dir = Path(r"D:\Project-1-Tb-DTC\dataset_combined")
    manifest_path = dataset_dir / "manifest.csv"
    
    real_test_images = []
    
    # Read manifest and find all test images that came from the "real" source
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == "test" and row["source"] == "real":
                img_path = dataset_dir / "images" / "test" / row["image"]
                real_test_images.append(str(img_path.absolute()))
                
    # YOLO allows you to provide a .txt file containing a list of images instead of a folder
    test_txt_path = dataset_dir / "test_real_only.txt"
    with open(test_txt_path, 'w') as f:
        f.write("\n".join(real_test_images))
        
    # Create a temporary YAML config that points to this text file
    yaml_content = f"""
path: {str(dataset_dir.absolute())}
train: images/train
val: images/val
test: {test_txt_path.name}  # Points to our filtered text list
nc: 2
names: ["bordered", "borderless"]
"""
    yaml_path = dataset_dir / "data_real_only.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
        
    print(f"Found {len(real_test_images)} real-world images in the test set. Evaluating...\n")
    
    # 3. Run evaluation on the real-only subset
    results_real = model.val(
        data=str(yaml_path), 
        split="test",
        name="eval_test_real_only"
    )
    
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"POOLED TEST mAP50     : {results_pooled.box.map50:.4f} (Evaluated on full test split)")
    print(f"REAL-ONLY TEST mAP50  : {results_real.box.map50:.4f} (Evaluated on {len(real_test_images)} real-world images)")
    print("\nCompare the two numbers above. The 'REAL-ONLY' score tells you how well")
    print("the model actually performs on your genuine document data.")

if __name__ == "__main__":
    main()
