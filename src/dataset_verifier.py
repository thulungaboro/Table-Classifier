"""
Verify the split balance of a combined dataset produced by build_combined_dataset.py.

Reads the output directory (images + labels per split) and reports:
  - Per-split totals (real vs synthetic)
  - Per-class annotation counts in each split
  - Whether val and test each have >= a minimum threshold per class
  - Negative page counts per split

Exits with code 1 if any threshold is not met.

Usage:
  python verify_split_balance.py D:/Project-1-Tb-DTC/dataset_combined --min-per-class 150
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    import csv
    HAS_CSV = True
except ImportError:
    HAS_CSV = False

IMG_EXTS = (".png", ".jpg", ".jpeg")


def count_classes_in_dir(label_dir):
    """Count annotations per class and negative (empty) label files."""
    class_counts = defaultdict(int)
    total_files = 0
    negative_files = 0

    label_dir = Path(label_dir)
    if not label_dir.exists():
        return class_counts, 0, 0

    for lbl in sorted(label_dir.glob("*.txt")):
        total_files += 1
        text = lbl.read_text().strip()
        if not text:
            negative_files += 1
            continue
        for line in text.splitlines():
            parts = line.strip().split()
            if parts:
                class_counts[int(parts[0])] += 1

    return class_counts, total_files, negative_files


def count_images_in_dir(image_dir):
    """Count image files."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return 0
    return sum(1 for f in image_dir.iterdir() if f.suffix.lower() in IMG_EXTS)


def load_manifest(dataset_root):
    """Load manifest.csv and return per-split source counts."""
    manifest_path = Path(dataset_root) / "manifest.csv"
    if not manifest_path.exists():
        return None

    split_source = defaultdict(lambda: defaultdict(int))
    with open(manifest_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_source[row["split"]][row["source"]] += 1
    return split_source


def main():
    ap = argparse.ArgumentParser(description="Verify split balance of a combined YOLO dataset.")
    ap.add_argument("dataset", help="Root directory of the combined dataset")
    ap.add_argument("--min-per-class", type=int, default=150,
                     help="Minimum annotations per class required in val and test (default: 150)")
    args = ap.parse_args()

    root = Path(args.dataset)
    if not root.exists():
        print(f"ERROR: Dataset directory not found: {root}")
        sys.exit(1)

    splits = ["train", "val", "test"]
    class_names = {0: "bordered", 1: "borderless"}
    all_classes = set()
    failures = []

    # Load manifest for source breakdown
    manifest_data = load_manifest(root)

    print("=" * 70)
    print(f"  DATASET VERIFICATION: {root}")
    print("=" * 70)
    print()

    # ── Per-split summary ────────────────────────────────────
    print(f"{'Split':<8} {'Images':>8} {'Labels':>8} {'Negatives':>10} {'Real':>6} {'Synth':>7}")
    print("-" * 55)

    split_data = {}
    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        n_images = count_images_in_dir(img_dir)
        class_counts, n_labels, n_negatives = count_classes_in_dir(lbl_dir)
        all_classes.update(class_counts.keys())
        split_data[split] = {
            "n_images": n_images,
            "n_labels": n_labels,
            "n_negatives": n_negatives,
            "class_counts": class_counts,
        }

        n_real = manifest_data[split].get("real", 0) if manifest_data else "?"
        n_synth = manifest_data[split].get("synthetic", 0) if manifest_data else "?"
        print(f"{split:<8} {n_images:>8} {n_labels:>8} {n_negatives:>10} {str(n_real):>6} {str(n_synth):>7}")

    total_images = sum(d["n_images"] for d in split_data.values())
    print("-" * 55)
    print(f"{'TOTAL':<8} {total_images:>8}")
    print()

    # ── Per-class annotation counts ──────────────────────────
    sorted_classes = sorted(all_classes)
    header = f"{'Split':<8}" + "".join(f" {class_names.get(c, f'cls_{c}'):>12}" for c in sorted_classes) + f" {'Total':>8}"
    print(header)
    print("-" * len(header))

    for split in splits:
        cc = split_data[split]["class_counts"]
        row = f"{split:<8}"
        total_ann = 0
        for c in sorted_classes:
            count = cc.get(c, 0)
            total_ann += count
            row += f" {count:>12}"
        row += f" {total_ann:>8}"
        print(row)
    print()

    # ── Threshold check for val and test ──────────────────────
    print(f"Threshold check: >= {args.min_per_class} annotations per class in val & test")
    print("-" * 55)

    for split in ("val", "test"):
        cc = split_data[split]["class_counts"]
        for c in sorted_classes:
            count = cc.get(c, 0)
            name = class_names.get(c, f"class_{c}")
            status = "PASS" if count >= args.min_per_class else "FAIL"
            marker = "[PASS]" if status == "PASS" else "[FAIL]"
            print(f"  {marker} {split}/{name}: {count} annotations (need >= {args.min_per_class})")
            if status == "FAIL":
                failures.append(f"{split}/{name}: {count} < {args.min_per_class}")

    print()

    # ── Image/label mismatch check ────────────────────────────
    print("Image-label pairing check:")
    print("-" * 55)
    for split in splits:
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.exists():
            continue
        img_stems = {f.stem for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
        lbl_stems = {f.stem for f in lbl_dir.glob("*.txt")}
        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems
        status = "[OK]" if not missing_labels and not orphan_labels else "[ERR]"
        print(f"  {status} {split}: {len(img_stems)} images, {len(lbl_stems)} labels", end="")
        if missing_labels:
            print(f"  (MISSING {len(missing_labels)} labels)", end="")
            failures.append(f"{split}: {len(missing_labels)} images without labels")
        if orphan_labels:
            print(f"  (ORPHAN {len(orphan_labels)} labels)", end="")
        print()

    print()

    # ── Final verdict ─────────────────────────────────────────
    if failures:
        print("=" * 70)
        print("  RESULT: FAIL")
        print("=" * 70)
        for f in failures:
            print(f"  - {f}")
        print()
        print("Action: Generate more synthetic data and re-split.")
        print("        Do NOT shrink val/test percentages to force the numbers.")
        sys.exit(1)
    else:
        print("=" * 70)
        print("  RESULT: PASS — all checks passed")
        print("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
