"""
Combine a REAL annotated pool and a SYNTHETIC generated pool into one
train/val/test dataset, using different split ratios per source:
  - real data is scarce and is your only real-world signal -> weighted to val/test
  - synthetic data is cheap/expandable -> weighted to train

Stratifies by each image's dominant class (or "negative" for no-table pages)
so class balance and negative-page representation hold across splits.

Writes a manifest.csv (image, split, source, strat_key) so you can later
score the real-only and synthetic-only subsets of val/test separately --
a pooled metric will hide whether the model actually works on real docs,
since synthetic will usually outnumber real by a wide margin.

Usage:
  python build_combined_dataset.py ^
      --real_images D:/Project-1-Tb-DTC/dataset/images/train D:/Project-1-Tb-DTC/dataset/images/val ^
      --real_labels D:/Project-1-Tb-DTC/dataset/labels/train D:/Project-1-Tb-DTC/dataset/labels/val ^
      --synth_images D:/box4/images/train D:/box4/images/val ^
      --synth_labels D:/box4/labels/train D:/box4/labels/val ^
      --out D:/Project-1-Tb-DTC/dataset_combined
"""
import argparse
import random
import shutil
import csv
from pathlib import Path
from collections import defaultdict

IMG_EXTS = (".png", ".jpg", ".jpeg")


def read_label_classes(label_path):
    """Return list of integer class ids found in a YOLO label file."""
    if not label_path.exists():
        return []
    text = label_path.read_text().strip()
    if not text:
        return []
    return [int(line.split()[0]) for line in text.splitlines()]


def dominant_key(classes):
    """Stratification key: 'negative' if no annotations, else 'class_<id>'
    where <id> is the most frequent class in the label file."""
    if not classes:
        return "negative"
    counts = defaultdict(int)
    for c in classes:
        counts[c] += 1
    return f"class_{max(counts, key=counts.get)}"


def collect_pool(images_dirs, labels_dirs, source_tag):
    """Walk one or more image/label directory pairs and build a flat list of
    dicts, each carrying paths, source tag, and stratification key."""
    pool = []
    for images_dir, labels_dir in zip(images_dirs, labels_dirs):
        for img_path in sorted(Path(images_dir).glob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            label_path = Path(labels_dir) / f"{img_path.stem}.txt"
            classes = read_label_classes(label_path)
            pool.append({
                "image": img_path,
                "label": label_path,
                "source": source_tag,
                "strat_key": dominant_key(classes),
            })
    return pool


def stratified_split(pool, ratios, seed=42):
    """Split a pool into train/val/test using per-stratum shuffled cuts.
    ratios: (train, val, test) fractions, summing to ~1.0."""
    rnd = random.Random(seed)
    by_key = defaultdict(list)
    for item in pool:
        by_key[item["strat_key"]].append(item)

    splits = {"train": [], "val": [], "test": []}
    for key in sorted(by_key):  # sorted for reproducibility
        items = by_key[key]
        rnd.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # remainder goes to test (avoids rounding-loss)
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train:n_train + n_val])
        splits["test"].extend(items[n_train + n_val:])
    return splits


def write_split(splits, out_root):
    """Copy images + labels into out_root/{images,labels}/{train,val,test}/
    and write manifest.csv for later source-aware evaluation."""
    out_root = Path(out_root)
    manifest_rows = []
    for split, items in splits.items():
        img_dir = out_root / "images" / split
        lbl_dir = out_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for item in items:
            shutil.copy2(item["image"], img_dir / item["image"].name)
            dest_label = lbl_dir / f"{item['image'].stem}.txt"
            if item["label"].exists():
                shutil.copy2(item["label"], dest_label)
            else:
                # negative page — write empty label so YOLO still sees the image
                dest_label.write_text("")
            manifest_rows.append({
                "image": item["image"].name,
                "split": split,
                "source": item["source"],
                "strat_key": item["strat_key"],
            })
    # write manifest
    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "split", "source", "strat_key"])
        w.writeheader()
        w.writerows(sorted(manifest_rows, key=lambda r: (r["split"], r["source"], r["image"])))
    print(f"Manifest written to {manifest_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Merge real + synthetic pools into a combined train/val/test dataset."
    )
    ap.add_argument("--real_images", nargs="+", required=True,
                     help="One or more directories containing real images (e.g. dataset/images/train dataset/images/val)")
    ap.add_argument("--real_labels", nargs="+", required=True,
                     help="Matching label directories, same order as --real_images")
    ap.add_argument("--synth_images", nargs="+", required=True,
                     help="One or more directories containing synthetic images")
    ap.add_argument("--synth_labels", nargs="+", required=True,
                     help="Matching label directories, same order as --synth_images")
    ap.add_argument("--out", default="dataset_combined",
                     help="Output directory for the combined dataset (default: dataset_combined)")
    ap.add_argument("--seed", type=int, default=42,
                     help="Random seed for reproducibility (default: 42)")
    args = ap.parse_args()

    # Validate matching dir counts
    if len(args.real_images) != len(args.real_labels):
        ap.error("--real_images and --real_labels must have the same number of directories")
    if len(args.synth_images) != len(args.synth_labels):
        ap.error("--synth_images and --synth_labels must have the same number of directories")

    real_pool = collect_pool(args.real_images, args.real_labels, "real")
    synth_pool = collect_pool(args.synth_images, args.synth_labels, "synthetic")

    print(f"Real pool  : {len(real_pool)} images")
    print(f"Synth pool : {len(synth_pool)} images")
    print()

    # Real is scarce — skew toward val/test so evaluation is grounded in reality
    real_splits = stratified_split(real_pool, ratios=(0.30, 0.35, 0.35), seed=args.seed)
    # Synthetic is cheap/expandable — skew toward train
    synth_splits = stratified_split(synth_pool, ratios=(0.70, 0.15, 0.15), seed=args.seed)

    combined = {k: real_splits[k] + synth_splits[k] for k in ("train", "val", "test")}
    write_split(combined, args.out)

    # Summary
    print()
    print(f"{'Split':<8} {'Real':>6} {'Synth':>6} {'Total':>6}")
    print("-" * 30)
    for split in ("train", "val", "test"):
        n_real = sum(1 for i in combined[split] if i["source"] == "real")
        n_synth = sum(1 for i in combined[split] if i["source"] == "synthetic")
        print(f"{split:<8} {n_real:>6} {n_synth:>6} {n_real + n_synth:>6}")
    print()
    print(f"Written to {args.out}/ with manifest.csv for later real-vs-synthetic breakdown.")


if __name__ == "__main__":
    main()
