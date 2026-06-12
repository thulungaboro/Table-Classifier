"""
YOLO Dataset Splitter
---------------------
This program takes a YOLO dataset (images + label .txt files) stored in a single
folder and splits it into training and validation sets at an 80/20 ratio.

What it does, step by step:
  1. Reads all images from the images/ folder inside the dataset.
  2. Keeps only images that have a matching .txt label file in labels/.
  3. Shuffles the pairs randomly (so the split is not biased by file order).
  4. Puts 80% of the pairs into train/ and the remaining 20% into val/.
  5. COPIES (never moves or deletes) files into a brand new output folder
     that sits OUTSIDE and next to the original dataset folder.

IMPORTANT — original dataset is never touched:
  All output is written to a new folder named "<dataset>_split" placed
  next to your original dataset. The original files are never modified.

Output folder structure produced:
  <parent>/
    dataset/             <- your original folder, completely untouched
    dataset_split/       <- new folder created next to the original
      images/
        train/           <- 80% of images go here
        val/             <- 20% of images go here
      labels/
        train/           <- matching labels for train images
        val/             <- matching labels for val images

Usage:
  python split_dataset.py "path/to/dataset"
"""

import shutil
import random
from pathlib import Path


def split_dataset(
    dataset_dir: str,
    images_subdir: str = "images",   # name of the folder that holds images
    labels_subdir: str = "labels",   # name of the folder that holds .txt labels
    val_ratio: float = 0.2,          # fraction of data to use for validation (0.2 = 20%)
    seed: int = 42,                  # fixed seed so the same split is produced every run
    image_exts: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),  # accepted image formats
):
    # convert the dataset path string into a Path object for easy joining
    dataset = Path(dataset_dir)

    # build full paths to the source images and labels folders (inside the original dataset)
    src_images = dataset / images_subdir
    src_labels = dataset / labels_subdir

    # stop early if the images folder does not exist
    if not src_images.exists():
        raise FileNotFoundError(f"Images folder not found: {src_images}")

    # stop early if the labels folder does not exist
    if not src_labels.exists():
        raise FileNotFoundError(f"Labels folder not found: {src_labels}")

    # create the output root folder next to (not inside) the original dataset
    # e.g. if dataset is "D:/data/dataset", output goes to "D:/data/dataset_split"
    output_dir = dataset.parent / f"{dataset.name}_split"

    # build output paths for images and labels inside the new split folder
    out_images = output_dir / images_subdir
    out_labels = output_dir / labels_subdir

    # collect every file in src_images whose extension is an accepted image format
    image_files = [f for f in src_images.iterdir() if f.suffix.lower() in image_exts]

    # keep only images that have a corresponding .txt label file (same base name)
    paired = [f for f in image_files if (src_labels / f.with_suffix(".txt").name).exists()]

    # find images that have no label file so we can warn the user
    unpaired = [f for f in image_files if f not in paired]

    # warn the user about images that will be skipped due to missing labels
    if unpaired:
        print(f"Warning: {len(unpaired)} image(s) have no matching label and will be skipped.")

    # if nothing is paired at all, there is nothing to split — raise an error
    if not paired:
        raise ValueError("No paired image+label files found. Check your folder paths.")

    # fix the random seed so the shuffle gives the same result every time
    random.seed(seed)

    # shuffle the list so images are in random order before splitting
    random.shuffle(paired)

    # calculate how many samples go into the val set (at least 1)
    val_count = max(1, int(len(paired) * val_ratio))

    # the first val_count items become the validation set
    val_files = paired[:val_count]

    # everything after val_count becomes the training set
    train_files = paired[val_count:]

    # print a summary so the user can confirm the numbers look right
    print(f"Total paired samples : {len(paired)}")
    print(f"Train                : {len(train_files)}")
    print(f"Val                  : {len(val_files)}")

    # group train and val together so we can process both in one loop
    splits = {"train": train_files, "val": val_files}

    for split, files in splits.items():
        # create output/images/train/ or output/images/val/ outside the original dataset
        (out_images / split).mkdir(parents=True, exist_ok=True)

        # create output/labels/train/ or output/labels/val/ outside the original dataset
        (out_labels / split).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            # derive the label path from the image filename (same name, .txt extension)
            lbl_path = src_labels / img_path.with_suffix(".txt").name

            # COPY (not move) the image into the output split folder — source untouched
            shutil.copy2(img_path, out_images / split / img_path.name)

            # COPY (not move) the label into the output split folder — source untouched
            shutil.copy2(lbl_path, out_labels / split / lbl_path.name)

    # print the final folder layout so the user knows where to find the output
    print(f"\nDone. Original dataset untouched at: {dataset}")
    print(f"Split output written to            : {output_dir}")
    print(f"  {out_images}/train/  ({len(train_files)} images)")
    print(f"  {out_images}/val/    ({len(val_files)} images)")
    print(f"  {out_labels}/train/  ({len(train_files)} labels)")
    print(f"  {out_labels}/val/    ({len(val_files)} labels)")


if __name__ == "__main__":
    import argparse

    # set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Split a YOLO dataset into train/val sets.")

    # required argument: path to the dataset root folder
    parser.add_argument("dataset_dir", help="Root folder containing images/ and labels/ subdirectories")

    # optional: override the default subfolder names if yours differ
    parser.add_argument("--images-subdir", default="images", help="Name of the images subfolder (default: images)")
    parser.add_argument("--labels-subdir", default="labels", help="Name of the labels subfolder (default: labels)")

    # optional: change the validation ratio (e.g. 0.1 for 90/10 split)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation set (default: 0.2)")

    # optional: change the random seed for a different shuffle
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")

    # parse whatever the user typed on the command line
    args = parser.parse_args()

    # call the main function with the parsed arguments
    split_dataset(
        dataset_dir=args.dataset_dir,
        images_subdir=args.images_subdir,
        labels_subdir=args.labels_subdir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


...