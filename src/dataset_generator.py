"""
Synthetic Table Dataset Generator for YOLOv8 Table Detection
--------------------------------------------------------------
Generates synthetic document pages containing bordered or borderless tables
along with YOLO format bounding box annotations (0: bordered, 1: borderless).

Requirements:
    pip install pillow numpy reportlab
"""

import os
import random
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Canvas dimensions (A4 @ ~150 DPI)
PAGE_WIDTH = 1240
PAGE_HEIGHT = 1754

CLASS_BORDERED = 0
CLASS_BORDERLESS = 1

LIPSUM_WORDS = [
    "Lorem", "ipsum", "dolor", "sit", "amet,", "consectetur", "adipiscing", "elit.",
    "Item", "Description", "Quantity", "Price", "Total", "Status", "Date", "Category",
    "Amount", "Tax", "Discount", "Invoice", "Reference", "Subtotal", "Balance", "Notes"
]

def get_random_text():
    return " ".join(random.choices(LIPSUM_WORDS, k=random.randint(1, 3)))

def draw_synthetic_page(is_bordered=True, is_negative=False):
    """Draw a synthetic document page and return PIL Image + list of YOLO bbox annotations."""
    # Blank white page
    img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw header text to simulate document content
    y_cursor = random.randint(80, 150)
    for _ in range(random.randint(2, 5)):
        draw.text((100, y_cursor), get_random_text() + " " + get_random_text(), fill=(50, 50, 50))
        y_cursor += random.randint(25, 40)

    boxes = []
    if is_negative:
        # Draw document paragraphs without any table
        for _ in range(random.randint(10, 20)):
            draw.text((100, y_cursor), " ".join(random.choices(LIPSUM_WORDS, k=10)), fill=(80, 80, 80))
            y_cursor += 30
        return img, boxes

    # Generate 1 to 2 tables on page
    num_tables = random.randint(1, 2)
    for _ in range(num_tables):
        if y_cursor + 300 >= PAGE_HEIGHT - 100:
            break

        rows = random.randint(4, 12)
        cols = random.randint(3, 7)
        
        table_width = random.randint(700, 1000)
        table_left = (PAGE_WIDTH - table_width) // 2
        row_height = random.randint(35, 50)
        table_height = rows * row_height
        table_top = y_cursor + random.randint(20, 50)

        col_width = table_width / cols

        # Draw cell contents and borders
        for r in range(rows):
            cell_top = table_top + r * row_height
            for c in range(cols):
                cell_left = int(table_left + c * col_width)
                cell_right = int(cell_left + col_width)
                cell_bottom = cell_top + row_height

                # Draw cell text
                text = get_random_text()
                draw.text((cell_left + 8, cell_top + 8), text, fill=(0, 0, 0))

                # Draw grid lines if bordered
                if is_bordered:
                    draw.rectangle([cell_left, cell_top, cell_right, cell_bottom], outline=(100, 100, 100), width=1)

        # Draw header row background shading optionally
        if random.random() > 0.5:
            draw.rectangle([table_left, table_top, table_left + table_width, table_top + row_height], fill=None, outline=(0, 0, 0), width=1)

        # Calculate normalized YOLO bbox: [class_id, x_center, y_center, width, height]
        x_center = (table_left + table_width / 2.0) / PAGE_WIDTH
        y_center = (table_top + table_height / 2.0) / PAGE_HEIGHT
        norm_w = table_width / float(PAGE_WIDTH)
        norm_h = table_height / float(PAGE_HEIGHT)

        class_id = CLASS_BORDERED if is_bordered else CLASS_BORDERLESS
        boxes.append((class_id, x_center, y_center, norm_w, norm_h))

        y_cursor = table_top + table_height + 40

    return img, boxes

def generate_dataset(output_dir, num_samples=100, negative_ratio=0.15):
    """Generate synthetic image and YOLO txt label pairs into output_dir/images and output_dir/labels."""
    out_path = Path(output_dir)
    img_dir = out_path / "images"
    lbl_dir = out_path / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_samples} synthetic document pages into '{output_dir}'...")

    for i in range(1, num_samples + 1):
        filename = f"synth_{i:05d}"
        
        # Decide page type
        r = random.random()
        if r < negative_ratio:
            is_negative = True
            is_bordered = False
        else:
            is_negative = False
            is_bordered = (random.random() > 0.5)

        img, boxes = draw_synthetic_page(is_bordered=is_bordered, is_negative=is_negative)
        
        # Save image
        img.save(img_dir / f"{filename}.png")

        # Save YOLO annotation file
        label_file = lbl_dir / f"{filename}.txt"
        with open(label_file, "w") as f:
            for cls_id, xc, yc, w, h in boxes:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    print(f"Generation complete! {num_samples} pages written to '{output_dir}'.")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic table dataset for YOLO object detection.")
    parser.add_argument("--out", default="synth_dataset", help="Output directory for generated dataset")
    parser.add_argument("--num", type=int, default=100, help="Number of synthetic pages to generate")
    parser.add_argument("--neg_ratio", type=float, default=0.15, help="Fraction of negative (no-table) pages")
    args = parser.parse_args()

    generate_dataset(args.out, args.num, args.neg_ratio)

if __name__ == "__main__":
    main()
