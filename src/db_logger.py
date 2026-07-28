import sqlite3
from pathlib import Path
from ultralytics import YOLO

DB_PATH = Path(__file__).resolve().parent.parent / "classifications.db"

def init_db():
    """Initialize SQLite database table for storing image table classifications."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS table_classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            confidence REAL NOT NULL,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_classification(image_path, class_id, class_name, confidence, bbox=None):
    """Save a single detection result into the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    x1, y1, x2, y2 = bbox if bbox else (None, None, None, None)
    cursor.execute("""
        INSERT INTO table_classifications (image_path, class_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (str(image_path), class_id, class_name, confidence, x1, y1, x2, y2))
    conn.commit()
    conn.close()

def classify_and_store(model, image_path, conf=0.25, iou=0.45):
    """Run model inference on an image and log all table classifications into the database."""
    init_db()
    results = model.predict(image_path, conf=conf, iou=iou, imgsz=1024, save=True)
    
    saved_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names[class_id]
            coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            save_classification(image_path, class_id, class_name, confidence, coords)
            saved_count += 1
            
    print(f"Logged {saved_count} table classification(s) for '{image_path}' into database '{DB_PATH.name}'.")
    return saved_count

def fetch_all_classifications():
    """Retrieve all logged classifications from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path, class_name, confidence, timestamp FROM table_classifications ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at: {DB_PATH}")
    records = fetch_all_classifications()
    print(f"Total stored classification records: {len(records)}")
