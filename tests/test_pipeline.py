"""
Unit tests for dataset generation, script imports, and database logger.
"""

import unittest
import sqlite3
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class TestPipeline(unittest.TestCase):

    def test_imports(self):
        """Verify that all project source modules can be imported cleanly."""
        import src.dataset_builder as db
        import src.dataset_generator as dg
        import src.dataset_verifier as dv
        import src.db_logger as dbl
        self.assertIsNotNone(db)
        self.assertIsNotNone(dg)
        self.assertIsNotNone(dv)
        self.assertIsNotNone(dbl)

    def test_synthetic_page_generation(self):
        """Verify that synthetic generator creates a page image and bounding boxes."""
        from src.dataset_generator import draw_synthetic_page
        img, boxes = draw_synthetic_page(is_bordered=True, is_negative=False)

        self.assertIsNotNone(img)
        self.assertEqual(img.size, (1240, 1754))
        self.assertGreater(len(boxes), 0)

        # Check YOLO box normalization format: [class_id, x_center, y_center, width, height]
        for box in boxes:
            cls_id, xc, yc, w, h = box
            self.assertIn(cls_id, (0, 1))
            self.assertTrue(0.0 <= xc <= 1.0)
            self.assertTrue(0.0 <= yc <= 1.0)
            self.assertTrue(0.0 < w <= 1.0)
            self.assertTrue(0.0 < h <= 1.0)

    def test_db_logger_schema(self):
        """Test SQLite database initialization and logging logic."""
        conn = sqlite3.connect(":memory:")
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
        cursor.execute("""
            INSERT INTO table_classifications (image_path, class_id, class_name, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES ('test.png', 0, 'bordered', 0.95, 10.0, 20.0, 100.0, 200.0)
        """)
        conn.commit()

        cursor.execute("SELECT class_name, confidence FROM table_classifications")
        row = cursor.fetchone()
        conn.close()

        self.assertEqual(row[0], 'bordered')
        self.assertEqual(row[1], 0.95)


if __name__ == "__main__":
    unittest.main()
