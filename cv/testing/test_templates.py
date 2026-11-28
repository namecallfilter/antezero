import unittest
import cv2
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config
from template import template_loader
import eval_template


class TestTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config()
        
        # Disable debug plotting during automated tests to avoid popups
        cls.config.debug_plot = False
        cls.config.setup_mode = False
        
        # Load all templates using the existing application loader
        cls.templates = template_loader(
            cls.config.template_json_path,
            cls.config.template_img_dir,
            cls.config.resolution
        )
        
        # Load the raw JSON to parse the newly annotated test_frames
        json_path = Path(cls.config.template_json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Template JSON not found at: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            cls.templates_data = json.load(f)

    def test_annotated_frames(self):
        """
        Dynamically run tests for every template and every test_frame assigned to it.
        """
        template_dict = {t.name: t for t in self.templates}
        
        tests_run = 0
        for template_name, data in self.templates_data.items():
            test_frames = data.get("test_frames", [])
            
            if not test_frames:
                continue
                
            template = template_dict.get(template_name)
            self.assertIsNotNone(template, f"Template '{template_name}' missing from template_loader output")
            
            for tf in test_frames:
                path = tf["path"]
                expected_pressed = tf["pressed"]
                frame_idx = tf.get("frame_idx", "unknown")
                
                # Using subTest so a single failure doesn't abort the entire loop
                with self.subTest(template=template_name, frame=frame_idx, path=path):
                    tests_run += 1
                    
                    img_path = Path(path)
                    self.assertTrue(img_path.exists(), f"Test image does not exist: {path}")
                    
                    img = cv2.imread(str(img_path))
                    self.assertIsNotNone(img, f"Failed to read image via cv2: {path}")
                    
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Run the core evaluation logic
                    result = eval_template.eval(
                        template, 
                        img, 
                        img_gray,
                        self.config, 
                        video_timestamp_str=f"Frame {frame_idx}"
                    )
                    
                    # For expected_pressed=False, we only fail on false positive
                    if expected_pressed is False:
                        if result.found and result.clicked:
                            self.fail(f"False positive: Algorithm incorrectly found '{template_name}' as pressed in frame {frame_idx}.")
                    else:
                        # Assert template found
                        self.assertTrue(
                            result.found, 
                            f"Expected template '{template_name}' to be found in frame {frame_idx}"
                        )
                        
                        # Assert 'pressed'/'clicked' state
                        self.assertEqual(
                            result.clicked, 
                            expected_pressed, 
                            f"Mismatch in pressed state for '{template_name}' at frame {frame_idx}. "
                            f"Expected {expected_pressed}, got {result.clicked}."
                        )
                    
        # Basic sanity check to ensure we didn't quietly skip everything
        self.assertGreater(tests_run, 0, "No test_frames found to execute.")

if __name__ == "__main__":
    unittest.main()
