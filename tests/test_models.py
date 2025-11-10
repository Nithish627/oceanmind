import torch
import unittest
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.coral_reef_detector import CoralReefDetector
from models.fish_detector import FishDetector
from models.illegal_fishing_detector import IllegalFishingDetector

class TestModels(unittest.TestCase):
    def test_coral_reef_detector(self):
        model = CoralReefDetector(num_classes=4)
        dummy_input = torch.randn(1, 3, 640, 640)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 4))
    
    def test_fish_detector(self):
        model = FishDetector(num_species=10)
        dummy_input = torch.randn(1, 3, 640, 640)
        bbox_output, class_output = model(dummy_input)
        self.assertEqual(bbox_output.shape, (1, 5))
        self.assertEqual(class_output.shape, (1, 10))
    
    def test_illegal_fishing_detector(self):
        model = IllegalFishingDetector(num_classes=3)
        dummy_input = torch.randn(1, 10, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 3))

if __name__ == '__main__':
    unittest.main()