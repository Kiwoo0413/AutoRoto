"""
Unit Test Suite for AutoRoto CoTracker GPU Backend
"""

import os
import sys
import unittest
import numpy as np

import tracker_core

class TestCoTrackerBackend(unittest.TestCase):

    def test_ai_python_discovery(self):
        py_exe = tracker_core.find_ai_python()
        self.assertIsNotNone(py_exe)
        self.assertTrue(os.path.isfile(py_exe))

    def test_ai_environment_status(self):
        info = tracker_core.check_ai_environment()
        self.assertTrue(info['available'])
        self.assertTrue(info['cuda'])
        self.assertIn("RTX 4080", info['device_name'])
        self.assertIn("2.4", info['torch_version'])

    def test_coordinate_conversion(self):
        # Image height H = 1080
        h = 1080
        y_nuke = 350.0
        # Nuke bottom-left to image top-left
        y_img = (h - 1) - y_nuke
        self.assertEqual(y_img, 729.0)
        # Invert back to Nuke bottom-left
        y_nuke_back = (h - 1) - y_img
        self.assertEqual(y_nuke_back, 350.0)

if __name__ == '__main__':
    unittest.main()
