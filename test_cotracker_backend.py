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

    def test_downscaling_coordinate_math(self):
        orig_w, orig_h = 3840, 2160
        max_size = 720
        scale = float(max_size) / float(max(orig_w, orig_h))
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        scale_x = float(new_w) / float(orig_w)
        scale_y = float(new_h) / float(orig_h)

        test_points = [(150.25, 300.75), (1920.0, 1080.0), (3800.5, 2100.2)]
        for orig_x, orig_y in test_points:
            scaled_x = orig_x * scale_x
            scaled_y = orig_y * scale_y
            recovered_x = scaled_x / scale_x
            recovered_y = scaled_y / scale_y
            self.assertAlmostEqual(orig_x, recovered_x, places=5)
            self.assertAlmostEqual(orig_y, recovered_y, places=5)

    def test_chunk_planning_logic(self):
        total_frames = 250
        chunk_size = 100
        ref_idx = 120

        chunks_plan = []
        if total_frames <= chunk_size:
            chunks_plan.append({"type": "single", "start": 0, "end": total_frames})
        else:
            if ref_idx < total_frames - 1:
                fwd_cur = ref_idx
                while fwd_cur < total_frames - 1:
                    fwd_end = min(total_frames, fwd_cur + chunk_size)
                    chunks_plan.append({"type": "forward", "start": fwd_cur, "end": fwd_end})
                    fwd_cur = fwd_end - 1
            if ref_idx > 0:
                bwd_cur = ref_idx
                while bwd_cur > 0:
                    bwd_start = max(0, bwd_cur - chunk_size + 1)
                    chunks_plan.append({"type": "backward", "start": bwd_start, "end": bwd_cur + 1})
                    bwd_cur = bwd_start

        self.assertGreater(len(chunks_plan), 1)
        # All frames in sequence are covered
        fwd_coverage = set()
        for p in chunks_plan:
            if p["type"] == "forward":
                for f in range(p["start"], p["end"]):
                    fwd_coverage.add(f)
        for f in range(ref_idx, total_frames):
            self.assertIn(f, fwd_coverage)

        bwd_coverage = set()
        for p in chunks_plan:
            if p["type"] == "backward":
                for f in range(p["start"], p["end"]):
                    bwd_coverage.add(f)
        for f in range(0, ref_idx + 1):
            self.assertIn(f, bwd_coverage)

if __name__ == '__main__':
    unittest.main()
