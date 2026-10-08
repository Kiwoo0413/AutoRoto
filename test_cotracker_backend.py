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
        # Nuke bottom-left to image top-left (exact continuous subpixel inversion)
        y_img = h - y_nuke
        self.assertEqual(y_img, 730.0)
        # Invert back to Nuke bottom-left
        y_nuke_back = h - y_img
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

        # Test bidirectional chunk plan
        fwd_chunks = []
        if ref_idx < total_frames - 1:
            fwd_cur = ref_idx
            while fwd_cur < total_frames - 1:
                fwd_end = min(total_frames, fwd_cur + chunk_size)
                fwd_chunks.append({"dir": "forward", "start_orig": fwd_cur, "end_orig": fwd_end})
                fwd_cur = fwd_end - 1

        bwd_chunks = []
        if ref_idx > 0:
            bwd_total = ref_idx + 1
            bwd_cur = 0
            while bwd_cur < bwd_total - 1:
                bwd_end = min(bwd_total, bwd_cur + chunk_size)
                bwd_chunks.append({"dir": "backward", "bwd_start": bwd_cur, "bwd_end": bwd_end})
                bwd_cur = bwd_end - 1

        all_chunks = fwd_chunks + bwd_chunks
        self.assertGreater(len(all_chunks), 1)

        # Verify all frames 0 to total_frames - 1 are covered
        visited = set()
        visited.add(ref_idx)
        for chunk in fwd_chunks:
            for f in range(chunk["start_orig"], chunk["end_orig"]):
                visited.add(f)
        for chunk in bwd_chunks:
            for b in range(chunk["bwd_start"], chunk["bwd_end"]):
                visited.add(ref_idx - b)

        for f in range(total_frames):
            self.assertIn(f, visited)

    def test_custom_temp_dir_handling(self):
        import tempfile
        import shutil
        custom_base = tempfile.mkdtemp(prefix="test_autoroto_cache_")
        try:
            res = tracker_core.run_cotracker_point_tracking(
                image_paths=[],
                queries=[],
                start_frame=0,
                temp_dir_base=custom_base
            )
            self.assertEqual(res, {})
        finally:
            if os.path.exists(custom_base):
                shutil.rmtree(custom_base, ignore_errors=True)

if __name__ == '__main__':
    unittest.main()
