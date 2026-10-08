"""
Unit Test Suite for AutoRoto Multi-Reference Tracking & Bidirectional Blending
"""

import unittest
import numpy as np

def blend_bidirectional_trajectories(fwd_tracks, bwd_tracks, start_k, end_k):
    """
    Blends forward and backward trajectories between keyframes start_k and end_k
    using C1 smoothstep and CoTracker visibility weighting.
    Boundary guarantee:
      f == start_k -> exact fwd_tracks (100% match to start_k ground truth)
      f == end_k   -> exact bwd_tracks (100% match to end_k ground truth)
    """
    blended = {}
    span = end_k - start_k

    all_pts = set(fwd_tracks.keys()).intersection(set(bwd_tracks.keys()))
    for pt_idx in all_pts:
        blended[pt_idx] = {}
        pt_fwd = fwd_tracks[pt_idx]
        pt_bwd = bwd_tracks[pt_idx]

        for f in range(start_k, end_k + 1):
            if f not in pt_fwd or f not in pt_bwd:
                continue

            xf, yf, vf = pt_fwd[f]
            xb, yb, vb = pt_bwd[f]

            if span <= 0:
                blended[pt_idx][f] = (xf, yf, max(vf, vb))
                continue

            t = float(f - start_k) / float(span)
            t = max(0.0, min(1.0, t))
            # C1 smoothstep: s(0)=0, s(1)=1, s'(0)=0, s'(1)=0
            s = 3.0 * (t ** 2) - 2.0 * (t ** 3)

            w_a = (1.0 - s) * (max(0.0, float(vf)) + 0.05)
            w_b = s * (max(0.0, float(vb)) + 0.05)
            w_sum = w_a + w_b

            if w_sum < 1e-6:
                bx = 0.5 * (xf + xb)
                by = 0.5 * (yf + yb)
            else:
                bx = (w_a * xf + w_b * xb) / w_sum
                by = (w_a * yf + w_b * yb) / w_sum

            # Exact boundary snap to guarantee 100% fidelity on ground truth keyframes
            if f == start_k:
                bx, by = xf, yf
            elif f == end_k:
                bx, by = xb, yb

            b_vis = max(float(vf), float(vb))
            blended[pt_idx][f] = (bx, by, b_vis)

    return blended


def parse_keyframe_list_string(text: str):
    """
    Parses comma- or whitespace-separated keyframe list into sorted unique integers.
    """
    if not text or not str(text).strip():
        return []
    import re
    tokens = re.split(r'[,;\s]+', str(text).strip())
    frames = set()
    for tok in tokens:
        if not tok:
            continue
        try:
            frames.add(int(tok))
        except ValueError:
            pass
    return sorted(list(frames))


class TestMultiRefTracking(unittest.TestCase):

    def test_keyframe_string_parser(self):
        s1 = "1, 45, 90"
        self.assertEqual(parse_keyframe_list_string(s1), [1, 45, 90])
        s2 = " 10  25,50 ; 75  "
        self.assertEqual(parse_keyframe_list_string(s2), [10, 25, 50, 75])
        s3 = "invalid, 100, abc, 50"
        self.assertEqual(parse_keyframe_list_string(s3), [50, 100])
        self.assertEqual(parse_keyframe_list_string(""), [])

    def test_boundary_exactness(self):
        # Keyframe A at 10 (pos: 100, 200), Keyframe B at 30 (pos: 300, 400)
        start_k = 10
        end_k = 30

        fwd_tracks = {
            0: {f: (100.0 + (f - 10) * 8.0, 200.0 + (f - 10) * 8.0, 1.0) for f in range(10, 31)}
        }
        bwd_tracks = {
            0: {f: (300.0 - (30 - f) * 11.0, 400.0 - (30 - f) * 11.0, 1.0) for f in range(10, 31)}
        }

        blended = blend_bidirectional_trajectories(fwd_tracks, bwd_tracks, start_k, end_k)

        # Boundary at start_k must be EXACTLY keyframe A
        self.assertAlmostEqual(blended[0][10][0], 100.0, places=5)
        self.assertAlmostEqual(blended[0][10][1], 200.0, places=5)

        # Boundary at end_k must be EXACTLY keyframe B
        self.assertAlmostEqual(blended[0][30][0], 300.0, places=5)
        self.assertAlmostEqual(blended[0][30][1], 400.0, places=5)

    def test_smoothness_across_interval(self):
        start_k = 0
        end_k = 100
        # Forward starts at 0 and drifts by +2.0 by frame 100
        fwd_tracks = {0: {f: (float(f) + (f / 100.0) * 2.0, float(f) + (f / 100.0) * 2.0, 1.0) for f in range(0, 101)}}
        # Backward starts at 100 at (100, 100) and drifts by -2.0 as it moves backward to frame 0
        bwd_tracks = {0: {f: (float(f) - ((100.0 - f) / 100.0) * 2.0, float(f) - ((100.0 - f) / 100.0) * 2.0, 1.0) for f in range(0, 101)}}

        blended = blend_bidirectional_trajectories(fwd_tracks, bwd_tracks, start_k, end_k)

        # Check monotonicity, continuity, and exact boundary preservation
        prev_x = blended[0][0][0]
        self.assertEqual(prev_x, 0.0)
        for f in range(1, 101):
            cur_x = blended[0][f][0]
            self.assertGreaterEqual(cur_x, prev_x)
            diff = cur_x - prev_x
            self.assertLess(diff, 2.0)
            prev_x = cur_x
        self.assertEqual(blended[0][100][0], 100.0)

    def test_visibility_weighting_adaptation(self):
        start_k = 0
        end_k = 20

        # Feature gets occluded moving forward (fwd vis drops to 0.0 at frame 10)
        fwd_tracks = {
            0: {f: (float(f), float(f), 1.0 if f < 10 else 0.0) for f in range(0, 21)}
        }
        # Feature is fully visible backward
        bwd_tracks = {
            0: {f: (float(f) + 2.0, float(f) + 2.0, 1.0) for f in range(0, 21)}
        }
        bwd_tracks[0][20] = (20.0, 20.0, 1.0)

        blended = blend_bidirectional_trajectories(fwd_tracks, bwd_tracks, start_k, end_k)

        # At frame 15, fwd is occluded (vis=0), bwd is visible (vis=1)
        # Blended position should heavily favor bwd_tracks
        val_at_15 = blended[0][15][0]
        bwd_val_at_15 = bwd_tracks[0][15][0]
        self.assertAlmostEqual(val_at_15, bwd_val_at_15, delta=0.5)

    def test_in_between_bake_frames_selection(self):
        user_keyframes = {10, 50, 90}
        all_frames = set(range(10, 91))

        # In-between frames are all frames EXCLUDING user_keyframes
        in_between = all_frames - user_keyframes
        self.assertEqual(len(in_between), 81 - 3)
        self.assertNotIn(10, in_between)
        self.assertNotIn(50, in_between)
        self.assertNotIn(90, in_between)
        self.assertIn(11, in_between)
        self.assertIn(49, in_between)
        self.assertIn(51, in_between)

    def test_key_step_bake_selection(self):
        user_keyframes = {10, 50}
        first_ref = 10
        min_f = 10
        max_f = 50
        key_step = 5

        bake_frames = set()
        for f in range(min_f, max_f + 1):
            if (f - first_ref) % key_step == 0:
                bake_frames.add(f)
        bake_frames.update(user_keyframes)

        expected = {10, 15, 20, 25, 30, 35, 40, 45, 50}
        self.assertEqual(bake_frames, expected)

        # Non-overwriting filter:
        to_overwrite = [f for f in bake_frames if f not in user_keyframes]
        self.assertEqual(sorted(to_overwrite), [15, 20, 25, 30, 35, 40, 45])
        self.assertNotIn(10, to_overwrite)
        self.assertNotIn(50, to_overwrite)

    def test_multi_interval_master_tracks_assembly(self):
        # Timeline: 1 to 60, user keys at 10 and 40
        ref_keys = [10, 40]
        min_needed = 1
        max_needed = 60

        master_tracks = {0: {}}

        # 1. Pre-interval [1, 10]
        for f in range(1, 11):
            master_tracks[0][f] = (float(f * 2), float(f * 2), 1.0)

        # 2. In-between interval [10, 40]
        fwd_10_40 = {0: {f: (20.0 + (f - 10) * 1.0, 20.0 + (f - 10) * 1.0, 1.0) for f in range(10, 41)}}
        bwd_10_40 = {0: {f: (50.0 - (40 - f) * 1.0, 50.0 - (40 - f) * 1.0, 1.0) for f in range(10, 41)}}
        blended = blend_bidirectional_trajectories(fwd_10_40, bwd_10_40, 10, 40)
        for f, val in blended[0].items():
            master_tracks[0][f] = val

        # 3. Post-interval [40, 60]
        for f in range(40, 61):
            master_tracks[0][f] = (50.0 + (f - 40) * 2.0, 50.0 + (f - 40) * 2.0, 1.0)

        # Check coverage
        self.assertEqual(len(master_tracks[0]), 60)
        self.assertEqual(master_tracks[0][1], (2.0, 2.0, 1.0))
        self.assertEqual(master_tracks[0][10], (20.0, 20.0, 1.0))
        self.assertEqual(master_tracks[0][40], (50.0, 50.0, 1.0))
        self.assertEqual(master_tracks[0][60], (90.0, 90.0, 1.0))

    def test_unified_ref_frame_routing(self):
        # 1 frame -> single tracking mode
        frames_1 = parse_keyframe_list_string("42")
        self.assertEqual(len(frames_1), 1)
        self.assertEqual(frames_1, [42])

        # 2+ frames -> multi-reference tracking mode
        frames_multi = parse_keyframe_list_string("10, 45, 90")
        self.assertEqual(len(frames_multi), 3)
        self.assertEqual(frames_multi, [10, 45, 90])

        # Add current frame logic
        curr = 60
        combined = sorted(list(set(frames_1 + [curr])))
        self.assertEqual(combined, [42, 60])
        self.assertGreaterEqual(len(combined), 2)

        # Clear logic
        cleared = parse_keyframe_list_string("")
        self.assertEqual(len(cleared), 0)

if __name__ == '__main__':
    unittest.main()

