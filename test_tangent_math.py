"""
Unit Test for AutoRoto Tangent Math and Relative Coordinates
"""

import unittest
import math

class TestTangentMath(unittest.TestCase):

    def get_relative_offset(self, tx, ty, cx, cy):
        dist_from_origin = (tx**2 + ty**2) ** 0.5
        dist_from_center = ((tx - cx)**2 + (ty - cy)**2) ** 0.5
        if dist_from_center < dist_from_origin and (cx**2 + cy**2) ** 0.5 > 50.0:
            return (tx - cx, ty - cy)
        else:
            return (tx, ty)

    def test_cusp_tangent(self):
        # Cusp point (no handle): tangent is (0, 0)
        cx, cy = 600.0, 450.0
        lt_x, lt_y = 0.0, 0.0
        rel_x, rel_y = self.get_relative_offset(lt_x, lt_y, cx, cy)
        self.assertEqual(rel_x, 0.0)
        self.assertEqual(rel_y, 0.0)

    def test_already_relative_tangent(self):
        # Tangent is relative handle (-25.0, 10.0)
        cx, cy = 600.0, 450.0
        lt_x, lt_y = -25.0, 10.0
        rel_x, rel_y = self.get_relative_offset(lt_x, lt_y, cx, cy)
        self.assertEqual(rel_x, -25.0)
        self.assertEqual(rel_y, 10.0)

    def test_absolute_tangent_fallback(self):
        # Hypothetical absolute tangent (575.0, 460.0) with center (600, 450)
        cx, cy = 600.0, 450.0
        lt_x, lt_y = 575.0, 460.0
        rel_x, rel_y = self.get_relative_offset(lt_x, lt_y, cx, cy)
        self.assertEqual(rel_x, -25.0)
        self.assertEqual(rel_y, 10.0)

    def test_rigid_translation_keeps_tangents_exact(self):
        # Shape translates rigidly by (100, 50)
        # Tangents should NOT grow by (100, 50)!
        lt_dx, lt_dy = -20.0, 15.0
        # Reference chord
        p_prev_ref = (100.0, 100.0)
        p_next_ref = (200.0, 100.0)
        v0_x = p_next_ref[0] - p_prev_ref[0]
        v0_y = p_next_ref[1] - p_prev_ref[1]
        L0 = (v0_x**2 + v0_y**2) ** 0.5

        # Frame f translated chord
        p_prev_f = (200.0, 150.0)
        p_next_f = (300.0, 150.0)
        vf_x = p_next_f[0] - p_prev_f[0]
        vf_y = p_next_f[1] - p_prev_f[1]
        Lf = (vf_x**2 + vf_y**2) ** 0.5

        rot_cos = (v0_x * vf_x + v0_y * vf_y) / (L0 * Lf)
        rot_sin = (v0_x * vf_y - v0_y * vf_x) / (L0 * Lf)
        scale = Lf / L0

        lt_cur_x = scale * (rot_cos * lt_dx - rot_sin * lt_dy)
        lt_cur_y = scale * (rot_sin * lt_dx + rot_cos * lt_dy)

        # Under rigid translation, rot_cos=1, rot_sin=0, scale=1
        self.assertAlmostEqual(lt_cur_x, -20.0, places=4)
        self.assertAlmostEqual(lt_cur_y, 15.0, places=4)

    def test_rotation_rotates_tangents_correctly(self):
        # Tangent initially pointing right (20.0, 0.0)
        lt_dx, lt_dy = 20.0, 0.0
        # Reference chord horizontal (100px along X)
        v0_x, v0_y = 100.0, 0.0
        L0 = 100.0

        # Frame f chord rotated 90 degrees CCW (100px along Y)
        vf_x, vf_y = 0.0, 100.0
        Lf = 100.0

        rot_cos = (v0_x * vf_x + v0_y * vf_y) / (L0 * Lf) # 0
        rot_sin = (v0_x * vf_y - v0_y * vf_x) / (L0 * Lf) # 1
        scale = 1.0

        lt_cur_x = scale * (rot_cos * lt_dx - rot_sin * lt_dy) # 0 - 0 = 0
        lt_cur_y = scale * (rot_sin * lt_dx + rot_cos * lt_dy) # 20*1 + 0 = 20

        # Tangent should rotate 90 deg CCW to (0.0, 20.0)
        self.assertAlmostEqual(lt_cur_x, 0.0, places=4)
        self.assertAlmostEqual(lt_cur_y, 20.0, places=4)

    def test_cusp_stays_zero_under_any_rotation(self):
        lt_dx, lt_dy = 0.0, 0.0
        rot_cos, rot_sin = 0.7071, 0.7071
        scale = 1.2
        lt_cur_x = scale * (rot_cos * lt_dx - rot_sin * lt_dy)
        lt_cur_y = scale * (rot_sin * lt_dx + rot_cos * lt_dy)
        self.assertEqual(lt_cur_x, 0.0)
        self.assertEqual(lt_cur_y, 0.0)

if __name__ == '__main__':
    unittest.main()
