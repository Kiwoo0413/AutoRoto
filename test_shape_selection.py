"""
Tests for selected-shape targeting: each selected Roto shape is tracked independently.
Executes production functions from nuke_bridge.py against a minimal rotopaint stand-in.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class Layer(list):
    def __init__(self, name, items=()):
        super().__init__(items)
        self.name = name


class Shape(list):
    def __init__(self, name, num_points=3):
        super().__init__(range(num_points))
        self.name = name


class Curves:
    def __init__(self, root, selected=None, raise_on_select=False):
        self.rootLayer = root
        self._selected = selected or []
        self._raise = raise_on_select

    def getSelected(self):
        if self._raise:
            raise RuntimeError("no selection API")
        return self._selected


class Knob:
    def __init__(self, value=""):
        self.v = value

    def value(self):
        return self.v

    def setValue(self, value):
        self.v = value


class RotoNode(dict):
    def knob(self, name):
        return self.get(name)

    def name(self):
        return "AutoRoto1"


FUNCS = {
    "_is_roto_shape", "get_all_shapes", "get_selected_shapes", "get_target_shapes",
    "get_target_shape", "_shape_names", "_build_combined_queries", "_split_tracks_by_shape",
    "_group_shapes_by_ref_frames", "parse_keyframe_list_string", "on_node_track_range",
}


class TestSelectedShapeTargeting(unittest.TestCase):
    def setUp(self):
        import re
        tree = ast.parse(Path(__file__).with_name("nuke_bridge.py").read_text(encoding="utf-8"))
        self.nuke = SimpleNamespace(
            frame=lambda: 7,
            root=lambda: {"first_frame": Knob(1), "last_frame": Knob(100)},
            message=Mock(),
        )
        self.ns = dict(
            rp=SimpleNamespace(Layer=Layer, Shape=Shape),
            nuke=self.nuke, re=re, List=list,
            detect_shape_keyframes=lambda s: [],
            _run_tracking_for_range=Mock(return_value="ok"),
            _run_multi_reference_tracking=Mock(return_value="ok"),
        )
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in FUNCS:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "nuke_bridge.py", "exec"), self.ns)

        self.b1, self.b2, self.b3 = Shape("Bezier1"), Shape("Bezier2", 4), Shape("Bezier3", 2)
        self.layer = Layer("Layer1", [self.b3])
        self.root = Layer("Root", [self.b1, self.b2, self.layer])

    def node(self, selected=None, raise_on_select=False, ref=""):
        return RotoNode(
            curves=Curves(self.root, selected, raise_on_select),
            ar_ref_frame=Knob(ref), ar_start_frame=Knob(1), ar_end_frame=Knob(100),
            ar_status=Knob(""),
        )

    # --- shape resolution ---------------------------------------------------

    def test_selected_single_shape_is_targeted_not_top(self):
        shapes = self.ns["get_target_shapes"](self.node([self.b2]))
        self.assertEqual([s.name for s in shapes], ["Bezier2"])

    def test_multiple_selection_dedup_and_layer_expansion(self):
        shapes = self.ns["get_target_shapes"](self.node([self.b2, self.layer, self.b2]))
        self.assertEqual([s.name for s in shapes], ["Bezier2", "Bezier3"])

    def test_no_selection_falls_back_to_top_shape(self):
        shapes = self.ns["get_target_shapes"](self.node([]))
        self.assertEqual([s.name for s in shapes], ["Bezier1"])

    def test_selection_api_failure_falls_back(self):
        self.assertEqual(self.ns["get_target_shape"](self.node(raise_on_select=True)).name, "Bezier1")

    def test_all_shapes_recurses_layers(self):
        names = [s.name for s in self.ns["get_all_shapes"](self.node())]
        self.assertEqual(names, ["Bezier1", "Bezier2", "Bezier3"])

    # --- combined queries / split ------------------------------------------

    def test_combined_queries_roundtrip_per_shape(self):
        bridge = SimpleNamespace(extract_shape_points=lambda shape, frame, roto_node=None: [
            {"index": i, "x": float(i), "y": float(len(shape.name) + i)} for i in range(len(shape))
        ])
        shapes = [self.b2, self.b3]
        queries, index_map = self.ns["_build_combined_queries"](bridge, None, shapes, 10, 4.0)
        self.assertEqual(len(queries), 4 + 2)
        self.assertEqual(sorted(q["index"] for q in queries), list(range(6)))
        self.assertTrue(all(q["t"] == 4.0 for q in queries))
        self.assertEqual(index_map[4], (1, 0))

        tracks = {g: {10: (float(g), 0.0, 1.0)} for g in index_map}
        per_shape = self.ns["_split_tracks_by_shape"](tracks, index_map, 2)
        self.assertEqual(sorted(per_shape[0]), [0, 1, 2, 3])
        self.assertEqual(sorted(per_shape[1]), [0, 1])
        self.assertEqual(per_shape[1][1][10][0], 5.0)

    # --- per-shape keyframe grouping ---------------------------------------

    def test_grouping_by_per_shape_keyframes(self):
        keys = {"Bezier1": [1, 50], "Bezier2": [1, 50], "Bezier3": []}
        groups = self.ns["_group_shapes_by_ref_frames"](
            [self.b1, self.b2, self.b3], 7, detector=lambda s: keys[s.name])
        self.assertEqual([(g, [s.name for s in sh]) for g, sh in groups],
                         [([1, 50], ["Bezier1", "Bezier2"]), ([7], ["Bezier3"])])

    def test_blank_ref_tracks_each_group_independently(self):
        keys = {"Bezier2": [1, 50], "Bezier3": [20]}
        self.ns["detect_shape_keyframes"] = lambda s: keys[s.name]
        node = self.node([self.b2, self.b3])
        self.ns["on_node_track_range"](node)

        multi = self.ns["_run_multi_reference_tracking"]
        single = self.ns["_run_tracking_for_range"]
        multi.assert_called_once()
        self.assertEqual(multi.call_args.args[1], [1, 50])
        self.assertEqual([s.name for s in multi.call_args.kwargs["shapes"]], ["Bezier2"])
        self.assertFalse(multi.call_args.kwargs["notify"])
        single.assert_called_once()
        self.assertEqual(single.call_args.args[3], 20)
        self.assertEqual([s.name for s in single.call_args.kwargs["shapes"]], ["Bezier3"])
        # Different per-shape keys must not be collapsed into the shared Ref Frame field
        self.assertEqual(node["ar_ref_frame"].value(), "")
        self.nuke.message.assert_called_once()

    def test_blank_ref_single_group_writes_back_keys(self):
        self.ns["detect_shape_keyframes"] = lambda s: [5, 30]
        node = self.node([self.b1, self.b2])
        self.ns["on_node_track_range"](node)
        self.assertEqual(node["ar_ref_frame"].value(), "5, 30")
        self.assertTrue(self.ns["_run_multi_reference_tracking"].call_args.kwargs["notify"])

    def test_explicit_ref_routes_to_runner_without_shape_override(self):
        node = self.node([self.b2], ref="10")
        self.ns["on_node_track_range"](node)
        self.ns["_run_tracking_for_range"].assert_called_once_with(node, 1, 100, 10)


if __name__ == "__main__":
    unittest.main()

