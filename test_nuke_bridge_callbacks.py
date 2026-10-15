"""Exercise production callbacks with a minimal Nuke API stand-in."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class Knob:
    def __init__(self, name, value=None, kind="String_Knob"):
        self.name, self.v, self.kind = name, value, kind

    def value(self):
        return self.v

    def setValue(self, value):
        self.v = value

    def Class(self):
        return self.kind

    def __getattr__(self, name):
        return lambda *args: None


class Node(dict):
    def knob(self, name):
        return self.get(name)

    def knobs(self):
        return self

    def addKnob(self, knob):
        self[knob.name] = knob

    def removeKnob(self, knob):
        del self[knob.name]

    def name(self):
        return "Roto1"


class TestBridgeCallbacks(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse(Path(__file__).with_name("nuke_bridge.py").read_text(encoding="utf-8"))
        self.nuke = SimpleNamespace(frame=lambda: 25, root=lambda: {
            "first_frame": Knob("first_frame", 1), "last_frame": Knob("last_frame", 100)
        }, message=Mock(), STARTLINE=1)
        for kind in ("Tab_Knob", "Text_Knob", "PyScript_Knob", "String_Knob",
                     "Int_Knob", "Enumeration_Knob", "Boolean_Knob"):
            setattr(self.nuke, kind, lambda name, *args, kind=kind: Knob(name, kind=kind))
        self.ns = dict(nuke=self.nuke, re=re, List=list, hide_intermediate_native_tabs=Mock(),
                       make_autoroto_first_tab=Mock())
        names = {"parse_keyframe_list_string", "setup_autoroto_knobs", "on_node_track_range",
                 "on_node_track_multi_ref", "_run_tracking_for_range"}
        for node in self.tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "nuke_bridge.py", "exec"), self.ns)

    def old_node(self):
        return Node(AutoRoto=Knob("AutoRoto"), knobChanged=Knob("knobChanged"),
                    ar_ref_frame=Knob("ar_ref_frame", 42.0, "Int_Knob"),
                    ar_start_frame=Knob("ar_start_frame", 10),
                    ar_end_frame=Knob("ar_end_frame", 90))

    def test_upgrade_preserves_legacy_reference_list(self):
        node = self.old_node()
        node["ar_multiref_list"] = Knob("ar_multiref_list", "10, 45, 90")
        self.ns["setup_autoroto_knobs"](node)
        self.assertEqual(node["ar_ref_frame"].value(), "10, 45, 90")
        self.assertNotIn("ar_multiref_list", node)
        self.assertEqual(node["ar_start_frame"].value(), 10)
        self.assertEqual(node["ar_end_frame"].value(), 90)
        self.ns["setup_autoroto_knobs"](node)
        self.assertEqual(node["ar_ref_frame"].value(), "10, 45, 90")

    def test_upgrade_normalizes_integer_knob_value(self):
        node = self.old_node()
        self.ns["setup_autoroto_knobs"](node)
        self.assertEqual(self.ns["parse_keyframe_list_string"](node["ar_ref_frame"].value()), [42])

    def test_outside_reference_rejected_before_source_access(self):
        for ref in (1, 201):
            with self.subTest(ref=ref):
                self.ns["_run_tracking_for_range"](Node(), 100, 200, ref)
        self.assertEqual(self.nuke.message.call_count, 2)

    def test_compatibility_handler_routes_unified_references(self):
        definitions = [n for n in self.tree.body if isinstance(n, ast.FunctionDef)
                       and n.name == "on_node_track_multi_ref"]
        self.assertEqual(len(definitions), 1)
        runner = self.ns["_run_multi_reference_tracking"] = Mock()
        node = Node(ar_ref_frame=Knob("ar_ref_frame", "10, 45, 90"),
                    ar_start_frame=Knob("ar_start_frame", 1), ar_end_frame=Knob("ar_end_frame", 100))
        self.ns["on_node_track_multi_ref"](node)
        runner.assert_called_once_with(node, [10, 45, 90], 1, 100)


if __name__ == "__main__":
    unittest.main()
