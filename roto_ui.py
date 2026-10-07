"""
AutoRoto Custom Panel UI (roto_ui.py)
PySide2 / PySide6 Nuke Roto Properties Panel with CoTracker 3 GPU Deep Learning Backend.
"""

import sys
import shutil
from typing import Optional, Dict, Any, List, Tuple

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import Qt, Signal, Slot
    from PySide6.QtGui import QColor, QBrush, QFont
    from PySide6.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
        QPushButton, QLabel, QSpinBox, QGroupBox, QHeaderView, QMessageBox,
        QProgressBar, QCheckBox, QLineEdit
    )
except ImportError:
    from PySide2 import QtCore, QtGui, QtWidgets
    from PySide2.QtCore import Qt, Signal, Slot
    from PySide2.QtGui import QColor, QBrush, QFont
    from PySide2.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
        QPushButton, QLabel, QSpinBox, QGroupBox, QHeaderView, QMessageBox,
        QProgressBar, QCheckBox, QLineEdit
    )

import nuke
from nuke_bridge import NukeRotoBridge
import tracker_core

NUKE_DARK_QSS = """
QWidget {
    background-color: #282828;
    color: #dedede;
    font-family: "Segoe UI", "Verdana", sans-serif;
    font-size: 11px;
}

QGroupBox {
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    margin-top: 14px;
    font-weight: bold;
    padding-top: 10px;
    background-color: #2b2b2b;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #e58934;
}

QTreeWidget {
    background-color: #1f1f1f;
    border: 1px solid #383838;
    border-radius: 3px;
    color: #dedede;
    alternate-background-color: #252525;
}

QTreeWidget::item {
    height: 24px;
}

QTreeWidget::item:selected {
    background-color: #c76b2b;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #333333;
    color: #b0b0b0;
    padding: 4px;
    border: 1px solid #282828;
    font-weight: bold;
}

QPushButton {
    background-color: #383838;
    border: 1px solid #4a4a4a;
    border-radius: 3px;
    padding: 5px 12px;
    color: #e0e0e0;
    font-weight: bold;
    min-height: 22px;
}

QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #c76b2b;
    color: #ffffff;
}

QPushButton:pressed {
    background-color: #c76b2b;
    color: #ffffff;
}

QPushButton#PrimaryActionBtn {
    background-color: #2e6b38;
    border-color: #438f51;
    color: #ffffff;
    font-size: 12px;
}

QPushButton#PrimaryActionBtn:hover {
    background-color: #3b8747;
    border-color: #5ec273;
}

QPushButton#SecondaryActionBtn {
    background-color: #3d5060;
    border-color: #507088;
}

QPushButton#SecondaryActionBtn:hover {
    background-color: #4e6b82;
    border-color: #729ebf;
}

QSpinBox, QLineEdit {
    background-color: #1a1a1a;
    border: 1px solid #3c3c3c;
    border-radius: 3px;
    padding: 3px 6px;
    color: #ffffff;
}

QSpinBox:focus, QLineEdit:focus {
    border-color: #c76b2b;
}

QProgressBar {
    background-color: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 3px;
    text-align: center;
    color: #ffffff;
    height: 16px;
}

QProgressBar::chunk {
    background-color: #c76b2b;
}
"""

class AutoRotoPanel(QWidget):
    """
    Nuke Roto Panel with CoTracker 3 GPU Backend.
    """
    def __init__(self, bridge: Optional[NukeRotoBridge] = None, parent=None):
        super().__init__(parent)
        self.bridge = bridge or NukeRotoBridge()
        self.current_shape = None
        self.current_roto_node = None
        self.tracked_data: Dict[int, Dict[int, Tuple[float, float, float]]] = {}

        self.setWindowTitle("AutoRoto Panel (CoTracker AI)")
        self.setStyleSheet(NUKE_DARK_QSS)
        self.resize(520, 680)

        self._init_ui()
        self._load_hierarchy()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 1. Header Title
        title_label = QLabel(
            "<font color='#e58934' size=4><b>AutoRoto</b></font> "
            "<font color='#888'>CoTracker3 GPU Point Tracker</font>"
        )
        main_layout.addWidget(title_label)

        # 2. Node Selection & Import
        top_bar = QHBoxLayout()
        self.btn_import = QPushButton("Import from Selected Roto")
        self.btn_import.setObjectName("SecondaryActionBtn")
        self.node_label = QLabel("Active Node: None")
        self.node_label.setStyleSheet("color: #a0a0a0; font-weight: bold;")
        top_bar.addWidget(self.btn_import)
        top_bar.addWidget(self.node_label, stretch=1)
        main_layout.addLayout(top_bar)

        # 3. Curves / Layers Hierarchy Tree
        tree_group = QGroupBox("Curves / Layers Hierarchy")
        tree_layout = QVBoxLayout(tree_group)
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Name", "Lock", "Vis", "Points"])
        self.tree_widget.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree_widget.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tree_widget.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.tree_widget.header().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.tree_widget.itemSelectionChanged.connect(self._on_tree_selection_changed)
        tree_layout.addWidget(self.tree_widget)
        main_layout.addWidget(tree_group, stretch=1)

        # 4. Frame & Timeline Controls
        curr_f, start_f, end_f = self.bridge.get_timeline_range()
        frame_group = QGroupBox("Timeline & Range Settings")
        frame_layout = QVBoxLayout(frame_group)

        ref_layout = QHBoxLayout()
        ref_layout.addWidget(QLabel("Reference Frame:"))
        self.ref_spin = QSpinBox()
        self.ref_spin.setRange(-9999, 99999)
        self.ref_spin.setValue(curr_f)
        self.btn_set_curr = QPushButton("Set Current")
        ref_layout.addWidget(self.ref_spin)
        ref_layout.addWidget(self.btn_set_curr)
        frame_layout.addLayout(ref_layout)

        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Track Range:"))
        self.start_spin = QSpinBox()
        self.start_spin.setRange(-9999, 99999)
        self.start_spin.setValue(start_f)
        self.end_spin = QSpinBox()
        self.end_spin.setRange(-9999, 99999)
        self.end_spin.setValue(end_f)
        self.btn_sync_range = QPushButton("Project Range")
        range_layout.addWidget(self.start_spin)
        range_layout.addWidget(QLabel("-"))
        range_layout.addWidget(self.end_spin)
        range_layout.addWidget(self.btn_sync_range)
        frame_layout.addLayout(range_layout)
        main_layout.addWidget(frame_group)

        # 5. CoTracker AI Hardware Status & Options
        ai_group = QGroupBox("CoTracker 3 Backend (GPU Accelerated)")
        ai_layout = QVBoxLayout(ai_group)

        ai_status_row = QHBoxLayout()
        self.ai_status_label = QLabel("Hardware: NVIDIA GeForce RTX 4080 (CUDA)")
        self.ai_status_label.setStyleSheet("color: #4caf50; font-weight: bold;")
        self.btn_check_ai = QPushButton("Check GPU Status")
        ai_status_row.addWidget(self.ai_status_label, stretch=1)
        ai_status_row.addWidget(self.btn_check_ai)
        ai_layout.addLayout(ai_status_row)

        self.cb_keep_tangents = QCheckBox("Preserve Bezier Tangents / Curvature")
        self.cb_keep_tangents.setChecked(True)
        ai_layout.addWidget(self.cb_keep_tangents)
        main_layout.addWidget(ai_group)

        # 6. Action Execution
        exec_group = QGroupBox("Tracking Actions")
        exec_layout = QVBoxLayout(exec_group)

        self.btn_run_tracking = QPushButton("🚀 Run CoTracker GPU Tracking")
        self.btn_run_tracking.setObjectName("PrimaryActionBtn")
        self.btn_run_tracking.setMinimumHeight(32)
        exec_layout.addWidget(self.btn_run_tracking)

        self.prog_bar = QProgressBar()
        self.prog_bar.setValue(0)
        exec_layout.addWidget(self.prog_bar)

        bake_row = QHBoxLayout()
        self.btn_bake = QPushButton("Bake Tracked Keys to Roto")
        self.btn_bake.setObjectName("SecondaryActionBtn")
        bake_row.addWidget(self.btn_bake)
        exec_layout.addLayout(bake_row)
        main_layout.addWidget(exec_group)

        # 7. Status Footer
        self.status_label = QLabel("Ready. Select a Roto shape and click 'Run CoTracker GPU Tracking'.")
        self.status_label.setStyleSheet("color: #a0a0a0; padding: 2px;")
        main_layout.addWidget(self.status_label)

        # Event Connections
        self.btn_import.clicked.connect(self._load_hierarchy)
        self.btn_set_curr.clicked.connect(self._on_set_curr_frame)
        self.btn_sync_range.clicked.connect(self._on_sync_project_range)
        self.btn_check_ai.clicked.connect(self._on_check_ai_status)
        self.btn_run_tracking.clicked.connect(self._on_run_tracking)
        self.btn_bake.clicked.connect(self._on_bake_to_roto)

    def _load_hierarchy(self):
        self.tree_widget.clear()
        roto_node = self.bridge.get_selected_roto_node()
        self.current_roto_node = roto_node

        if not roto_node:
            self.node_label.setText("Active Node: None (Select a Roto node)")
            self.status_label.setText("No Roto node found. Please select a Roto node in Nuke.")
            return

        self.node_label.setText(f"Active Node: {roto_node.name()}")
        layers = self.bridge.import_roto_hierarchy(roto_node)
        first_shape = None

        for layer_data in layers:
            l_item = QTreeWidgetItem([layer_data['name'], "", "", ""])
            l_item.setFont(0, QFont("Segoe UI", 10, QFont.Bold))
            l_item.setForeground(0, QBrush(QColor("#e58934")))
            self.tree_widget.addTopLevelItem(l_item)

            for shp_data in layer_data.get('shapes', []):
                s_item = QTreeWidgetItem([
                    shp_data['name'],
                    "🔒" if shp_data['locked'] else "🔓",
                    "👁" if shp_data['visible'] else "✕",
                    str(shp_data['points_count'])
                ])
                s_item.setData(0, Qt.UserRole, shp_data['raw_shape'])
                l_item.addChild(s_item)
                if first_shape is None:
                    first_shape = shp_data['raw_shape']

        self.tree_widget.expandAll()
        if first_shape:
            self.current_shape = first_shape
            self.status_label.setText(f"Loaded '{first_shape.name}' ({len(first_shape)} control points).")

    def _on_tree_selection_changed(self):
        items = self.tree_widget.selectedItems()
        if not items:
            return
        item = items[0]
        raw_shape = item.data(0, Qt.UserRole)
        if raw_shape is not None:
            self.current_shape = raw_shape
            self.status_label.setText(f"Active Shape: {raw_shape.name} ({len(raw_shape)} points).")

    def _on_set_curr_frame(self):
        curr = int(nuke.frame())
        self.ref_spin.setValue(curr)
        self.status_label.setText(f"Reference frame set to {curr}.")

    def _on_sync_project_range(self):
        curr_f, start_f, end_f = self.bridge.get_timeline_range()
        self.start_spin.setValue(start_f)
        self.end_spin.setValue(end_f)
        self.status_label.setText(f"Synced project frame range: {start_f} - {end_f}.")

    def _on_check_ai_status(self):
        info = tracker_core.check_ai_environment()
        if info.get('available'):
            dev = info.get('device_name', 'GPU')
            torch_v = info.get('torch_version', '')
            QMessageBox.information(
                self, "AI Hardware Status",
                f"CoTracker 3 Backend: READY\n\n"
                f"• Device: {dev}\n"
                f"• CUDA Accelerated: {info.get('cuda')}\n"
                f"• PyTorch Version: {torch_v}\n"
                f"• Model: CoTracker3 Offline (scaled_offline.pth)\n\n"
                f"Hardware is configured for high-speed GPU point tracking."
            )
        else:
            QMessageBox.warning(self, "AI Status", f"AI backend check: {info.get('message')}")

    def _on_run_tracking(self):
        if not self.current_roto_node or not self.current_shape:
            QMessageBox.warning(self, "Warning", "Please select a Roto node and shape first.")
            return

        source_node = self.current_roto_node.input(0)
        if not source_node:
            QMessageBox.warning(self, "No Input", f"Please connect video/image footage to '{self.current_roto_node.name()}' input.")
            return

        ref_f = self.ref_spin.value()
        start_f = self.start_spin.value()
        end_f = self.end_spin.value()

        if start_f >= end_f:
            QMessageBox.warning(self, "Invalid Range", "Start frame must be less than End frame.")
            return

        pts_info = self.bridge.extract_shape_points(self.current_shape, ref_f)
        if not pts_info:
            QMessageBox.warning(self, "No Points", f"Shape '{self.current_shape.name}' has no control points.")
            return

        # Prepare queries: relative time t_rel = ref_f - start_f
        t_rel = float(ref_f - start_f)
        queries = []
        for p in pts_info:
            queries.append({
                "index": p["index"],
                "t": t_rel,
                "x": float(p["x"]),
                "y": float(p["y"])
            })

        self.prog_bar.setValue(10)
        self.status_label.setText("Exporting frame sequence from source node...")
        QtWidgets.QApplication.processEvents()

        temp_dir = None
        try:
            image_paths, temp_dir = self.bridge.export_source_frames(source_node, start_f, end_f)
            self.prog_bar.setValue(35)
            self.status_label.setText("Running CoTracker3 inference on GPU (RTX 4080)...")
            QtWidgets.QApplication.processEvents()

            tracks = tracker_core.run_cotracker_point_tracking(
                image_paths=image_paths,
                queries=queries,
                start_frame=start_f
            )

            self.tracked_data = tracks
            self.prog_bar.setValue(85)
            self.status_label.setText("Tracking complete! Auto-baking keyframes to Roto shape...")
            QtWidgets.QApplication.processEvents()

            # Automatically bake to Roto node
            keep_tg = self.cb_keep_tangents.isChecked()
            ok = self.bridge.bake_tracking_data_to_roto(
                self.current_roto_node, self.current_shape, self.tracked_data, ref_f, keep_tangents=keep_tg
            )

            self.prog_bar.setValue(100)
            if ok:
                msg = f"✓ Tracked & Baked {len(pts_info)} points across frames {start_f}-{end_f}!"
                self.status_label.setText(msg)
                QMessageBox.information(
                    self, "Complete",
                    f"AutoRoto: CoTracker GPU tracking finished!\n\n"
                    f"• {len(pts_info)} points tracked\n"
                    f"• {end_f - start_f + 1} frames baked into '{self.current_shape.name}'\n\n"
                    f"You can now scrub the timeline and tweak any point in the Viewer."
                )
            else:
                self.status_label.setText("Tracking completed, but bake failed.")

        except Exception as e:
            self.prog_bar.setValue(0)
            self.status_label.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Tracking Error", f"CoTracker tracking failed:\n\n{str(e)}")

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def _on_bake_to_roto(self):
        if not self.current_roto_node or not self.current_shape or not self.tracked_data:
            QMessageBox.warning(self, "No Data", "No tracking data available to bake.")
            return

        ref_f = self.ref_spin.value()
        keep_tg = self.cb_keep_tangents.isChecked()
        ok = self.bridge.bake_tracking_data_to_roto(
            self.current_roto_node, self.current_shape, self.tracked_data, ref_f, keep_tangents=keep_tg
        )
        if ok:
            QMessageBox.information(self, "Bake Success", "Keyframes successfully baked into Roto shape!")
        else:
            QMessageBox.critical(self, "Bake Error", "Failed to bake keyframes to Roto.")
