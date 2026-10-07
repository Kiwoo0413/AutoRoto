"""
AutoRoto Menu Registration (menu.py)
Integrates AutoRoto Roto Node (Tracker Buttons) into Nuke.
"""

import os
import sys
import nuke

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

import nuke_bridge

# 1. Add to Nodes Toolbar
toolbar = nuke.menu('Nodes')
autoroto_menu = toolbar.addMenu('AutoRoto', icon='Roto.png')

# Primary: Create AutoRoto Node (Roto Node with Tracker Buttons)
autoroto_menu.addCommand(
    'Create AutoRoto Node',
    'import nuke_bridge; nuke_bridge.create_autoroto_node()',
    icon='Roto.png',
    shortcut='Ctrl+Alt+R'
)

# Convert Selected Roto
autoroto_menu.addCommand(
    'Convert Selected Roto to AutoRoto',
    'import nuke_bridge; nuke_bridge.add_autoroto_to_selected()',
    icon='Roto.png'
)

# 2. Add to Draw Menu
draw_menu = toolbar.findItem('Draw')
if draw_menu:
    draw_menu.addCommand(
        'AutoRoto (Tracker Buttons)',
        'import nuke_bridge; nuke_bridge.create_autoroto_node()',
        icon='Roto.png'
    )

# 3. Add to Main Menu Bar
main_menubar = nuke.menu('Nuke')
top_menu = main_menubar.addMenu('&AutoRoto')
top_menu.addCommand(
    'Create AutoRoto Node (Tracker Buttons)',
    'import nuke_bridge; nuke_bridge.create_autoroto_node()',
    shortcut='Ctrl+Alt+R'
)
top_menu.addCommand(
    'Convert Selected Roto to AutoRoto',
    'import nuke_bridge; nuke_bridge.add_autoroto_to_selected()'
)
top_menu.addSeparator()
top_menu.addCommand(
    'Check CoTracker GPU Status',
    """import tracker_core; info = tracker_core.check_ai_environment(); nuke.message(f"CoTracker AI Status:\\n\\n• Device: {info.get('device_name')}\\n• CUDA: {info.get('cuda')}\\n• PyTorch: {info.get('torch_version')}\\n• Status: {info.get('message')}")"""
)
top_menu.addSeparator()
top_menu.addCommand(
    'About AutoRoto',
    """import nuke; nuke.message("AutoRoto v2.1 (Tracker Buttons on Roto)\\n\\nNative Nuke Roto Node equipped with Tracker-style VCR buttons & CoTracker3 GPU backend.\\n\\n• Draw shapes natively in Nuke Viewer\\n• VCR Tracker buttons: |◀, ◀, ▶, ▶|, Track Range\\n• Zero-drift bidirectional tracking on RTX 4080")"""
)
