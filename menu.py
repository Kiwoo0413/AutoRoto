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
