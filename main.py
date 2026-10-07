"""
AutoRoto Main Entry Point (main.py)
Registers and launches the AutoRoto PySide Panel inside Nuke.
"""

import sys
import os
import nuke
import nukescripts

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

from roto_ui import AutoRotoPanel
from nuke_bridge import NukeRotoBridge

_active_panel = None

def launch_panel(dockable: bool = False):
    """
    Launches AutoRoto Panel inside Nuke.
    """
    global _active_panel

    if dockable:
        pane_id = 'uk.co.autoroto.AutoRotoPanel'
        return nukescripts.panels.registerWidgetAsPanel(
            'roto_ui.AutoRotoPanel',
            'AutoRoto Panel',
            pane_id
        )
    else:
        bridge = NukeRotoBridge()
        _active_panel = AutoRotoPanel(bridge=bridge)
        _active_panel.show()
        return _active_panel

if __name__ == "__main__":
    launch_panel(dockable=False)
