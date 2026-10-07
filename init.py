"""
AutoRoto Package Initialization
Ensures AutoRoto modules are readily importable across Nuke sessions.
"""

import os
import sys

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)
