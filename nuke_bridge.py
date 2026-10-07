"""
AutoRoto Nuke Bridge (nuke_bridge.py)
Direct integration with Nuke Roto / RotoPaint API (nuke.roto / _rotopaint).
Provides:
  1. Fix for RotoKnob rootLayer access across all Nuke versions.
  2. Roto node cloning with Tracker-style VCR buttons & properties.
  3. CoTracker 3 GPU point tracking callbacks directly from the Roto node.
"""

import os
import shutil
import tempfile
from typing import List, Dict, Any, Tuple, Optional

import nuke
import nuke.rotopaint as rp
import tracker_core

class NukeRotoBridge:
    """
    Bridge connecting Roto nodes directly to active Nuke session & CoTracker backend.
    """

    def __init__(self):
        pass

    def get_timeline_range(self) -> Tuple[int, int, int]:
        """
        Returns (current_frame, first_frame, last_frame) from Nuke root.
        """
        curr = int(nuke.frame())
        start = int(nuke.root()['first_frame'].value())
        end = int(nuke.root()['last_frame'].value())
        return curr, start, end

    def set_current_frame(self, frame: int):
        nuke.frame(int(frame))

    def get_selected_roto_node(self):
        """
        Retrieves the currently selected Roto/RotoPaint node, or first one in DAG.
        """
        try:
            sel = nuke.selectedNode()
            if sel and sel.Class() in ('Roto', 'RotoPaint'):
                return sel
        except Exception:
            pass

        for node in nuke.allNodes():
            if node.Class() in ('Roto', 'RotoPaint'):
                return node
        return None

    def import_roto_hierarchy(self, roto_node=None) -> List[Dict[str, Any]]:
        """
        Collects Layers and Shapes hierarchy from the target Roto node.
        Uses rootLayer (robust fallback to root) to support all Nuke versions.
        """
        if roto_node is None:
            roto_node = self.get_selected_roto_node()
        if not roto_node:
            return []

        curves = roto_node['curves']
        # Fix for '_rotopaint.RotoKnob' object has no attribute 'root':
        # In Nuke Python API, RotoKnob has .rootLayer
        root = getattr(curves, 'rootLayer', getattr(curves, 'root', None))
        if root is None:
            return []

        curr_frame = float(nuke.frame())
        tree = []
        default_layer = {'name': 'RootLayer', 'locked': False, 'visible': True, 'shapes': []}

        def _walk(item, current_layer):
            for elem in item:
                if isinstance(elem, rp.Layer):
                    layer_data = {
                        'name': elem.name,
                        'locked': elem.getAttributes().getValue(curr_frame, 'locked') if hasattr(elem, 'getAttributes') else False,
                        'visible': elem.getAttributes().getValue(curr_frame, 'visible') if hasattr(elem, 'getAttributes') else True,
                        'shapes': []
                    }
                    tree.append(layer_data)
                    _walk(elem, layer_data)
                elif isinstance(elem, rp.Shape) or hasattr(elem, 'getControlPoint') or type(elem).__name__ == 'Shape':
                    current_layer['shapes'].append({
                        'name': elem.name,
                        'locked': False,
                        'visible': True,
                        'points_count': len(elem),
                        'raw_shape': elem
                    })

        _walk(root, default_layer)
        if default_layer['shapes']:
            tree.insert(0, default_layer)

        return tree

    @staticmethod
    def _get_relative_offset(tan_pos, center_pos) -> Tuple[float, float]:
        """
        Guarantees returning (dx, dy) relative to center_pos (cx, cy).
        In Nuke _rotopaint, tan_pos is already relative to center.
        If tan_pos were somehow absolute, converts it to relative.
        """
        tx, ty = float(tan_pos.x), float(tan_pos.y)
        cx, cy = float(center_pos.x), float(center_pos.y)

        dist_from_origin = (tx**2 + ty**2) ** 0.5
        dist_from_center = ((tx - cx)**2 + (ty - cy)**2) ** 0.5

        if dist_from_center < dist_from_origin and (cx**2 + cy**2) ** 0.5 > 50.0:
            return (tx - cx, ty - cy)
        else:
            return (tx, ty)

    def extract_shape_points(self, raw_shape, frame: int) -> List[Dict[str, Any]]:
        """
        Extracts control point coordinates and relative tangents at specified frame.
        """
        num_pts = len(raw_shape)
        pts_info = []

        for i in range(num_pts):
            pt = raw_shape[i]
            c_pos = pt.center.getPosition(float(frame))
            cx, cy = float(c_pos.x), float(c_pos.y)

            lt_pos = pt.leftTangent.getPosition(float(frame))
            rt_pos = pt.rightTangent.getPosition(float(frame))
            fc_pos = pt.featherCenter.getPosition(float(frame))

            lt_rel = self._get_relative_offset(lt_pos, c_pos)
            rt_rel = self._get_relative_offset(rt_pos, c_pos)
            fc_rel = self._get_relative_offset(fc_pos, c_pos)

            pts_info.append({
                'index': i,
                'x': cx,
                'y': cy,
                'lt_rel': lt_rel,
                'rt_rel': rt_rel,
                'fc_rel': fc_rel,
                'raw_point': pt
            })

        return pts_info

    def export_source_frames(self, source_node, start_f: int, end_f: int) -> Tuple[List[str], Optional[str]]:
        """
        Gathers or renders frame sequence for the specified frame range.
        Returns: (list_of_image_paths, temp_dir_or_None)
        """
        if source_node.Class() == 'Read':
            try:
                paths = []
                all_exist = True
                for f in range(start_f, end_f + 1):
                    p = nuke.filename(source_node, nuke.REPLACE)
                    if p and os.path.exists(p) and not p.lower().endswith(('.mov', '.mp4', '.mkv')):
                        paths.append(p)
                    else:
                        all_exist = False
                        break
                if all_exist and len(paths) == (end_f - start_f + 1):
                    return paths, None
            except Exception:
                pass

        # Fast temporary render via Write node
        temp_dir = tempfile.mkdtemp(prefix="autoroto_frames_")
        pattern = os.path.join(temp_dir, "frame_%04d.jpg").replace("\\", "/")

        write_node = nuke.nodes.Write(
            inputs=[source_node],
            file=pattern,
            file_type="jpeg",
            _jpeg_quality=0.92
        )

        try:
            nuke.execute(write_node, start_f, end_f, 1)
        finally:
            nuke.delete(write_node)

        paths = []
        for f in range(start_f, end_f + 1):
            p = os.path.join(temp_dir, f"frame_{f:04d}.jpg")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Rendered frame not found: {p}")
            paths.append(p)

        return paths, temp_dir

    def bake_tracking_data_to_roto(
        self,
        roto_node,
        raw_shape,
        tracking_data: Dict[int, Dict[int, Tuple[float, float, float]]],
        ref_frame: int,
        keep_tangents: bool = True
    ) -> bool:
        """
        Bakes CoTracker trajectories into Roto control point animation curves.
        Preserves natural curvature and rotates tangents with local shape deformation.
        Never adds center coordinates to relative tangent handles.
        """
        if not roto_node or not raw_shape:
            return False

        curves = roto_node['curves']
        pts_info = self.extract_shape_points(raw_shape, ref_frame)
        if not pts_info:
            return False

        num_pts = len(pts_info)

        # Check if shape is open or closed
        is_open = False
        if hasattr(raw_shape, 'getFlag'):
            try:
                is_open = bool(raw_shape.getFlag(5))  # eOpenFlag = 5
            except Exception:
                is_open = False

        # Precompute neighbor indices for rotation tracking
        neighbors = {}
        for i in range(num_pts):
            if not is_open and num_pts >= 3:
                prev_i = (i - 1) % num_pts
                next_i = (i + 1) % num_pts
            elif is_open and num_pts >= 2:
                if i == 0:
                    prev_i = 0
                    next_i = 1
                elif i == num_pts - 1:
                    prev_i = num_pts - 2
                    next_i = num_pts - 1
                else:
                    prev_i = i - 1
                    next_i = i + 1
            else:
                prev_i = i
                next_i = i
            neighbors[i] = (prev_i, next_i)

        baked_count = 0

        for pt_data in pts_info:
            i = pt_data['index']
            if i not in tracking_data:
                continue

            raw_pt = pt_data['raw_point']
            lt_dx, lt_dy = pt_data['lt_rel']
            rt_dx, rt_dy = pt_data['rt_rel']
            fc_dx, fc_dy = pt_data['fc_rel']

            is_lt_zero = (abs(lt_dx) < 1e-3 and abs(lt_dy) < 1e-3)
            is_rt_zero = (abs(rt_dx) < 1e-3 and abs(rt_dy) < 1e-3)
            is_fc_zero = (abs(fc_dx) < 1e-3 and abs(fc_dy) < 1e-3)

            c_curve_x = raw_pt.center.getPositionAnimCurve(0)
            c_curve_y = raw_pt.center.getPositionAnimCurve(1)

            lt_curve_x = raw_pt.leftTangent.getPositionAnimCurve(0)
            lt_curve_y = raw_pt.leftTangent.getPositionAnimCurve(1)
            rt_curve_x = raw_pt.rightTangent.getPositionAnimCurve(0)
            rt_curve_y = raw_pt.rightTangent.getPositionAnimCurve(1)
            fc_curve_x = raw_pt.featherCenter.getPositionAnimCurve(0)
            fc_curve_y = raw_pt.featherCenter.getPositionAnimCurve(1)

            prev_i, next_i = neighbors[i]
            p_prev_ref = (pts_info[prev_i]['x'], pts_info[prev_i]['y'])
            p_next_ref = (pts_info[next_i]['x'], pts_info[next_i]['y'])
            v0_x = p_next_ref[0] - p_prev_ref[0]
            v0_y = p_next_ref[1] - p_prev_ref[1]
            L0 = (v0_x**2 + v0_y**2) ** 0.5

            for f, val in tracking_data[i].items():
                tx = float(val[0])
                ty = float(val[1])

                # 1. Center gets absolute image coordinates
                c_curve_x.addKey(f, tx)
                c_curve_y.addKey(f, ty)

                if keep_tangents:
                    # Calculate local rotation and gentle scale from tracked neighbors
                    rot_cos = 1.0
                    rot_sin = 0.0
                    scale = 1.0

                    if prev_i != next_i and L0 > 2.0:
                        has_prev = (prev_i in tracking_data and f in tracking_data[prev_i])
                        has_next = (next_i in tracking_data and f in tracking_data[next_i])
                        if has_prev and has_next:
                            p_prev_f = tracking_data[prev_i][f]
                            p_next_f = tracking_data[next_i][f]
                            vf_x = float(p_next_f[0]) - float(p_prev_f[0])
                            vf_y = float(p_next_f[1]) - float(p_prev_f[1])
                            Lf = (vf_x**2 + vf_y**2) ** 0.5
                            if Lf > 2.0:
                                rot_cos = (v0_x * vf_x + v0_y * vf_y) / (L0 * Lf)
                                rot_sin = (v0_x * vf_y - v0_y * vf_x) / (L0 * Lf)
                                raw_scale = Lf / L0
                                scale = max(0.7, min(raw_scale, 1.4))

                    m_xx = scale * rot_cos
                    m_yx = scale * rot_sin

                    # 2. Left Tangent
                    if lt_curve_x and lt_curve_y:
                        if is_lt_zero:
                            lt_curve_x.addKey(f, 0.0)
                            lt_curve_y.addKey(f, 0.0)
                        else:
                            lt_cur_x = m_xx * lt_dx - m_yx * lt_dy
                            lt_cur_y = m_yx * lt_dx + m_xx * lt_dy
                            lt_curve_x.addKey(f, float(lt_cur_x))
                            lt_curve_y.addKey(f, float(lt_cur_y))

                    # 3. Right Tangent
                    if rt_curve_x and rt_curve_y:
                        if is_rt_zero:
                            rt_curve_x.addKey(f, 0.0)
                            rt_curve_y.addKey(f, 0.0)
                        else:
                            rt_cur_x = m_xx * rt_dx - m_yx * rt_dy
                            rt_cur_y = m_yx * rt_dx + m_xx * rt_dy
                            rt_curve_x.addKey(f, float(rt_cur_x))
                            rt_curve_y.addKey(f, float(rt_cur_y))

                    # 4. Feather Center
                    if fc_curve_x and fc_curve_y:
                        if is_fc_zero:
                            fc_curve_x.addKey(f, 0.0)
                            fc_curve_y.addKey(f, 0.0)
                        else:
                            fc_cur_x = m_xx * fc_dx - m_yx * fc_dy
                            fc_cur_y = m_yx * fc_dx + m_xx * fc_dy
                            fc_curve_x.addKey(f, float(fc_cur_x))
                            fc_curve_y.addKey(f, float(fc_cur_y))
                else:
                    # If keep_tangents is False, ensure tangents stay clean zero cusps
                    if lt_curve_x and lt_curve_y:
                        lt_curve_x.addKey(f, 0.0)
                        lt_curve_y.addKey(f, 0.0)
                    if rt_curve_x and rt_curve_y:
                        rt_curve_x.addKey(f, 0.0)
                        rt_curve_y.addKey(f, 0.0)
                    if fc_curve_x and fc_curve_y:
                        fc_curve_x.addKey(f, 0.0)
                        fc_curve_y.addKey(f, 0.0)

            baked_count += 1

        curves.changed()
        return baked_count > 0


# =========================================================================
# Tracker-Style Knobs Setup for Native Roto Nodes
# =========================================================================

def get_target_shape(roto_node):
    """
    Finds the active or first shape within the Roto node.
    """
    curves = roto_node['curves']
    root = getattr(curves, 'rootLayer', getattr(curves, 'root', None))
    if not root:
        return None

    shapes = []
    def _walk(item):
        for elem in item:
            if isinstance(elem, rp.Shape) or hasattr(elem, 'getControlPoint') or type(elem).__name__ == 'Shape':
                shapes.append(elem)
            elif isinstance(elem, rp.Layer):
                _walk(elem)

    _walk(root)
    return shapes[0] if shapes else None


def setup_autoroto_knobs(node):
    """
    Equips a native Roto node with Tracker-style VCR buttons & CoTracker controls.
    """
    if node.knob('AutoRotoTrackerTab'):
        return

    # 1. Dedicated Tracker Tab
    tab = nuke.Tab_Knob('AutoRotoTrackerTab', 'AutoRoto Tracker')
    node.addKnob(tab)

    # 2. Header
    title = nuke.Text_Knob(
        'ar_title', '',
        '<font size=4 color="#e58934"><b>AutoRoto Tracker</b></font> '
        '<font color="#888">CoTracker3 GPU (RTX 4080)</font>'
    )
    node.addKnob(title)

    # 3. VCR Tracking Buttons Row (Identical to Tracker node)
    btn_to_start = nuke.PyScript_Knob(
        'ar_to_start', '|◀',
        'import nuke_bridge; nuke_bridge.on_node_track_to_start(nuke.thisNode())'
    )
    btn_to_start.setTooltip('Track from reference frame to start frame')

    btn_step_bwd = nuke.PyScript_Knob(
        'ar_step_bwd', '◀',
        'import nuke_bridge; nuke_bridge.on_node_track_step(nuke.thisNode(), -1)'
    )
    btn_step_bwd.setTooltip('Track 1 frame backward')

    btn_step_fwd = nuke.PyScript_Knob(
        'ar_step_fwd', '▶',
        'import nuke_bridge; nuke_bridge.on_node_track_step(nuke.thisNode(), 1)'
    )
    btn_step_fwd.setTooltip('Track 1 frame forward')

    btn_to_end = nuke.PyScript_Knob(
        'ar_to_end', '▶|',
        'import nuke_bridge; nuke_bridge.on_node_track_to_end(nuke.thisNode())'
    )
    btn_to_end.setTooltip('Track from reference frame to end frame')

    btn_range = nuke.PyScript_Knob(
        'ar_track_range', '<b><font color="#4caf50">🚀 Track Full Range</font></b>',
        'import nuke_bridge; nuke_bridge.on_node_track_range(nuke.thisNode())'
    )
    btn_range.setTooltip('Run CoTracker GPU on RTX 4080 across entire specified frame range')

    node.addKnob(btn_to_start)
    node.addKnob(btn_step_bwd)
    node.addKnob(btn_step_fwd)
    node.addKnob(btn_to_end)
    node.addKnob(btn_range)

    # 4. Divider & Frame Range
    node.addKnob(nuke.Text_Knob('ar_div1', ''))

    curr_f = int(nuke.frame())
    start_f = int(nuke.root()['first_frame'].value())
    end_f = int(nuke.root()['last_frame'].value())

    k_ref = nuke.Int_Knob('ar_ref_frame', 'Ref Frame')
    k_ref.setValue(curr_f)
    node.addKnob(k_ref)

    k_set_curr = nuke.PyScript_Knob(
        'ar_set_curr', 'Set Current',
        'import nuke_bridge; nuke_bridge.on_node_set_current_frame(nuke.thisNode())'
    )
    node.addKnob(k_set_curr)

    k_start = nuke.Int_Knob('ar_start_frame', 'Start Frame')
    k_start.setValue(start_f)
    node.addKnob(k_start)

    k_end = nuke.Int_Knob('ar_end_frame', 'End Frame')
    k_end.setValue(end_f)
    node.addKnob(k_end)

    k_sync_range = nuke.PyScript_Knob(
        'ar_sync_range', 'Project Range',
        'import nuke_bridge; nuke_bridge.on_node_sync_range(nuke.thisNode())'
    )
    node.addKnob(k_sync_range)

    # 5. Options
    k_tangents = nuke.Boolean_Knob('ar_keep_tangents', 'Preserve Curvature / Tangents')
    k_tangents.setValue(True)
    k_tangents.setTooltip('When enabled, rotates and preserves Bezier tangent handles naturally with tracked surface without wild loop distortion.')
    node.addKnob(k_tangents)

    btn_fix_tangents = nuke.PyScript_Knob(
        'ar_fix_tangents', 'Fix / Clean Tangents',
        'import nuke_bridge; nuke_bridge.on_node_fix_tangents(nuke.thisNode())'
    )
    btn_fix_tangents.setTooltip('Cleans corrupted Bezier tangent handles and restores crisp curvature matching reference frame, instantly eliminating loops.')
    node.addKnob(btn_fix_tangents)

    # 6. Roto Parameters Section (Original Roto settings directly accessible and synchronized here)
    node.addKnob(nuke.Text_Knob('ar_div_roto', '<b><font color="#e58934">Roto Settings</font></b>'))

    # Output channel mask (e.g. alpha, rgba)
    k_out = nuke.ChannelMask_Knob('ar_output', 'output')
    if node.knob('output'):
        k_out.setValue(node['output'].value())
    node.addKnob(k_out)

    # Premultiply
    k_premult = nuke.Channel_Knob('ar_premultiply', 'premultiply')
    if node.knob('premultiply'):
        k_premult.setValue(node['premultiply'].value())
    node.addKnob(k_premult)

    # Clip to format
    k_cliptype = nuke.Enumeration_Knob('ar_cliptype', 'clip to', ['no clip', 'bbox', 'format', 'union', 'intersect'])
    if node.knob('cliptype'):
        k_cliptype.setValue(node['cliptype'].value())
    node.addKnob(k_cliptype)

    k_replace = nuke.Boolean_Knob('ar_replace', 'replace')
    if node.knob('replace'):
        k_replace.setValue(bool(node['replace'].value()))
    node.addKnob(k_replace)

    # Opacity slider (0.0 to 1.0)
    k_opacity = nuke.Double_Knob('ar_opacity', 'opacity')
    k_opacity.setRange(0.0, 1.0)
    if node.knob('opacity'):
        k_opacity.setValue(float(node['opacity'].value()))
    else:
        k_opacity.setValue(1.0)
    node.addKnob(k_opacity)

    # Feather slider (-100 to 100)
    k_feather = nuke.Double_Knob('ar_feather', 'feather')
    k_feather.setRange(-100.0, 100.0)
    if node.knob('feather'):
        k_feather.setValue(float(node['feather'].value()))
    else:
        k_feather.setValue(0.0)
    node.addKnob(k_feather)

    # Feather falloff slider
    k_falloff = nuke.Double_Knob('ar_feather_falloff', 'feather falloff')
    k_falloff.setRange(0.2, 5.0)
    if node.knob('feather_falloff'):
        k_falloff.setValue(float(node['feather_falloff'].value()))
    else:
        k_falloff.setValue(1.0)
    node.addKnob(k_falloff)

    k_ftype = nuke.Enumeration_Knob('ar_feather_type', '', ['linear', 'smooth', 'ease-in', 'ease-out'])
    if node.knob('feather_type'):
        k_ftype.setValue(node['feather_type'].value())
    node.addKnob(k_ftype)

    # 7. Utilities & Status
    node.addKnob(nuke.Text_Knob('ar_div2', ''))

    k_open_panel = nuke.PyScript_Knob(
        'ar_open_panel', 'Open Full PySide Panel',
        'import main; main.launch_panel(dockable=False)'
    )
    node.addKnob(k_open_panel)

    k_check_gpu = nuke.PyScript_Knob(
        'ar_check_gpu', 'Check GPU Status',
        'import tracker_core; info = tracker_core.check_ai_environment(); nuke.message(f"Device: {info.get(\'device_name\')}\\nCUDA: {info.get(\'cuda\')}\\nPyTorch: {info.get(\'torch_version\')}")'
    )
    node.addKnob(k_check_gpu)

    k_status = nuke.String_Knob('ar_status', 'Status')
    k_status.setValue('Ready. Draw a shape on Ref Frame and click Track.')
    node.addKnob(k_status)

    # Set knobChanged callback for bidirectional synchronization between ar_* and native knobs
    cb_code = """
import nuke_bridge
nuke_bridge.on_autoroto_knob_changed(nuke.thisNode(), nuke.thisKnob())
"""
    node.knob('knobChanged').setValue(cb_code)


def on_autoroto_knob_changed(node, knob):
    """
    Synchronizes AutoRoto tab Roto controls with native Roto knobs bi-directionally.
    """
    if not node or not knob:
        return
    kname = knob.name()

    mapping = {
        'ar_output': 'output',
        'ar_premultiply': 'premultiply',
        'ar_cliptype': 'cliptype',
        'ar_replace': 'replace',
        'ar_opacity': 'opacity',
        'ar_feather': 'feather',
        'ar_feather_falloff': 'feather_falloff',
        'ar_feather_type': 'feather_type'
    }

    if kname in mapping:
        target = mapping[kname]
        if node.knob(target):
            try:
                node.knob(target).setValue(knob.value())
            except Exception:
                pass
    elif kname in mapping.values():
        for ar_k, nat_k in mapping.items():
            if nat_k == kname and node.knob(ar_k):
                try:
                    node.knob(ar_k).setValue(knob.value())
                except Exception:
                    pass


def create_autoroto_node():
    """
    Creates a native Nuke Roto node equipped with Tracker-style VCR buttons.
    """
    node = nuke.createNode('Roto')
    node.setName('AutoRoto1')
    setup_autoroto_knobs(node)
    if node.knob('AutoRotoTrackerTab'):
        node.knob('AutoRotoTrackerTab').setFlag(0)
    return node


def add_autoroto_to_selected():
    """
    Adds Tracker-style controls to the currently selected Roto node.
    """
    node = nuke.selectedNode()
    if node and node.Class() in ('Roto', 'RotoPaint'):
        setup_autoroto_knobs(node)
        nuke.message(f"Added AutoRoto Tracker controls to '{node.name()}'!")
    else:
        nuke.message("Please select a Roto or RotoPaint node first.")


# =========================================================================
# Node Callback Event Handlers (VCR Buttons on Roto Node)
# =========================================================================

def on_node_set_current_frame(roto_node):
    curr = int(nuke.frame())
    if roto_node.knob('ar_ref_frame'):
        roto_node.knob('ar_ref_frame').setValue(curr)
    if roto_node.knob('ar_status'):
        roto_node.knob('ar_status').setValue(f"Reference frame set to {curr}.")


def on_node_sync_range(roto_node):
    start = int(nuke.root()['first_frame'].value())
    end = int(nuke.root()['last_frame'].value())
    if roto_node.knob('ar_start_frame'):
        roto_node.knob('ar_start_frame').setValue(start)
    if roto_node.knob('ar_end_frame'):
        roto_node.knob('ar_end_frame').setValue(end)
    if roto_node.knob('ar_status'):
        roto_node.knob('ar_status').setValue(f"Synced project range: {start} - {end}.")


def on_node_fix_tangents(roto_node):
    """
    Cleans up any corrupted or looping tangents by setting them to the reference frame's
    tangent offsets (or zero if sharp corners), removing all wild Bezier loop petals.
    """
    shape = get_target_shape(roto_node)
    if not shape:
        nuke.message(f"No shape found in '{roto_node.name()}'.")
        return

    ref_f = int(roto_node['ar_ref_frame'].value()) if roto_node.knob('ar_ref_frame') else int(nuke.frame())
    bridge = NukeRotoBridge()
    pts_info = bridge.extract_shape_points(shape, ref_f)
    if not pts_info:
        nuke.message("No points found in shape.")
        return

    curves = roto_node['curves']
    fixed_count = 0

    for pt_data in pts_info:
        raw_pt = pt_data['raw_point']
        lt_dx, lt_dy = pt_data['lt_rel']
        rt_dx, rt_dy = pt_data['rt_rel']
        fc_dx, fc_dy = pt_data['fc_rel']

        is_lt_zero = (abs(lt_dx) < 1e-3 and abs(lt_dy) < 1e-3)
        is_rt_zero = (abs(rt_dx) < 1e-3 and abs(rt_dy) < 1e-3)
        is_fc_zero = (abs(fc_dx) < 1e-3 and abs(fc_dy) < 1e-3)

        targets = [
            (raw_pt.leftTangent, 0.0 if is_lt_zero else lt_dx, 0.0 if is_lt_zero else lt_dy),
            (raw_pt.rightTangent, 0.0 if is_rt_zero else rt_dx, 0.0 if is_rt_zero else rt_dy),
            (raw_pt.featherCenter, 0.0 if is_fc_zero else fc_dx, 0.0 if is_fc_zero else fc_dy),
        ]

        for comp, target_x, target_y in targets:
            c_x = comp.getPositionAnimCurve(0)
            c_y = comp.getPositionAnimCurve(1)
            if c_x and c_y:
                for k in list(c_x.keys()):
                    c_x.addKey(k.x, float(target_x))
                    c_y.addKey(k.x, float(target_y))

        fixed_count += 1

    curves.changed()
    msg = f"Cleaned tangents for {fixed_count} points! Wild loops removed."
    if roto_node.knob('ar_status'):
        roto_node.knob('ar_status').setValue(msg)
    nuke.message(f"AutoRoto: {msg}")


def _run_tracking_for_range(roto_node, start_f: int, end_f: int, ref_f: int):
    source_node = roto_node.input(0)
    if not source_node:
        nuke.message(f"Please connect video/footage to input of '{roto_node.name()}'.")
        return

    shape = get_target_shape(roto_node)
    if not shape:
        nuke.message(f"No shape found in '{roto_node.name()}'. Please draw a roto shape at frame {ref_f}.")
        return

    bridge = NukeRotoBridge()
    pts_info = bridge.extract_shape_points(shape, ref_f)
    if not pts_info:
        nuke.message(f"Shape '{shape.name}' has no control points at frame {ref_f}.")
        return

    t_rel = float(ref_f - start_f)
    queries = [{"index": p["index"], "t": t_rel, "x": float(p["x"]), "y": float(p["y"])} for p in pts_info]

    progress = nuke.ProgressTask(f"AutoRoto: Tracking {len(pts_info)} points on RTX 4080...")
    progress.setProgress(15)
    progress.setMessage("Exporting frames...")

    temp_dir = None
    try:
        image_paths, temp_dir = bridge.export_source_frames(source_node, start_f, end_f)
        progress.setProgress(45)
        progress.setMessage("Running CoTracker GPU inference...")

        tracks = tracker_core.run_cotracker_point_tracking(
            image_paths=image_paths,
            queries=queries,
            start_frame=start_f
        )

        progress.setProgress(85)
        progress.setMessage("Baking keyframes to Roto...")

        keep_tg = bool(roto_node['ar_keep_tangents'].value()) if roto_node.knob('ar_keep_tangents') else True
        ok = bridge.bake_tracking_data_to_roto(
            roto_node, shape, tracks, ref_f, keep_tangents=keep_tg
        )

        progress.setProgress(100)
        if ok:
            msg = f"✓ Tracked & Baked {len(pts_info)} points across frames {start_f}-{end_f}!"
            if roto_node.knob('ar_status'):
                roto_node.knob('ar_status').setValue(msg)
            nuke.message(f"AutoRoto: Complete!\n{len(pts_info)} points tracked and baked across {end_f - start_f + 1} frames.")
        else:
            if roto_node.knob('ar_status'):
                roto_node.knob('ar_status').setValue("Bake failed.")

    except Exception as e:
        if roto_node.knob('ar_status'):
            roto_node.knob('ar_status').setValue(f"Error: {str(e)}")
        nuke.message(f"AutoRoto Error:\n{str(e)}")

    finally:
        del progress
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def on_node_track_range(roto_node):
    ref_f = int(roto_node['ar_ref_frame'].value())
    start_f = int(roto_node['ar_start_frame'].value())
    end_f = int(roto_node['ar_end_frame'].value())
    if start_f >= end_f:
        nuke.message("Start frame must be less than End frame.")
        return
    _run_tracking_for_range(roto_node, start_f, end_f, ref_f)


def on_node_track_to_end(roto_node):
    ref_f = int(roto_node['ar_ref_frame'].value())
    end_f = int(roto_node['ar_end_frame'].value())
    if ref_f >= end_f:
        nuke.message("Reference frame must be less than End frame.")
        return
    _run_tracking_for_range(roto_node, ref_f, end_f, ref_f)


def on_node_track_to_start(roto_node):
    ref_f = int(roto_node['ar_ref_frame'].value())
    start_f = int(roto_node['ar_start_frame'].value())
    if start_f >= ref_f:
        nuke.message("Start frame must be less than Reference frame.")
        return
    _run_tracking_for_range(roto_node, start_f, ref_f, ref_f)


def on_node_track_step(roto_node, direction: int):
    curr = int(nuke.frame())
    target = curr + direction
    start_f = min(curr, target)
    end_f = max(curr, target)
    _run_tracking_for_range(roto_node, start_f, end_f, curr)
    nuke.frame(target)
