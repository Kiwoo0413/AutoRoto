"""
AutoRoto CoTracker GPU Backend (tracker_core.py)
Dedicated Meta CoTracker 3 offline deep learning point tracking backend.
Uses PyTorch + CUDA (NVIDIA GeForce RTX 4080) for bidirectional zero-drift tracking.
"""

import os
import sys
import json
import subprocess
import tempfile
from typing import List, Dict, Any, Tuple, Optional

# Default discovered Python executable with PyTorch & CUDA
KNOWN_AI_PYTHONS = [
    r"C:\Users\yido2\AppData\Local\Programs\Python\Python312\python.exe",
    "python",
    "python3"
]

def find_ai_python(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Locates Python interpreter with PyTorch and CUDA support.
    """
    candidates = []
    if custom_path and os.path.exists(custom_path.strip()):
        candidates.append(custom_path.strip())

    candidates.extend(KNOWN_AI_PYTHONS)
    if sys.executable not in candidates:
        candidates.append(sys.executable)

    for py_exe in candidates:
        if not os.path.isfile(py_exe) and not any(py_exe.startswith(prefix) for prefix in ("python", "python3")):
            continue
        try:
            cmd = [py_exe, "-c", "import torch; print(torch.cuda.is_available())"]
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=6).decode().strip()
            if "True" in out or "False" in out:
                return py_exe
        except Exception:
            continue

    return None


def check_ai_environment(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Diagnostic report for PyTorch, CUDA, and CoTracker.
    """
    py_exe = find_ai_python(custom_path)
    if not py_exe:
        return {
            'available': False,
            'python_path': None,
            'cuda': False,
            'device_name': 'None',
            'torch_version': 'None',
            'message': 'No PyTorch environment detected.'
        }

    try:
        check_script = (
            "import torch, json\n"
            "has_cuda = torch.cuda.is_available()\n"
            "device_name = torch.cuda.get_device_name(0) if has_cuda else 'CPU'\n"
            "print(json.dumps({'cuda': has_cuda, 'device': device_name, 'torch': torch.__version__}))\n"
        )
        out = subprocess.check_output([py_exe, "-c", check_script], timeout=8).decode().strip()
        data = json.loads(out.splitlines()[-1])
        return {
            'available': True,
            'python_path': py_exe,
            'cuda': data.get('cuda', False),
            'device_name': data.get('device', 'CPU'),
            'torch_version': data.get('torch', ''),
            'message': f"PyTorch {data.get('torch')} ready on {data.get('device')}"
        }
    except Exception as e:
        return {
            'available': False,
            'python_path': py_exe,
            'cuda': False,
            'device_name': 'None',
            'torch_version': 'None',
            'message': f"Check failed: {str(e)}"
        }


# Worker script code executed by external PyTorch Python process
_WORKER_SCRIPT_CONTENT = '''
import os
import sys
import json
import argparse
from PIL import Image
import numpy as np
import torch

def emit_progress(percent: float, message: str):
    payload = json.dumps({"type": "progress", "percent": round(float(percent), 1), "message": str(message)})
    print(f"PROGRESS:{payload}", flush=True)

def load_frame_array(path, max_size):
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    if max_size > 0 and max(orig_w, orig_h) > max_size:
        scale = float(max_size) / float(max(orig_w, orig_h))
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        img_resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        scale_x = float(new_w) / float(orig_w)
        scale_y = float(new_h) / float(orig_h)
        arr = np.array(img_resized, dtype=np.float32)
        return arr, orig_w, orig_h, scale_x, scale_y
    else:
        arr = np.array(img, dtype=np.float32)
        return arr, orig_w, orig_h, 1.0, 1.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    image_paths = cfg["image_paths"]
    queries_input = cfg["queries"]  # [{"index": 0, "t": rel_f, "x": x_nuke, "y": y_nuke}, ...]
    start_frame = cfg["start_frame"]
    output_path = cfg["output_path"]
    max_size = int(cfg.get("max_size", 720))
    chunk_size = int(cfg.get("chunk_size", 100))
    model_name = cfg.get("model", "cotracker3_offline")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    total_frames = len(image_paths)
    if total_frames == 0 or not queries_input:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"status": "success", "tracks": {}}, f)
        return

    emit_progress(5.0, f"Initializing AI runtime on {device.upper()}...")

    # Load CoTracker Model
    emit_progress(10.0, "Loading CoTracker 3 model weights into GPU VRAM...")
    try:
        import cotracker
        if hasattr(cotracker, "build_cotracker"):
            model = cotracker.build_cotracker(checkpoint=None).to(device)
        else:
            model = torch.hub.load("facebookresearch/co-tracker", model_name).to(device)
    except Exception:
        model = torch.hub.load("facebookresearch/co-tracker", model_name).to(device)
    model.eval()

    emit_progress(20.0, "Model ready. Analyzing sequence and building chunks...")

    # Reference frame index in image_paths (relative to start_frame)
    ref_idx = int(round(float(queries_input[0]["t"])))
    ref_idx = max(0, min(total_frames - 1, ref_idx))

    # Read first frame to determine original dimensions and scale factors
    test_arr, orig_w, orig_h, scale_x, scale_y = load_frame_array(image_paths[0], max_size)

    # Convert initial queries from Nuke bottom-left to scaled image top-left
    initial_scaled_coords = {}
    for q in queries_input:
        pt_idx = q["index"]
        x_nuke = float(q["x"])
        y_nuke = float(q["y"])
        x_orig = x_nuke
        y_orig = max(0.0, min(float(orig_h - 1), float(orig_h - 1) - y_nuke))
        x_scaled = x_orig * scale_x
        y_scaled = y_orig * scale_y
        initial_scaled_coords[pt_idx] = (x_scaled, y_scaled)

    # Dictionary to collect all results:
    tracked_scaled_results = {q["index"]: {} for q in queries_input}
    for pt_idx, (sx, sy) in initial_scaled_coords.items():
        tracked_scaled_results[pt_idx][ref_idx] = (sx, sy, 1.0)

    # Build Chunks Plan
    chunks_plan = []
    if total_frames <= chunk_size:
        chunks_plan.append({"type": "single", "start": 0, "end": total_frames, "query_t": ref_idx, "queries": initial_scaled_coords})
    else:
        # Forward chunks from ref_idx to total_frames - 1
        if ref_idx < total_frames - 1:
            fwd_cur = ref_idx
            while fwd_cur < total_frames - 1:
                fwd_end = min(total_frames, fwd_cur + chunk_size)
                chunks_plan.append({"type": "forward", "start": fwd_cur, "end": fwd_end})
                fwd_cur = fwd_end - 1  # 1-frame boundary overlap

        # Backward chunks from ref_idx down to 0
        if ref_idx > 0:
            bwd_cur = ref_idx
            while bwd_cur > 0:
                bwd_start = max(0, bwd_cur - chunk_size + 1)
                chunks_plan.append({"type": "backward", "start": bwd_start, "end": bwd_cur + 1})
                bwd_cur = bwd_start

    total_chunks = len(chunks_plan)
    res_label = f"{max_size}p downscale" if max_size > 0 else "full res"
    emit_progress(22.0, f"Tracking {total_frames} frames ({res_label}) across {total_chunks} chunk(s)...")

    # Execute Chunks
    for c_i, plan in enumerate(chunks_plan):
        c_num = c_i + 1
        p_base = 22.0 + (float(c_i) / float(total_chunks)) * 68.0
        p_step = 68.0 / float(total_chunks)

        s_idx = plan["start"]
        e_idx = plan["end"]
        L = e_idx - s_idx
        chunk_paths = image_paths[s_idx:e_idx]

        f_start_label = start_frame + s_idx
        f_end_label = start_frame + e_idx - 1
        emit_progress(p_base + p_step * 0.1, f"Chunk {c_num}/{total_chunks}: Loading {L} frames ({f_start_label} - {f_end_label})...")

        # Load and stack frames for this chunk
        chunk_frames = []
        for p in chunk_paths:
            f_arr, _, _, _, _ = load_frame_array(p, max_size)
            chunk_frames.append(f_arr)

        video_chunk = torch.from_numpy(np.stack(chunk_frames, axis=0))
        video_chunk = video_chunk.permute(0, 3, 1, 2).unsqueeze(0).to(device)

        pt_indices = [q["index"] for q in queries_input]
        queries_tensor_list = []

        if plan["type"] == "single":
            q_t = float(plan["query_t"])
            for pt_idx in pt_indices:
                qx, qy = plan["queries"][pt_idx]
                queries_tensor_list.append((q_t, qx, qy))
            backward_mode = True

        elif plan["type"] == "forward":
            q_t = 0.0
            for pt_idx in pt_indices:
                qx, qy, _ = tracked_scaled_results[pt_idx][s_idx]
                queries_tensor_list.append((q_t, qx, qy))
            backward_mode = False

        elif plan["type"] == "backward":
            q_t = float(L - 1)
            anchor_idx = e_idx - 1
            for pt_idx in pt_indices:
                qx, qy, _ = tracked_scaled_results[pt_idx][anchor_idx]
                queries_tensor_list.append((q_t, qx, qy))
            backward_mode = True

        queries_tensor = torch.tensor(queries_tensor_list, dtype=torch.float32, device=device).unsqueeze(0)

        emit_progress(p_base + p_step * 0.45, f"Chunk {c_num}/{total_chunks}: GPU CoTracker inference on RTX 4080 ({L} frames)...")

        with torch.no_grad():
            pred_tracks, pred_vis = model(video_chunk, queries=queries_tensor, backward_tracking=backward_mode)

        pred_tracks = pred_tracks.cpu().numpy()
        pred_vis = pred_vis.cpu().numpy() if pred_vis is not None else None

        # Store predictions
        for pt_local_i, pt_idx in enumerate(pt_indices):
            for local_t in range(L):
                global_frame_idx = s_idx + local_t
                pred_x = float(pred_tracks[0, local_t, pt_local_i, 0])
                pred_y = float(pred_tracks[0, local_t, pt_local_i, 1])
                vis_v = float(pred_vis[0, local_t, pt_local_i]) if pred_vis is not None else 1.0
                tracked_scaled_results[pt_idx][global_frame_idx] = (pred_x, pred_y, vis_v)

        del video_chunk
        del queries_tensor
        if device == "cuda":
            torch.cuda.empty_cache()

        emit_progress(p_base + p_step * 0.95, f"Chunk {c_num}/{total_chunks} complete ({f_start_label}-{f_end_label}).")

    emit_progress(92.0, "Converting trajectories to Nuke resolution...")

    # Convert all scaled results back to original Nuke coordinates
    final_output = {}
    for pt_idx, frames_map in tracked_scaled_results.items():
        final_output[str(pt_idx)] = {}
        for f_idx, (sx, sy, vis_val) in frames_map.items():
            actual_frame = start_frame + f_idx
            x_orig = sx / scale_x
            y_orig = sy / scale_y
            x_nuke = x_orig
            y_nuke = float(orig_h - 1) - y_orig

            final_output[str(pt_idx)][str(actual_frame)] = {
                "x": x_nuke,
                "y": y_nuke,
                "vis": vis_val
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"status": "success", "tracks": final_output}, f, indent=2)

    emit_progress(95.0, "Tracking computation complete! Returning data to Nuke.")

if __name__ == "__main__":
    main()
'''

def run_cotracker_point_tracking(
    image_paths: List[str],
    queries: List[Dict[str, Any]],
    start_frame: int,
    custom_python_path: Optional[str] = None,
    model_name: str = "cotracker3_offline",
    max_size: int = 720,
    chunk_size: int = 100,
    progress_callback: Optional[Any] = None,
    timeout_sec: int = 600
) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    """
    Executes CoTracker 3 offline GPU tracking with real-time progress streaming,
    intelligent downscaling, and 100-frame chunking.
    """
    py_exe = find_ai_python(custom_python_path)
    if not py_exe:
        raise RuntimeError("CoTracker: PyTorch/CUDA environment not found.")

    temp_dir = tempfile.mkdtemp(prefix="autoroto_cotracker_")
    worker_script = os.path.join(temp_dir, "worker.py")
    config_path = os.path.join(temp_dir, "config.json")
    output_path = os.path.join(temp_dir, "output.json")

    with open(worker_script, "w", encoding="utf-8") as f:
        f.write(_WORKER_SCRIPT_CONTENT)

    config_data = {
        "image_paths": image_paths,
        "queries": queries,
        "start_frame": start_frame,
        "output_path": output_path,
        "model": model_name,
        "max_size": max_size,
        "chunk_size": chunk_size
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    cmd = [py_exe, "-u", worker_script, "--config", config_path]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        encoding="utf-8"
    )

    try:
        # Stream stdout line-by-line in real time
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("PROGRESS:"):
                payload = None
                try:
                    payload = json.loads(line[9:])
                    pct = float(payload.get("percent", 0.0))
                    msg = str(payload.get("message", ""))
                except Exception:
                    payload = None
                if payload is not None and progress_callback:
                    progress_callback(pct, msg)

        proc.wait(timeout=timeout_sec)
        if proc.returncode != 0:
            err = proc.stderr.read().strip()
            raise RuntimeError(f"CoTracker worker failed (code {proc.returncode}):\n{err}")

        if not os.path.isfile(output_path):
            raise FileNotFoundError(f"CoTracker output missing: {output_path}")

        with open(output_path, "r", encoding="utf-8") as f:
            out_data = json.load(f)

        raw_tracks = out_data.get("tracks", {})
        parsed: Dict[int, Dict[int, Tuple[float, float, float]]] = {}

        for pt_str, frames_dict in raw_tracks.items():
            pt_idx = int(pt_str)
            parsed[pt_idx] = {}
            for f_str, val_dict in frames_dict.items():
                f_num = int(f_str)
                x = float(val_dict["x"])
                y = float(val_dict["y"])
                vis = float(val_dict.get("vis", 1.0))
                parsed[pt_idx][f_num] = (x, y, vis)

        return parsed

    except Exception:
        if proc.poll() is None:
            proc.kill()
        raise

    finally:
        try:
            if os.path.exists(worker_script):
                os.remove(worker_script)
            if os.path.exists(config_path):
                os.remove(config_path)
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rmdir(temp_dir)
        except Exception:
            pass
