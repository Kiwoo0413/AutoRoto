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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg.get("model", "cotracker3_offline")

    # 1. Load Image Frames
    frames = []
    h, w = 0, 0
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[0], arr.shape[1]
        frames.append(arr)

    # Tensor: (1, T, 3, H, W)
    video_tensor = torch.from_numpy(np.stack(frames, axis=0))
    video_tensor = video_tensor.permute(0, 3, 1, 2).unsqueeze(0).to(device)

    # 2. Convert Queries (Nuke bottom-left -> Image top-left)
    queries = []
    for q in queries_input:
        t_rel = float(q["t"])
        x_nuke = float(q["x"])
        y_nuke = float(q["y"])
        y_img = max(0.0, min(float(h - 1), float(h - 1) - y_nuke))
        queries.append((t_rel, x_nuke, y_img))

    # queries shape: (1, N, 3)
    queries_tensor = torch.tensor(queries, dtype=torch.float32, device=device).unsqueeze(0)

    # 3. Load CoTracker Model & Run Inference
    try:
        import cotracker
        if hasattr(cotracker, "build_cotracker"):
            model = cotracker.build_cotracker(checkpoint=None).to(device)
        else:
            model = torch.hub.load("facebookresearch/co-tracker", model_name).to(device)
    except Exception:
        model = torch.hub.load("facebookresearch/co-tracker", model_name).to(device)

    with torch.no_grad():
        pred_tracks, pred_vis = model(video_tensor, queries=queries_tensor, backward_tracking=True)

    pred_tracks = pred_tracks.cpu().numpy()
    pred_vis = pred_vis.cpu().numpy() if pred_vis is not None else None

    T = pred_tracks.shape[1]
    N = pred_tracks.shape[2]

    # 4. Format Output & Convert Back to Nuke Coordinates
    results = {}
    for i, q in enumerate(queries_input):
        idx = q["index"]
        results[str(idx)] = {}
        for t_idx in range(T):
            actual_f = start_frame + t_idx
            x_img = float(pred_tracks[0, t_idx, i, 0])
            y_img = float(pred_tracks[0, t_idx, i, 1])
            vis_val = float(pred_vis[0, t_idx, i]) if pred_vis is not None else 1.0

            # Convert to Nuke bottom-left
            x_nuke = x_img
            y_nuke = float(h - 1) - y_img

            results[str(idx)][str(actual_f)] = {
                "x": x_nuke,
                "y": y_nuke,
                "vis": vis_val
            }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"status": "success", "tracks": results}, f, indent=2)

if __name__ == "__main__":
    main()
'''

def run_cotracker_point_tracking(
    image_paths: List[str],
    queries: List[Dict[str, Any]],
    start_frame: int,
    custom_python_path: Optional[str] = None,
    model_name: str = "cotracker3_offline",
    timeout_sec: int = 300
) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    """
    Executes CoTracker 3 offline GPU tracking.
    queries: List of dicts: [{"index": int, "t": float, "x": float, "y": float}, ...]
    Returns:
      {
         track_idx: {
             frame: (x, y, visibility)
         }
      }
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
        "model": model_name
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f)

    cmd = [py_exe, worker_script, "--config", config_path]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
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
