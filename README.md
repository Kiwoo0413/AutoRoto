<a id="top"></a>

# AutoRoto (v2.1) — Foundry Nuke Native Roto Node with Tracker Buttons & CoTracker 3 GPU

<div align="center">

**🌐 Language / 언어 선택:** &nbsp;
[**🇰🇷 한국어**](#korean) &nbsp;&nbsp;|&nbsp;&nbsp; [**🇺🇸 English**](#english)

</div>

---

<a id="korean"></a>

# 🇰🇷 한국어 (Korean)

<p align="right">
  <a href="#english"><b>🌐 Switch to English (영문 버전 보기)</b></a>
</p>

**AutoRoto**는 **100% 네이티브 Foundry Nuke Roto 노드**(Nuke 13.0–17.x+)의 1번 탭에 Tracker 스타일 VCR 버튼을 직접 탑재하고, **Meta CoTracker 3 GPU** 백엔드로 모든 베지에/B-Spline 제어점을 자동 추적해 키프레임을 구워주는 로토 트래킹 툴킷입니다.

Nuke 내장 Python과의 패키지 충돌을 피하기 위해 Nuke 내부 UI·커브 제어는 브리지([nuke_bridge.py](nuke_bridge.py))가 담당하며(`curves.rootLayer` API 정규화로 전 버전 호환), 무거운 GPU 딥러닝 추론은 외부 Python 워커([tracker_core.py](tracker_core.py)) 서브프로세스로 분리 실행합니다.

<https://github.com/user-attachments/assets/3fdd6cf2-98cc-49c6-ae54-dacb82a579e3>

---

## 1. Nuke 설치 방법 (Installation)

### 1) 저장소 클론 및 `init.py` 등록

Nuke 사용자 플러그인 디렉터리(`~/.nuke/`)에 저장소를 클론합니다 *(폴더명이 반드시 `AutoRoto`여야 합니다)*:

```bash
cd ~/.nuke
git clone https://github.com/Kiwoo0413/AutoRoto.git
```

`~/.nuke/init.py` (Windows 예: `C:\Users\<사용자명>\.nuke\init.py`)에 아래 2줄을 추가하면 Nuke 실행 시 메뉴와 단축키가 자동 등록됩니다:

```python
import nuke
nuke.pluginAddPath('./AutoRoto')
```

*(또는 절대 경로: `nuke.pluginAddPath(r"C:\Users\<사용자명>\.nuke\AutoRoto")`)*

- **툴바 메뉴**: `Nodes > AutoRoto > Create AutoRoto Node` (및 `Draw > AutoRoto (Tracker Buttons)`)
- **단축키**: `Ctrl+Alt+R`
- **GPU 상태 확인**: AutoRoto 노드 1번 탭의 `Check GPU Status` 버튼 *(Nuke 상단 메뉴 바에는 등록되지 않습니다)*

### 2) 외부 Python & AI 백엔드 환경 준비

AutoRoto는 Nuke 내장 파이썬과 분리된 외부 Python(3.10 ~ 3.12)의 PyTorch + CUDA 환경을 사용합니다:

```bash
# CUDA 지원 PyTorch 설치 (예: CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy Pillow

# (선택) CoTracker 패키지 설치 (미설치 시 torch.hub를 통해 모델 자동 로드)
pip install git+https://github.com/facebookresearch/co-tracker.git
```

> 가상환경(Conda 등)이나 특정 경로의 Python을 우선 사용하려면 시스템 환경 변수 `AUTOROTO_PYTHON`에 해당 `python.exe`의 절대 경로를 지정하세요.

---

## 2. 사용 방법 (Usage)

1. **노드 생성 또는 기존 Roto 변환**:
   - 노드 그래프에서 **`Ctrl+Alt+R`** (또는 `Tab` → `AutoRoto`, 툴바 `Nodes > AutoRoto`)로 네이티브 `AutoRoto` 노드를 생성합니다.
   - 이미 작업 중인 일반 `Roto` 노드가 있다면, 해당 노드를 선택한 뒤 툴바 **`Nodes > AutoRoto > Convert Selected Roto to AutoRoto`** 를 실행해 기존 셰이프를 유지한 채 1번 탭에 트래커 기능을 추가할 수 있습니다.
2. **1번 탭 VCR 트래커 버튼**:
   - `|◀` : 기준 프레임부터 시작 프레임까지 역방향 트래킹
   - `◀` : 1스텝 뒤로 트래킹 (`Key Step` 간격)
   - `▶` : 1스텝 앞으로 트래킹 (`Key Step` 간격)
   - `▶|` : 기준 프레임부터 끝 프레임까지 순방향 트래킹
   - **`🚀 Track Full Range`** : 지정한 전체 구간을 CoTracker 3 GPU로 일괄 추적 후 자동 베이크
3. **주요 파라미터**:
   - **`Ref Frame`** *(단일 & 멀티 통합 입력)*: 기준 프레임을 입력합니다. 단일 숫자(`1`)뿐 아니라 쉼표로 구분된 복수 프레임(`1, 45, 90`)을 지원하며, 비워두면 셰이프의 기존 키프레임을 자동 감지합니다.
   - **`[+ Add Current]` / `[Clear]`**: 현재 플레이헤드 프레임을 `Ref Frame` 목록에 추가하거나 필드를 초기화합니다.
   - **`Key Step`**: 키프레임 베이킹 간격 (`1` = 매 프레임, `2` = 2프레임마다 등).
   - **`Tracking Res`**: `720p (Fast / AI Optimized)` *(기본값, 3~5배 고속 추론 + 서브픽셀 원본 좌표 복원)* / `960p (Balanced)` / `Full (Original / Slow)`.
   - **`Preserve Curvature / Tangents`**: 표면 변형 및 회전에 맞춰 베지에 핸들 곡률을 자연스럽게 보존합니다.
   - **`Fix / Clean Tangents`**: 꼬인 베지에 핸들을 원클릭으로 정돈합니다.
4. **싱글 & 멀티 레퍼런스 양방향 트래킹**:
   - **단일 기준 프레임**: 해당 프레임을 앵커로 GPU 트래킹을 수행합니다.
   - **복수 기준 프레임 (`1, 45, 90` 등)**: 구간별 **순방향(Forward) + 역방향(Backward) CoTracker 3 양방향 추적**을 수행하고, $C^1$ Smoothstep ($s(t) = 3t^2 - 2t^3$) 및 Feature Visibility 가중치로 무봉제 블렌딩합니다. 아티스트가 직접 만든 기준 키프레임은 100% 보존되며 사이 구간(In-between)만 채워집니다.
   - 실시간 진행률 바를 통해 진행 상황을 확인하고 언제든 `Cancel`로 중단할 수 있습니다.
5. **선택 셰이프 독립 트래킹 규칙**:
   - Roto 커브 리스트나 뷰어에서 **선택한 셰이프만** 트래킹되며, 비선택 셰이프는 수정되지 않습니다.
   - 여러 셰이프를 동시에 선택하면 한 번의 GPU 패스로 함께 추론한 뒤 각 셰이프 커브에 **독립적으로** 베이크됩니다.
   - 레이어를 선택하면 해당 레이어 안의 모든 셰이프가 대상이 됩니다.
   - `Ref Frame`을 비워두면 셰이프마다 자신의 키프레임을 감지해 앵커로 사용하며(모든 셰이프 키가 같을 때만 `Ref Frame` 필드에 기록), 아무것도 선택하지 않으면 커브 리스트 최상단 셰이프 1개를 대상으로 합니다. VCR 버튼과 `Fix / Clean Tangents`도 동일한 규칙을 따릅니다.

---

## 3. 지원 환경 및 파일 구성

| 항목 | 지원 사양 |
| :--- | :--- |
| **Nuke 버전** | Foundry Nuke 13.0 ~ 17.x+ *(전 버전 호환)* |
| **UI 방식** | 100% 네이티브 Roto 노드 일체형 (Tab 1 직접 제어) |
| **운영체제(OS)** | Windows 10/11, Linux (Rocky, CentOS, Ubuntu) |
| **GPU 환경** | NVIDIA CUDA 지원 GPU |
| **AI 런타임** | Python 3.10 ~ 3.12, PyTorch 2.0+ (CUDA) |

```text
~/.nuke/AutoRoto/
├── init.py                        # AutoRoto sys.path 등록
├── menu.py                        # 툴바(Draw/Nodes), 단축키(Ctrl+Alt+R) 등록 (상단 메뉴 바 미등록)
├── nuke_bridge.py                 # Nuke Roto 노드 제어, Tab 1 배치, 베지에 탄젠트 복원 및 키프레임 베이킹
├── tracker_core.py                # CoTracker 3 GPU 딥러닝 트래커 (100프레임 청크, 스트리밍, 다운스케일)
├── test_cotracker_backend.py      # CoTracker 백엔드 단위 테스트
├── test_tangent_math.py           # 탄젠트 수학 및 Key Step 검증 테스트
├── test_multi_ref_tracking.py     # 멀티 레퍼런스 양방향 블렌딩 및 키프레임 보존 테스트
├── test_nuke_bridge_callbacks.py  # 브리지 콜백 및 레퍼런스 파싱/검증 테스트
├── test_shape_selection.py        # 선택 셰이프 타겟팅 및 셰이프별 독립 트래킹 테스트
├── README.md                      # 사용자 매뉴얼
└── SCRIPT_STRUCTURE.md            # 아키텍처 및 내부 구조 상세 문서
```

---

## 4. 헤드리스 테스트 실행

Nuke GUI를 실행하지 않고도 CoTracker 백엔드, 탄젠트 수학, 멀티 레퍼런스 블렌딩, 브리지 콜백, 선택 셰이프 타겟팅을 검증할 수 있습니다:

```bash
python -m unittest discover -v
```

<br>
<p align="center">
  <a href="#top"><b>⬆️ 맨 위로 이동 (Back to Top)</b></a> &nbsp;|&nbsp; <a href="#english"><b>🌐 Switch to English</b></a>
</p>

---
---

<a id="english"></a>

# 🇺🇸 English

<p align="right">
  <a href="#korean"><b>🌐 한국어로 전환 (Switch to Korean)</b></a>
</p>

**AutoRoto** is a **100% native Foundry Nuke Roto node toolkit** (Nuke 13.0–17.x+) that embeds Tracker-style VCR controls directly on Tab 1 of the Roto Properties panel and drives **Meta CoTracker 3 GPU** to track and bake keyframes onto every Bézier / B-Spline control point.

To avoid dependency conflicts with Nuke's bundled Python, Nuke-side UI and curve management run in [nuke_bridge.py](nuke_bridge.py) (normalized on the `curves.rootLayer` API for full cross-version compatibility), while heavy GPU inference runs in an isolated external Python worker subprocess ([tracker_core.py](tracker_core.py)).

<https://github.com/user-attachments/assets/3fdd6cf2-98cc-49c6-ae54-dacb82a579e3>

---

## 1. Nuke Installation

### 1) Clone Repository & Register `init.py`

Clone the repository into your Nuke user plugin directory (`~/.nuke/`) *(the folder name must be `AutoRoto`)*:

```bash
cd ~/.nuke
git clone https://github.com/Kiwoo0413/AutoRoto.git
```

Add the following two lines to `~/.nuke/init.py` (e.g., `C:\Users\<Username>\.nuke\init.py` on Windows) to register the menus and hotkeys on Nuke startup:

```python
import nuke
nuke.pluginAddPath('./AutoRoto')
```

*(Or use an absolute path: `nuke.pluginAddPath(r"C:\Users\<Username>\.nuke\AutoRoto")`)*

- **Toolbar Menu**: `Nodes > AutoRoto > Create AutoRoto Node` (and `Draw > AutoRoto (Tracker Buttons)`)
- **Shortcut**: `Ctrl+Alt+R`
- **GPU Status Check**: `Check GPU Status` button on Tab 1 of the AutoRoto node *(nothing is added to Nuke's top menu bar)*

### 2) External Python & AI Backend Setup

AutoRoto runs CoTracker 3 inside an external Python environment (3.10–3.12) with PyTorch + CUDA:

```bash
# Install CUDA-enabled PyTorch (e.g., CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy Pillow

# (Optional) Install CoTracker directly (auto-loaded via torch.hub if omitted)
pip install git+https://github.com/facebookresearch/co-tracker.git
```

> To point AutoRoto to a specific Conda/virtualenv `python.exe`, set the `AUTOROTO_PYTHON` environment variable to its absolute path.

---

## 2. Usage

1. **Create an AutoRoto Node or Convert an Existing Roto**:
   - Press **`Ctrl+Alt+R`** (or `Tab` → `AutoRoto`, or `Nodes > AutoRoto`) to create a native `AutoRoto` node.
   - To upgrade an existing `Roto` node without losing any shapes, select it and run **`Nodes > AutoRoto > Convert Selected Roto to AutoRoto`** from the toolbar.
2. **Tab 1 VCR Tracker Buttons**:
   - `|◀` : Track backward from reference frame to start frame
   - `◀` : Step backward by `Key Step`
   - `▶` : Step forward by `Key Step`
   - `▶|` : Track forward from reference frame to end frame
   - **`🚀 Track Full Range`** : Track and bake the entire frame range via CoTracker 3 GPU
3. **Key Parameters**:
   - **`Ref Frame`** *(Unified Single & Multi-Ref Input)*: Enter a single frame (e.g., `1`) or comma-separated reference frames (e.g., `1, 45, 90`). Leave blank to auto-detect existing keyframes on the shape.
   - **`[+ Add Current]` / `[Clear]`**: Append the current playhead frame to `Ref Frame` or clear the field.
   - **`Key Step`**: Keyframe baking interval (`1` = every frame, `2` = every 2 frames, etc.).
   - **`Tracking Res`**: `720p (Fast / AI Optimized)` *(default, 3–5× faster inference with lossless subpixel coordinate restoration)*, `960p (Balanced)`, or `Full (Original / Slow)`.
   - **`Preserve Curvature / Tangents`**: Preserves Bézier handle curvature across surface deformation and rotation.
   - **`Fix / Clean Tangents`**: One-click cleanup for tangled Bézier handles.
4. **Single & Multi-Reference Bidirectional Tracking**:
   - **Single reference frame**: Tracks using that frame as the anchor.
   - **Multiple reference frames (`1, 45, 90`, etc.)**: Runs **Forward + Backward CoTracker 3 bidirectional tracking** per interval and blends seamlessly with $C^1$ Smoothstep ($s(t) = 3t^2 - 2t^3$) and feature visibility weighting. Artist keyframes are preserved 100% while only in-between frames are baked.
   - Live progress is displayed with an interactive `Cancel` option.
5. **Selected-Shape Independent Tracking Rules**:
   - Only **selected shapes** in the Roto curve list or Viewer are tracked; unselected shapes remain untouched.
   - Selecting multiple shapes tracks them together in a single GPU pass and bakes each shape's curve **independently**.
   - Selecting a layer targets all shapes inside that layer.
   - When `Ref Frame` is blank, each shape detects its own keyframes as anchors (writing back to `Ref Frame` only if all selected shapes share the same keys). With no selection, the top shape in the curve list is targeted. VCR buttons and `Fix / Clean Tangents` follow the same rules.

---

## 3. Compatibility & File Structure

| Item | Specification |
| :--- | :--- |
| **Nuke Versions** | Foundry Nuke 13.0 – 17.x+ *(All versions supported)* |
| **UI Integration** | 100% Native Roto Node (Tab 1 Tracker Controls) |
| **Operating System** | Windows 10/11, Linux (Rocky, CentOS, Ubuntu) |
| **GPU** | NVIDIA CUDA-capable GPU |
| **AI Runtime** | Python 3.10 – 3.12, PyTorch 2.0+ (CUDA) |

```text
~/.nuke/AutoRoto/
├── init.py                        # Registers AutoRoto on sys.path
├── menu.py                        # Registers toolbar (Draw/Nodes) and Ctrl+Alt+R shortcut (no top menu bar)
├── nuke_bridge.py                 # Nuke Roto control, Tab 1 layout, tangent restoration, and keyframe baking
├── tracker_core.py                # CoTracker 3 GPU tracker (100-frame chunks, streaming, downscaling)
├── test_cotracker_backend.py      # CoTracker backend unit tests
├── test_tangent_math.py           # Tangent math and Key Step unit tests
├── test_multi_ref_tracking.py     # Multi-reference bidirectional blending and keyframe preservation tests
├── test_nuke_bridge_callbacks.py  # Bridge callbacks and reference validation tests
├── test_shape_selection.py        # Selected-shape targeting and independent per-shape tracking tests
├── README.md                      # User manual
└── SCRIPT_STRUCTURE.md            # Detailed architecture and internal structure docs
```

---

## 4. Headless Testing

Validate the CoTracker backend, tangent math, multi-reference blending, bridge callbacks, and selected-shape targeting without launching the Nuke GUI:

```bash
python -m unittest discover -v
```

<br>
<p align="center">
  <a href="#top"><b>⬆️ Back to Top (맨 위로 이동)</b></a> &nbsp;|&nbsp; <a href="#korean"><b>🌐 한국어로 전환</b></a>
</p>
