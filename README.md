# AutoRoto (v2.1) - Roto Node with Tracker Buttons & CoTracker GPU
>
> **Nuke 네이티브 Roto 노드 복제 + 트래커 VCR 버튼 탑재**: Nuke Roto 노드에 트래커 노드의 트래킹 버튼들을 그대로 탑재하고, Meta CoTracker 3 GPU 백엔드로 구동되는 혁신적인 로토 툴킷

<https://github.com/user-attachments/assets/3fdd6cf2-98cc-49c6-ae54-dacb82a579e3>

### 🖥️ 지원 환경 및 호환 버전 (Compatibility)

- **지원 Nuke 버전**: **Foundry Nuke 13.0 ~ 17.x+** *(Nuke 13, 14, 15, 16, 17 전 버전 완벽 호환)*
- **UI 방식**: **100% 네이티브 Roto 노드 일체형** (Roto 1번 탭 직접 제어)
- **운영체제(OS)**: Windows 10/11, Linux (Rocky, CentOS, Ubuntu)
- **GPU 환경**: NVIDIA CUDA 지원 GPU (RTX 30xx, 40xx, RTX Ada, Quadro 등)
- **AI 런타임**: Python 3.10 ~ 3.12 (PyTorch 2.0+ with CUDA)

---

## 📌 1. 핵심 개선 사항 및 아키텍처

1. **`rootLayer` API 오류 수정 완료**:
   - Nuke `_rotopaint.RotoKnob`에서 발생하는 `'_rotopaint.RotoKnob' object has no attribute 'root'` 오류를 해결했습니다.
   - Nuke Python API 공식 명세에 맞춰 `curves.rootLayer`로 참조를 정규화하여 모든 Nuke 버전(13, 14, 15, 16, 17)에서 에러 없이 완벽히 동작합니다.
2. **Roto 노드 자체에 트래커 버튼 탑재 (1번 탭 일체형)**:
   - **Nuke의 기본 Roto 노드 프로퍼티 창 1번 탭에 Tracker 노드와 동일한 VCR 트래킹 버튼(`|◀`, `◀`, `▶`, `▶|`, `🚀 Track Full Range`)을 직접 탑재**했습니다.
   - Nuke 뷰어에서 기본 Bezier / B-Spline 펜 도구로 점을 찍고, 프로퍼티 창에서 트래커 버튼만 누르면 CoTracker GPU(RTX 4080)가 즉시 구동되어 모든 제어점에 키프레임이 구워집니다.

---

## 📥 2. 설치 방법 (Installation)

### 1) GitHub 저장소 다운로드 / 클론

Nuke의 사용자 플러그인 디렉터리(`~/.nuke/`)에 저장소를 클론하거나 압축 해제합니다:

```bash
cd ~/.nuke
git clone https://github.com/Kiwoo0413/AutoRoto.git
```

*(폴더명이 반드시 `AutoRoto`여야 합니다.)*

### 2) Nuke `init.py` 경로 등록

`~/.nuke/init.py` 파일(없으면 새로 생성)을 열고 아래 1줄을 추가합니다:

```python
import nuke
nuke.pluginAddPath('AutoRoto')
```

### 3) 외부 Python & AI 백엔드 환경 준비

AutoRoto는 Nuke 내장 파이썬과 충돌하지 않도록 외부 Python의 PyTorch + CUDA 환경을 사용합니다.

- **Python 버전**: 3.10 ~ 3.12 (Windows / Linux)
- **필수 패키지 설치**:

  ```bash
  # CUDA 지원 PyTorch 설치 (예: CUDA 12.4)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  pip install numpy Pillow
  
  # (선택) CoTracker 패키지 설치 (미설치 시 torch.hub를 통해 모델 자동 로드)
  pip install git+https://github.com/facebookresearch/co-tracker.git
  ```

- *참고*: 가상환경(Conda)이나 특정 경로의 Python을 사용하고 싶다면 시스템 환경 변수 `AUTOROTO_PYTHON`에 해당 `python.exe`의 절대 경로를 지정하시면 우선적으로 인식됩니다.

---

## 📁 3. 파일 구성

```text
~/.nuke/AutoRoto/
├── init.py                    # AutoRoto sys.path 등록
├── menu.py                    # 메뉴 바, 툴바(Draw/Nodes), 단축키(Ctrl+Alt+R) 등록
├── nuke_bridge.py             # Nuke Roto 노드 제어, Tab 1 배치, 베지에 탄젠트 복원 및 키프레임 베이킹
├── tracker_core.py            # CoTracker 3 GPU 딥러닝 트래커 (100프레임 청크, 실시간 스트리밍, 다운스케일)
├── test_cotracker_backend.py  # 단위 테스트 스위트
├── test_tangent_math.py       # 탄젠트 수학 및 키 스텝 검증 테스트
├── test_multi_ref_tracking.py # 멀티 레퍼런스 양방향 블렌딩 및 키프레임 보존 검증 테스트
├── README.md                  # 사용자 매뉴얼
└── SCRIPT_STRUCTURE.md        # 아키텍처 및 내부 구조 상세 문서
```

---

## 🖥️ 4. 사용 방법 (Nuke)

### 방법 A: AutoRoto 노드 신규 생성 (권장)

1. **노드 생성**:
   - 노드 그래프에서 `Tab` 키를 누르고 **`AutoRoto`** 입력 (또는 단축키 **`Ctrl+Alt+R`**, 또는 노드 툴바의 **AutoRoto** 아이콘 클릭).
   - 생성되는 노드는 **100% 네이티브 Roto 노드**이므로 뷰어의 모든 펜/베지어 툴이 그대로 동작합니다.
2. **트래커 탭 (Tab 1) 확인**:
   - 노드를 더블클릭하면 **`AutoRoto` 탭이 1번 탭으로 자동 배치**되어 나타납니다:
     - `|◀` : 기준 프레임부터 시작 프레임까지 역방향 트래킹
     - `◀` : 1스텝 뒤로 트래킹 (`Key Step` 간격 이동)
     - `▶` : 1스텝 앞으로 트래킹 (`Key Step` 간격 이동)
     - `▶|` : 기준 프레임부터 끝 프레임까지 순방향 트래킹
     - **`🚀 Track Full Range`** : 지정한 전체 구간을 CoTracker GPU로 일괄 추적 후 자동 베이크
3. **주요 파라미터**:
   - **`Ref Frame` / `Set Current`**: 셰이프를 직접 그린 기준 프레임 설정.
   - **`Key Step`**: 키프레임 생성 간격 (1 = 매 프레임, 2 = 2프레임마다, 5 = 5프레임마다 베이킹하여 편집 용이).
   - **`Tracking Res`**:
     - `720p (Fast / AI Optimized)` *(기본값)*: 3~5배 고속 추론 & 서브픽셀 원본 좌표 무손실 복원.
     - `960p (Balanced)`: 고화질 균형 모드.
     - `Full (Original / Slow)`: 원본 해상도 유지 모드.
   - **`Preserve Curvature / Tangents`**: 표면 변형 및 회전에 맞춰 베지에 핸들 곡률을 자연스럽게 보존 (원형 루프 왜곡 방지).
   - **`Fix / Clean Tangents`**: 꼬인 베지에 핸들이 있을 때 원클릭으로 정돈.
4. **로토 작성 후 트래킹**:
   - 기준 프레임에서 뷰어에 점을 찍어 로토 셰이프를 완성합니다.
   - **`[Set Current]`**로 기준 프레임을 지정하고 **`[🚀 Track Full Range]`**를 클릭하면 끝!
   - 실시간 진행률 바를 통해 청크별 진행 상황이 표시되며, 언제든 `Cancel`로 안전하게 즉시 중단할 수 있습니다.

### 방법 B: 기존에 작업 중이던 Roto 노드에 트래커 버튼 추가

- 이미 셰이프를 따 둔 일반 Roto 노드가 있다면:
  - 해당 Roto 노드를 선택하고 Nuke 상단 메뉴: **`AutoRoto` ➔ `Convert Selected Roto to AutoRoto`** 클릭.
  - 기존 셰이프와 점을 그대로 유지한 채 1번 탭에 AutoRoto 트래커 기능이 즉시 추가됩니다.

### 방법 C: 멀티 레퍼런스 트래킹 (Multi-Reference Tracking)

여러 프레임에서 로토 셰이프를 수정하거나 키프레임을 잡은 경우, 이를 복수 앵커(Multi-Reference)로 활용하여 사이 구간을 완벽하게 보간 추적할 수 있습니다:

1. **키프레임 감지 / 등록**:
   - **`[Detect Shape Keys]`**: 활성 로토 셰이프의 애니메이션 커브를 자동 스캔하여 아티스트가 직접 생성/수정한 모든 키프레임 번호(예: `1, 35, 70, 100`)를 자동 탐지하여 `Keyframes` 입력 필드에 등록합니다.
   - **`[+ Add Current]`**: 현재 플레이헤드 프레임을 멀티 레퍼런스 키 목록에 즉시 추가합니다.
   - **`[Clear]`**: 키프레임 목록을 초기화합니다.
2. **`[⚡ Track Multi-Reference]` 실행**:
   - 각 키프레임 구간(예: 1~35, 35~70, 70~100) 사이에서 **순방향(Forward) 및 역방향(Backward) CoTracker 3 양방향 추적**을 각각 수행합니다.
   - $C^1$ Smoothstep ($s(t) = 3t^2 - 2t^3$) 및 CoTracker Feature Visibility(가시성) 가중치를 결합하여 두 궤적을 무봉제(Seamless)로 블렌딩합니다.
   - **아티스트 키프레임 100% 보존**: 사용자가 직접 작업한 기준 키프레임은 일절 덮어쓰거나 지우지 않고 그대로 유지되며, 오직 사이 구간(In-between)의 프레임들만 최적의 트래킹 데이터로 채워집니다.
