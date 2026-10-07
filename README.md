# AutoRoto (v2.0) - Roto Node with Tracker Buttons & CoTracker GPU
> **Nuke 네이티브 Roto 노드 복제 + 트래커 VCR 버튼 탑재**: Nuke Roto 노드에 트래커 노드의 트래킹 버튼들을 그대로 탑재하고, Meta CoTracker 3 GPU 백엔드로 구동되는 혁신적인 로토 툴킷

---

## 📌 1. 핵심 개선 사항 및 아키텍처

1. **`rootLayer` API 오류 수정 완료**:
   - Nuke `_rotopaint.RotoKnob`에서 발생하는 `'_rotopaint.RotoKnob' object has no attribute 'root'` 오류를 해결했습니다.
   - Nuke Python API 공식 명세에 맞춰 `curves.rootLayer`로 참조를 정규화하여 모든 Nuke 버전(13, 14, 15, 16, 17)에서 에러 없이 완벽히 동작합니다.
2. **Roto 노드 자체에 트래커 버튼 탑재**:
   - 별도 외부 패널을 띄우지 않고도, **Nuke의 기본 Roto 노드 프로퍼티 창에 Tracker 노드와 동일한 VCR 트래킹 버튼(`|◀`, `◀`, `▶`, `▶|`, `🚀 Track Full Range`)을 직접 탑재**했습니다.
   - Nuke 뷰어에서 기본 Bezier / B-Spline 펜 도구로 점을 찍고, 프로퍼티 창에서 트래커 버튼만 누르면 CoTracker GPU(RTX 4080)가 즉시 구동되어 모든 제어점에 키프레임이 구워집니다.

---

## 📁 2. 파일 구성

```text
c:\Users\yido2\.nuke\AutoRoto\
├── nuke_bridge.py             # Roto 노드 복제, Tracker 스타일 버튼 탑재, rootLayer 연동 및 베이킹
├── tracker_core.py            # CoTracker 3 GPU 딥러닝 포인트 트래킹 백엔드 (PyTorch + RTX 4080 CUDA)
├── roto_ui.py                 # PySide Roto 전용 패널 (트리 뷰 및 세부 제어 지원)
├── main.py                    # 패널 런처
├── menu.py                    # Nuke 툴바, 노드 메뉴, 단축키 (Ctrl+Alt+R) 등록
├── init.py                    # 패키지 sys.path 초기화
├── test_cotracker_backend.py  # 단위 테스트 스위트 (100% 통과)
└── README.md                  # 설명서
```

---

## 🖥️ 3. 사용 방법 (Nuke)

### 방법 A: AutoRoto 노드 신규 생성 (권장)
1. **노드 생성**:
   - 노드 그래프에서 `Tab` 키를 누르고 **`AutoRoto`** 입력 (또는 단축키 **`Ctrl+Alt+R`**, 또는 노드 툴바의 **AutoRoto** 아이콘 클릭).
   - 생성되는 노드는 **100% 네이티브 Roto 노드**이므로 뷰어의 모든 펜/베지어 툴이 그대로 동작합니다.
2. **트래커 탭 확인**:
   - 노드 프로퍼티 창의 **`AutoRoto Tracker`** 탭을 클릭하면 트래커 노드와 동일한 버튼들이 배치되어 있습니다:
     - `|◀` : 기준 프레임부터 시작 프레임까지 역방향 트래킹
     - `◀` : 1프레임 뒤로 스텝 트래킹
     - `▶` : 1프레임 앞으로 스텝 트래킹
     - `▶|` : 기준 프레임부터 끝 프레임까지 순방향 트래킹
     - **`🚀 Track Full Range`** : 지정한 전체 구간(Start ~ End)을 CoTracker GPU로 일괄 추적 후 자동 베이크
3. **로토 작성 후 트래킹**:
   - 기준 프레임에서 뷰어에 점을 찍어 로토 셰이프를 완성합니다.
   - **`[Set Current]`**로 기준 프레임을 지정하고 **`[🚀 Track Full Range]`**를 클릭하면 끝!

### 방법 B: 기존에 작업 중이던 Roto 노드에 트래커 버튼 추가
- 이미 셰이프를 따 둔 일반 Roto 노드가 있다면:
  - 해당 Roto 노드를 선택하고 Nuke 상단 메뉴: **`AutoRoto` ➔ `Convert Selected Roto to AutoRoto`** 클릭.
  - 기존 셰이프와 점을 그대로 유지한 채 트래커 버튼 탭이 즉시 추가됩니다.
