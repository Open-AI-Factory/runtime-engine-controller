# GRP Runtime Engine Controller

학습 워크로드의 GPU 구성을 실행 중에 바꾸는 런타임 엔진 제어 프로토타입.
실행 요청을 받아 **ExecutionPlan 생성 → 배치 → 실행**하고, RuntimeAction이 오면
**Safe Point → checkpoint → 새 구성으로 재시작 → 재개**까지 처리한 결과를 status로 남긴다.

| 구성 요소 | 코드 |
|---|---|
| Execution Plan Controller | `intake.py` (요청 검증 → 요구사항 분석 → DP/TP/PP 결정 → ExecutionPlan 생성·등록), `planner.py`, `k8s/execution_plan.py` |
| Runtime Action Controller | `controller.py` (상태 전이), `operator.py` (kopf Watch · 순차 처리 · 중복 무시) |
| Adapter Manager · Registry | `adapters/__init__.py` (Capability 선언, Adapter 선택) |
| Runtime Adapter | `adapters/` mock · local(단일 프로세스) · torchrun(다중 프로세스) · k8s(rank Pod) |
| Runtime Agent | `agent/hook.py` (학습 코드에 붙이는 Hook), `agent/channel.py` (파일 · ConfigMap 통신) |
| 자원 풀 연계 | `pool.py` (후보 · snapshot 인터페이스 + Mock) |
| 서비스 복구 | `recovery/` (구성 요소 의존성 그래프 기반 부분 재구성) |
| CLI | `cli.py` (`run` / `plan`), `keticti.py` (`runtime agents/watch`, `plan create/request`, `quota set`) |

## 폴더 구성

| 폴더 | 내용 |
|---|---|
| `src/grp_runtime/` | 소스 코드 |
| `scenarios/runtime-action/` | RuntimeAction 시나리오 (축소 · 확장 · 기한 초과 · 재구성 실패 · 계획 불가) |
| `scenarios/execution-plan/` | ExecutionPlan 예제 |
| `scenarios/plan-request/` | 실행 요청(PlanRequest) 예제 — 정상 · GPU 지정 · 입력 오류 · 자원 부족 · 버전 불일치 |
| `scenarios/recovery/` | 서비스 정의 · 복구 요청 |
| `workloads/` | Hook을 적용한 DDP 학습 워크로드와 Profiling 값 |
| `experiments/` | 재구성 전후 학습 결과 비교 |
| `mock/` | 자원 풀 Mock snapshot |
| `deploy/` | CRD · RBAC, WSL · Kind 환경 구성, operator 실행, 시나리오 스크립트 |
| `tests/` | 자동 테스트 |

## 상태 전이

```
Accepted     request validated
Preparing    preparation=Ready       실행 계획 생성, Agent 연결·격리값 일치(ReadyToRun) 확인
Checkpoint   serviceConfig=Valid     Safe Point 도달, checkpoint 저장·크기·digest 검증
Reconfigure  restart=Running         새 구성으로 재시작 (mode=restart)
Succeeded    ready=true response=OK  재시작 후 Agent 재연결·격리값 일치·학습 진행 확인

Aborted      reason=… previous=…
  변경 전 실패 → previous=Kept     (InvalidSpec / InvalidAction / WorkloadNotFound / InsufficientResources /
                                    NoFeasiblePlan / AdapterRejected / PreparationTimeout / DuplicateRequest)
  변경 후 실패 → previous=Restored (ReconfigureFailed, 변경 전 구성으로 복구) / Lost (복구 실패)
```

`safePolicy.timeoutSeconds`는 변경 전 준비 기한(Preparing + Checkpoint)이다. 초과하면 `PreparationTimeout`으로 끝나고 기존 구성을 유지한다.

## 설치

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,k8s]" numpy
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU 사용 시
```

## 로컬 실행

```bash
python -m grp_runtime plan scenarios/runtime-action/ra-scaledown-mock.yaml        # 조합 후보와 선택 사유
python -m grp_runtime run  scenarios/runtime-action/ra-scaledown-mock.yaml        # 상태 전이 실행
python -m grp_runtime run  scenarios/runtime-action/ra-scaledown-local-gpu.yaml   # 실제 GPU
python -m grp_runtime run  scenarios/runtime-action/ra-scaledown-torchrun.yaml    # torchrun DDP (Linux/WSL)
python -m pytest -q
```

결과는 `runs/<이름>/status.yaml`(최종 status)과 `runs/<이름>/events.jsonl`(phase별 기록)에 남는다.

## Runtime Hook

학습 코드에 몇 줄만 추가하면 Safe Point · checkpoint · 재개를 지원한다 (`workloads/train_ddp.py` 참고).

```python
from grp_runtime.agent import RuntimeHook

hook = RuntimeHook()
start = hook.restore(model, optimizer)
for step in range(start, max_steps):
    ...
    if hook.step_end(step + 1, model, optimizer, loss=loss.item()):
        break
```

- rank 0이 요청을 읽고 `all_reduce`로 공유하므로 모든 rank가 같은 step에서 멈춘다. rank 0이 저장하고 `barrier` 후 종료한다
- 재구성은 torchrun을 새 world size로 다시 실행하고 checkpoint에서 재개한다
- checkpoint는 RuntimeAction별 폴더에 저장하고, 폴더마다 최근 `GRP_CKPT_KEEP`개(기본 3)만 유지한다

## planner 메모리 검사

```
rank당 메모리 = (가중치 + 기울기 + 옵티마이저) ÷ (TP×PP) + (전체 batch ÷ DP) × 샘플당 activation ÷ PP
검사: rank당 메모리 × 같은 물리 GPU를 쓰는 rank 수 ≤ 물리 GPU 메모리 × 0.9
```

메모리에 들어가는 조합 중 모델 분할(TP×PP)이 가장 작은 조합을 고르고, 분할이 필요하면 단일 노드 고속 연결(NVLink)에서는 TP를, 그 외에는 PP를 우선한다.

## Kubernetes (Kind 다중 노드)

```bash
bash deploy/setup-wsl.sh                  # Docker Engine · kubectl · kind 설치 (WSL Ubuntu)
bash deploy/kind-up.sh                    # 이미지 빌드 · 클러스터 생성 · CRD/RBAC 등록
bash deploy/operator.sh start             # operator 기동 (로그: bash deploy/operator.sh log)

kubectl apply -f scenarios/plan-request/pr-train-kind.yaml   # 실행 요청 → ExecutionPlan 생성 → 배치 → rank Pod
kubectl get planrequest
kubectl get executionplan

keticti runtime agents train              # Agent 연결 · 격리값 비교
keticti runtime watch                     # RuntimeAction 단계 진행
keticti plan request scenarios/plan-request/pr-train-kind.yaml   # 클러스터 없이 계획 생성만
```

`deploy/slide15-scenario.sh`는 `setup` / `resize` / `timeout` / `restore` / `dup` / `request` / `reset` 단계로 전체 흐름을 재현하고,
`deploy/capture-scenario.sh all`은 단계별 출력을 `runs/capture/<시각>/`에 저장한다.

## 서비스 복구

```
image ───────────────┐
volume ── model ─────┤
config ──────────────┼── runtime ── endpoint
gpu ─────────────────┘
```

배포 시 요소별 구성 정보를 `record.json`으로 기록해 둔다. 복구 요청이 오면 모든 요소를 진단하고,
이상이 있는 요소와 그 하위 의존 요소만 의존성 순서대로 재구성한다. 이상이 없으면 재구성하지 않는다(`NO_FAULT_FOUND`).

```
Received → Diagnosing → Reacquiring → Restarting → Verifying → Recovered
   ├─ 같은 requestId → Duplicate
   └─ 같은 서비스 복구 진행 중 → Merged
```

```bash
python -m grp_runtime svc-deploy    scenarios/recovery/svc-mlp-serve-local.yaml
python -m grp_runtime svc-fail      scenarios/recovery/svc-mlp-serve-local.yaml runtime
python -m grp_runtime recover       scenarios/recovery/rr-detector-runtime.yaml scenarios/recovery/rr-pipeline-runtime.yaml
python -m grp_runtime recover-bench scenarios/recovery/svc-mlp-serve-local.yaml
python -m grp_runtime svc-stop      scenarios/recovery/svc-mlp-serve-local.yaml
```
