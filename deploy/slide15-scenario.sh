#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
K=/root/venvs/grp/bin/keticti
OUT=runs/slide15
mkdir -p $OUT

wait_ra() {
  for _ in $(seq 1 120); do
    p=$(kubectl get ra "$1" -o jsonpath='{.status.phase}' 2>/dev/null)
    [ "$p" = Succeeded ] || [ "$p" = Aborted ] && return 0
    sleep 2
  done
}

watch_start() { $K runtime watch --timeout "${1:-200}" > "$OUT/watch.txt" 2>&1 & WPID=$!; sleep 2; }
watch_show() { sleep 3; kill $WPID 2>/dev/null; echo '$ keticti runtime watch'; cat "$OUT/watch.txt"; echo; }

case "${1:-}" in
  setup)
    kubectl apply -f scenarios/execution-plan/ep-train-plan.yaml >/dev/null
    for _ in $(seq 1 60); do
      [ "$(kubectl get executionplan train-plan -o jsonpath='{.status.state}')" = ReadyToRun ] && break; sleep 2; done
    bash deploy/show.sh plan | tee "$OUT/1-plan.txt"
    bash deploy/show.sh agents | tee "$OUT/2-agents.txt" ;;
  request)
    f=${2:-scenarios/plan-request/pr-train-kind.yaml}
    {
      echo "\$ kubectl apply -f $f"; kubectl apply -f "$f"
      for _ in $(seq 1 30); do [ -n "$(kubectl get planrequest train-plan -o jsonpath='{.status.result}')" ] && break; sleep 1; done
      sleep 2
      echo; echo '$ kubectl get planrequest'; kubectl get planrequest
      echo; echo '$ kubectl get planrequest train-plan -o yaml  (status)'
      kubectl get planrequest train-plan -o json | python3 -c '
import json, sys
s = json.load(sys.stdin).get("status", {})
for k in ("result", "reason", "gpuCount", "workers", "mode", "candidates", "selected", "decisionReason", "feedback", "executionPlan", "placement", "message"):
    v = s.get(k)
    if v in (None, "", [], {}): continue
    if isinstance(v, list):
        print(f"  {k}:"); [print(f"    - {x}") for x in v]
    else:
        print(f"  {k}: {v}")'
    } | tee "$OUT/0-plan-request.txt"
    [ "$(kubectl get planrequest train-plan -o jsonpath='{.status.result}')" = Registered ] || exit 1
    for _ in $(seq 1 60); do
      [ "$(kubectl get executionplan train-plan -o jsonpath='{.status.state}')" = ReadyToRun ] && break; sleep 2; done
    bash deploy/show.sh plan | tee "$OUT/1-plan.txt"
    bash deploy/show.sh agents | tee "$OUT/2-agents.txt" ;;
  resize)
    watch_start
    $K quota set train --gpus "$2" --deadline "${3:-60}" ${4:+--name $4}
    name=${4:-$(kubectl get ra -l grp.keti.re.kr/workload=train --sort-by=.metadata.creationTimestamp \
                -o jsonpath='{.items[-1:].metadata.name}')}
    wait_ra "$name"
    watch_show | tee "$OUT/3-$name.txt"
    sleep 4; bash deploy/show.sh agents | tee -a "$OUT/3-$name.txt"; bash deploy/show.sh pods ;;
  timeout)
    echo "# pausing rank ${4:-1} (simulating unresponsive rank)"; $K debug pause train --rank "${4:-1}"
    sleep 12
    watch_start
    $K quota set train --gpus "${2:-1}" --deadline "${3:-15}"
    name=$(kubectl get ra -l grp.keti.re.kr/workload=train --sort-by=.metadata.creationTimestamp \
           -o jsonpath='{.items[-1:].metadata.name}')
    wait_ra "$name"
    watch_show | tee "$OUT/5-$name.txt"
    echo "# resuming rank ${4:-1}"; $K debug resume train --rank "${4:-1}"
    sleep 8; bash deploy/show.sh agents | tee -a "$OUT/5-$name.txt"; bash deploy/show.sh pods ;;
  dup)
    watch_start 300
    $K quota set train --gpus 2 --name seq-a
    $K quota set train --gpus 2 --name seq-b
    $K quota set train --gpus 1 --name seq-c
    wait_ra seq-a; wait_ra seq-b; wait_ra seq-c
    watch_show | tee "$OUT/4-sequential-dup.txt"
    kubectl get ra seq-a seq-b seq-c ;;
  restore)
    watch_start 300
    $K quota set train --gpus "${2:-2}" --name "${3:-resize-03}" --inject-failure
    wait_ra "${3:-resize-03}"
    watch_show | tee "$OUT/6-${3:-resize-03}.txt"
    kubectl get ra "${3:-resize-03}" -o jsonpath='{.status.message}{"\n"}' | tee -a "$OUT/6-${3:-resize-03}.txt"
    sleep 4; bash deploy/show.sh agents | tee -a "$OUT/6-${3:-resize-03}.txt"; bash deploy/show.sh pods ;;
  full)
    wait_running() {
      for _ in $(seq 1 90); do
        [ "$(kubectl get executionplan train-plan -o jsonpath='{.status.state}')" = ReadyToRun ] && \
          kubectl get cm grp-train -o jsonpath='{.data.state}' 2>/dev/null | grep -q '"status": "running"' && return 0
        sleep 2
      done; echo "!! workload not running"; return 1; }
    bash "$0" reset >/dev/null
    kubectl apply -f scenarios/execution-plan/ep-train-plan.yaml >/dev/null
    wait_running || exit 1; echo "# workload running (2 ranks)"
    $K quota set train --gpus 1 --name resize-01; wait_ra resize-01
    wait_running || exit 1
    $K debug pause train --rank 0; echo "# waiting for heartbeat to expire"; sleep 13
    $K quota set train --gpus 2 --deadline 15 --name resize-02; wait_ra resize-02
    $K debug resume train --rank 0; sleep 3; wait_running || exit 1
    $K quota set train --gpus 2 --name resize-03 --inject-failure; wait_ra resize-03
    wait_running || exit 1
    $K quota set train --gpus 2 --name seq-a
    $K quota set train --gpus 2 --name seq-b
    $K quota set train --gpus 1 --name seq-c
    wait_ra seq-a; wait_ra seq-b; wait_ra seq-c
    echo
    kubectl get ra -o custom-columns='NAME:.metadata.name,PHASE:.status.phase,RESULT:.status.result,PLAN:.spec.executionPlanRef,GPUS:.spec.target.gpus' \
      | tee "$OUT/7-all-actions.txt" ;;
  reset)
    kubectl delete ra --all >/dev/null; kubectl delete executionplan train-plan --ignore-not-found >/dev/null
    kubectl delete planrequest --all --ignore-not-found >/dev/null 2>&1
    kubectl delete executionplan big-plan --ignore-not-found >/dev/null 2>&1
    kubectl delete pod,svc,cm -l grp.keti.re.kr/workload=train --ignore-not-found >/dev/null
    rm -rf $OUT/plans
    echo reset ;;
  *) sed -n '2,8p' "$0" ;;
esac
