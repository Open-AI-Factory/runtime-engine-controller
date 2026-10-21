#!/usr/bin/env bash
cd "$(dirname "$0")/.."
K=/root/venvs/grp/bin/keticti
case "$1" in
  plan)
    echo '$ kubectl get executionplan train-plan -o yaml'
    kubectl get executionplan train-plan -o json | /root/venvs/grp/bin/python -c '
import json, sys, yaml
o = json.load(sys.stdin); s, st = o["spec"], o.get("status", {})
show = {"kind": o["kind"], "metadata": {"name": o["metadata"]["name"]},
        "spec": {k: s[k] for k in ("workloadRef", "runtime", "parallelism", "placement", "interconnect")},
        "status": {"adapter": st.get("adapter"), "validation": st.get("validation"), "state": st.get("state"),
                   "generation": st.get("generation"), "assignedNodes": st.get("assignedNodes"),
                   "memoryEstimate": (st.get("memoryEstimate") or {}).get("note")}}
print(yaml.safe_dump(show, allow_unicode=True, sort_keys=False, default_flow_style=None, width=200))' ;;
  agents)
    echo '$ keticti runtime agents train'
    $K runtime agents train ${2:-} ;;
  pods)
    kubectl get pods -o wide -l grp.keti.re.kr/workload=train ;;
esac
