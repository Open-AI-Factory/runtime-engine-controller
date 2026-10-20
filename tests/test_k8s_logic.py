import json
import time
from pathlib import Path
from types import SimpleNamespace

from grp_runtime.k8s.execution_plan import validate
from grp_runtime.k8s.workload import Workload
from grp_runtime.pool import MockPoolClient

ROOT = Path(__file__).resolve().parents[1]


class FakeCore:
    def __init__(self, data):
        self.data = data

    def read_namespaced_config_map(self, name, ns):
        return SimpleNamespace(data={k: json.dumps(v) for k, v in self.data.items()})


def agent(rank, node, cpu=1.0, mem=1024, world=2, age=0.0, hook="connected"):
    return {"hook": hook, "rank": rank, "world": world, "node": node, "updatedAt": time.time() - age,
            "isolation": {"cpuLimit": cpu, "memoryLimitMb": mem, "device": "cpu"}}


REQ = [{"rank": r, "node": n, "cpuLimit": 1.0, "memoryLimitMb": 1024, "device": "cpu", "world": 2}
       for r, n in enumerate(["w1", "w2"])]


def readiness(agents):
    return Workload("default", "train", core=FakeCore({f"agent-{a['rank']}": a for a in agents})).readiness(REQ)


def test_ready_when_all_connected_and_isolation_matches():
    r = readiness([agent(0, "w1"), agent(1, "w2")])
    assert (r["hook"], r["isolation"], r["state"], r["ranks"]) == ("connected", "requested-match", "ReadyToRun", [0, 1])


def test_isolation_mismatch_blocks_ready():
    r = readiness([agent(0, "w1"), agent(1, "w1", mem=512)])
    assert r["state"] == "NotReady" and r["isolation"] == "mismatch"
    assert any("rank1.node" in m for m in r["mismatches"]) and any("memoryLimitMb" in m for m in r["mismatches"])


def test_stale_heartbeat_is_not_connected():
    r = readiness([agent(0, "w1"), agent(1, "w2", age=30)])
    assert r["state"] == "NotReady" and r["staleRanks"] == [1] and r["hook"] == "partial"


def test_missing_agent_pending():
    r = readiness([agent(0, "w1")])
    assert r["isolation"] == "pending" and r["state"] == "NotReady"


def ep(**spec):
    base = {"workloadRef": "train", "runtime": "fsdp", "parallelism": {"dp": 2, "tp": 1, "pp": 1},
            "placement": {"nodes": 2, "gpusPerNode": 1}, "interconnect": "rdma",
            "profiling": {"weightsGb": 0.02, "gradientsGb": 0.02, "optimizerGb": 0.04,
                          "activationGbPerSample": 0.000035, "globalBatch": 1024}}
    return {"metadata": {"name": "train-plan", "namespace": "default"}, "spec": {**base, **spec}}


def pool():
    return MockPoolClient(ROOT / "mock" / "kind-snapshot.yaml")


def test_plan_accepted_assigns_two_nodes():
    r = validate(ep(), pool())
    assert r["validation"] == "Accepted" and r["adapter"] == "fsdp"
    assert r["assignedNodes"] == ["grp-worker", "grp-worker2"]


def test_plan_rejected_reasons():
    assert "does not support TP>1" in validate(ep(parallelism={"dp": 1, "tp": 2, "pp": 1}), pool())["reason"]
    assert "≠ nodes×gpusPerNode" in validate(ep(parallelism={"dp": 4, "tp": 1, "pp": 1}), pool())["reason"]
    r = validate(ep(parallelism={"dp": 3, "tp": 1, "pp": 1}, placement={"nodes": 3, "gpusPerNode": 1}), pool())
    assert r["validation"] == "Accepted" and r["placement"]["state"] == "Unschedulable"
    assert "cannot secure 3 node(s)" in r["placement"]["reason"] and r["assignedNodes"] == []
    assert "does not support runtime 'vllm'" in validate(ep(runtime="vllm"), pool())["reason"]


def test_fsdp_memory_grows_when_dp_shrinks():
    big = {"weightsGb": 1.0, "gradientsGb": 1.0, "optimizerGb": 2.0}
    assert validate(ep(profiling=big, parallelism={"dp": 2, "tp": 1, "pp": 1}), pool())["validation"] == "Rejected"
    r2 = validate(ep(profiling={"weightsGb": 0.5, "gradientsGb": 0.5, "optimizerGb": 1.0}), pool())
    r1 = validate(ep(profiling={"weightsGb": 0.5, "gradientsGb": 0.5, "optimizerGb": 1.0},
                     parallelism={"dp": 1, "tp": 1, "pp": 1}, placement={"nodes": 1, "gpusPerNode": 1}), pool())
    assert r2["validation"] == "Accepted" and r1["validation"] == "Rejected"
