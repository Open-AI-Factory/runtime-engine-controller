from __future__ import annotations

import hashlib
import time
from pathlib import Path

from kubernetes import client, config

from ..k8s.workload import Workload, to_host_path
from ..models import ExecutionPlan, ParallelConfig, RuntimeActionSpec, WorkloadState, now
from .base import AdapterError, RuntimeAdapter

GROUP, VERSION, EP_PLURAL = "grp.keti.re.kr", "v1alpha1", "executionplans"
_loaded = False


def load_kube() -> None:
    global _loaded
    if not _loaded:
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        _loaded = True


def _digest(path: Path) -> str:
    h = hashlib.sha256(str(path.stat().st_size).encode())
    with path.open("rb") as f:
        h.update(f.read(2**20))
    return h.hexdigest()[:16]


def rank_nodes(nodes: list[str], world: int, gpus_per_node: int) -> list[str]:
    return [nodes[r // gpus_per_node] for r in range(world)]


class K8sAdapter(RuntimeAdapter):
    def __init__(self, spec: RuntimeActionSpec, pool):
        load_kube()
        self.name = spec.runtime
        self.pool = pool
        self.ns = spec.namespace
        self.ep_name = spec.execution_plan_ref or spec.adapter_options.get("executionPlan")
        if not self.ep_name:
            raise AdapterError("k8s mode requires executionPlanRef")
        self.co = client.CustomObjectsApi()
        self.wl = Workload(self.ns, spec.workload_ref.name)
        self._ready_samples: list[tuple[float, int]] = []

    def ep(self) -> dict:
        return self.co.get_namespaced_custom_object(GROUP, VERSION, self.ns, EP_PLURAL, self.ep_name)

    def patch_ep(self, spec: dict | None = None, status: dict | None = None) -> None:
        if spec:
            self.co.patch_namespaced_custom_object(GROUP, VERSION, self.ns, EP_PLURAL, self.ep_name,
                                                   {"spec": spec})
        if status:
            self.co.patch_namespaced_custom_object_status(GROUP, VERSION, self.ns, EP_PLURAL, self.ep_name,
                                                          {"status": status})

    def _requested(self) -> list[dict]:
        return (self.ep().get("status", {}).get("isolation", {}) or {}).get("requested", [])

    def current_state(self, spec):
        ep = self.ep()
        st, s = ep.get("status", {}), self.wl.state()
        if s.get("status") != "running":
            raise AdapterError(f"workload {self.wl.name} is not running (state={s.get('status')})")
        par = ep["spec"]["parallelism"]
        gpn = ep["spec"]["placement"].get("gpusPerNode", 1)
        ids = [f"{n}:{i}" for n in st.get("assignedNodes", []) for i in range(gpn)]
        self.pool.assign(spec.workload_ref.key, ids)
        return WorkloadState(ids, ParallelConfig(par["dp"], par["tp"], par["pp"]), {
            "runtime": spec.runtime, "mode": "k8s", "generation": st.get("generation"),
            "nodes": st.get("assignedNodes"), "worldSize": s.get("worldSize"), "step": s.get("step"),
            "loss": s.get("loss"), "rankPids": s.get("rankPids"), "rankPeakRssMb": s.get("rankPeakRssMb"),
            "perRankBatch": s.get("perRankBatch"), "globalBatch": s.get("globalBatch")})

    def prepare(self, spec, plan: ExecutionPlan):
        if plan.tp > 1 or plan.pp > 1:
            raise AdapterError(f"{plan.parallel}: k8s Adapter does not support TP/PP>1")

    def preparation_ready(self, spec, plan):
        r = self.wl.readiness(self._requested())
        if r["state"] == "ReadyToRun":
            return True, "preparation=Ready"
        return False, r["detail"]

    def request_safe_point(self, spec):
        self.wl.channel.write("control", {"request": "safepoint", "action": spec.name})

    def safe_point_ready(self, spec):
        return self.wl.state().get("status") == "safepoint"

    def cancel_safe_point(self, spec):
        self.wl.channel.delete("control")
        time.sleep(1.0)
        s = self.wl.state()
        if s.get("status") == "safepoint":
            ep = self.ep()
            par = ep["spec"]["parallelism"]
            self._relaunch(spec, ParallelConfig(par["dp"], par["tp"], par["pp"]),
                           ep["status"]["assignedNodes"], ep["spec"]["placement"].get("gpusPerNode", 1),
                           s["checkpoint"], f"{spec.name}-keep", "safepoint cancelled, resume previous")

    def verify_checkpoint(self, spec):
        s = self.wl.state()
        try:
            p = to_host_path(s["checkpoint"])
            ok = p.exists() and p.stat().st_size == s.get("checkpointBytes") \
                and _digest(p) == s.get("checkpointDigest")
        except (KeyError, ValueError, OSError) as e:
            return False, f"serviceConfig=Invalid ({e})"
        if not ok:
            return False, "serviceConfig=Invalid (checkpoint size/digest mismatch)"
        launch = self.ep()["spec"].get("launch", {})
        if not to_host_path(f"/work/{launch.get('script', 'workloads/train_ddp.py')}").exists():
            return False, "serviceConfig=Invalid (training script not found)"
        return True, "serviceConfig=Valid"

    def _relaunch(self, spec, par: ParallelConfig, nodes: list[str], gpn: int, resume: str, tag: str,
                  history: str) -> int:
        ep = self.ep()
        world = par.gpus
        gen = int(ep.get("status", {}).get("generation", 0)) + 1
        self.wl.teardown()
        requested = self.wl.launch(world, rank_nodes(nodes, world, gpn), ep["spec"]["runtime"],
                                   ep["spec"].get("launch", {}), gen, resume, tag)
        hist = list(ep.get("status", {}).get("history", []))
        hist.append({"time": now(), "action": spec.name, "event": history, "generation": gen,
                     "parallelism": {"dp": par.dp, "tp": par.tp, "pp": par.pp}, "nodes": nodes})
        self.patch_ep(spec={"parallelism": {"dp": par.dp, "tp": par.tp, "pp": par.pp},
                            "placement": {"nodes": len(nodes), "gpusPerNode": gpn}},
                      status={"generation": gen, "world": world, "assignedNodes": nodes,
                              "isolation": {"requested": requested}, "state": "Reconfiguring",
                              "history": hist[-10:]})
        return world

    def reconfigure(self, spec, plan: ExecutionPlan):
        sp = self.wl.state()
        self._sp = sp
        nodes = list(dict.fromkeys(g.split(":")[0] for g in plan.gpu_ids))
        resume = sp["checkpoint"]
        if spec.adapter_options.get("failAt") == "Reconfigure":
            resume += ".missing"
        world = self._relaunch(spec, plan.parallel, nodes, plan.gpus_per_node, resume, spec.name,
                               f"reconfigure {sp.get('worldSize')}→{plan.parallel.gpus} ranks")
        end = time.monotonic() + 180
        while time.monotonic() < end:
            s = self.wl.state()
            if s.get("status") == "running" and s.get("worldSize") == world and s.get("resumedStep") == sp["step"]:
                ws = WorkloadState(plan.gpu_ids, plan.parallel, {
                    "runtime": spec.runtime, "mode": "k8s", "nodes": nodes, "worldSize": world,
                    "step": s.get("step"), "loss": s.get("loss"), "rankPids": s.get("rankPids"),
                    "rankPeakRssMb": s.get("rankPeakRssMb"), "perRankBatch": s.get("perRankBatch"),
                    "globalBatch": s.get("globalBatch"), "resumedStep": s.get("resumedStep"),
                    "checkpointWorldSize": s.get("checkpointWorldSize"),
                    "checkpointStrategy": s.get("checkpointStrategy"), "checkpoint": sp["checkpoint"]})
                return ws
            failed = [p.metadata.name for p in self.wl.pods() if p.status.phase == "Failed"]
            if failed:
                raise AdapterError(f"restarted Pod failed: {failed}")
            time.sleep(1)
        raise AdapterError("timed out waiting for training to resume after restart (180s)")

    def verify_ready(self, spec):
        end = time.monotonic() + 60
        first = self.wl.state().get("step", 0)
        while time.monotonic() < end:
            r = self.wl.readiness(self._requested())
            s = self.wl.state()
            if r["state"] == "ReadyToRun" and s.get("status") == "running" and s.get("step", 0) > first:
                self.patch_ep(status={"state": "ReadyToRun"})
                return True, "ready=true response=OK"
            time.sleep(1)
        return False, f"ready=false ({self.wl.readiness(self._requested())['detail']})"

    def rollback(self, spec, before):
        sp = getattr(self, "_sp", None) or self.wl.state()
        if not sp.get("checkpoint"):
            raise AdapterError("no checkpoint to restore from")
        nodes = before.detail.get("nodes") or []
        gpn = max(1, len(before.gpu_ids) // max(1, len(nodes)))
        world = self._relaunch(spec, before.parallel, nodes, gpn, sp["checkpoint"], f"{spec.name}-rollback",
                               "rollback to previous")
        end = time.monotonic() + 180
        while time.monotonic() < end:
            s = self.wl.state()
            if s.get("status") == "running" and s.get("worldSize") == world:
                self.patch_ep(status={"state": "ReadyToRun"})
                return
            time.sleep(1)
        raise AdapterError("timed out waiting for resume after restore")
