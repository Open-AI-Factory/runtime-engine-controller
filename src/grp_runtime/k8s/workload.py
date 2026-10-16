from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from kubernetes import client

from ..agent.channel import ConfigMapChannel

LABEL = "grp.keti.re.kr/workload"
GEN = "grp.keti.re.kr/generation"
POD_ROOT = "/work"
HOST_ROOT = Path(__file__).resolve().parents[3]
HEARTBEAT_TIMEOUT = 10.0


def cm_name(workload: str) -> str:
    return f"grp-{workload}"


def svc_name(workload: str) -> str:
    return f"{workload}-ranks"


def to_host_path(pod_path: str) -> Path:
    return HOST_ROOT / Path(pod_path).relative_to(POD_ROOT)


def ckpt_dir(ns: str, workload: str, tag: str) -> str:
    return f"{POD_ROOT}/runs/k8s/{ns}_{workload}/ckpt/{tag}"


def _mem_mb(q: str) -> int:
    units = {"Ki": 1 / 1024, "Mi": 1, "Gi": 1024}
    for u, f in units.items():
        if q.endswith(u):
            return int(float(q[:-len(u)]) * f)
    return int(int(q) / 2**20)


class Workload:
    def __init__(self, ns: str, name: str, core: client.CoreV1Api | None = None):
        self.ns, self.name = ns, name
        self.core = core or client.CoreV1Api()
        self.channel = ConfigMapChannel(cm_name(name), ns, api=self.core)

    def pods(self) -> list:
        return self.core.list_namespaced_pod(self.ns, label_selector=f"{LABEL}={self.name}").items

    def state(self) -> dict:
        return self.channel.read("state") or {}

    def agents(self) -> list[dict]:
        data = self.channel.read_all()
        return sorted((v for k, v in data.items() if k.startswith("agent-")), key=lambda a: a["rank"])

    def ensure_base(self) -> None:
        try:
            self.core.read_namespaced_config_map(cm_name(self.name), self.ns)
        except client.ApiException as e:
            if e.status != 404:
                raise
            self.core.create_namespaced_config_map(self.ns, client.V1ConfigMap(
                metadata=client.V1ObjectMeta(name=cm_name(self.name), labels={LABEL: self.name}), data={}))
        try:
            self.core.read_namespaced_service(svc_name(self.name), self.ns)
        except client.ApiException as e:
            if e.status != 404:
                raise
            self.core.create_namespaced_service(self.ns, client.V1Service(
                metadata=client.V1ObjectMeta(name=svc_name(self.name), labels={LABEL: self.name}),
                spec=client.V1ServiceSpec(cluster_ip="None", publish_not_ready_addresses=True,
                                          selector={LABEL: self.name},
                                          ports=[client.V1ServicePort(name="dist", port=29500)])))

    def reset_channel(self) -> None:
        data = self.channel.read_all()
        if data:
            self.core.patch_namespaced_config_map(cm_name(self.name), self.ns,
                                                  {"data": {k: None for k in data}})

    def launch(self, world: int, nodes: list[str], runtime: str, launch: dict[str, Any],
               generation: int, resume: str | None, ckpt_tag: str) -> list[dict]:
        self.ensure_base()
        self.reset_channel()
        cpu, mem = str(launch.get("cpu", "1")), str(launch.get("memory", "1Gi"))
        script = launch.get("script", "workloads/train_ddp.py")
        args = " ".join(map(str, launch.get("args", [])))
        strategy = runtime if runtime in ("ddp", "fsdp") else "ddp"
        master = f"{self.name}-g{generation}-r0"
        requested = []
        for r in range(world):
            pod = f"{self.name}-g{generation}-r{r}"
            env = {"RANK": str(r), "WORLD_SIZE": str(world), "LOCAL_RANK": "0",
                   "MASTER_ADDR": f"{master}.{svc_name(self.name)}.{self.ns}.svc.cluster.local",
                   "MASTER_PORT": "29500", "GRP_CONFIGMAP": cm_name(self.name),
                   "GRP_CKPT_DIR": ckpt_dir(self.ns, self.name, ckpt_tag), "GRP_RESUME": resume or "",
                   "OMP_NUM_THREADS": cpu.rstrip("m") if not cpu.endswith("m") else "1",
                   "CUDA_VISIBLE_DEVICES": "",
                   "MALLOC_ARENA_MAX": "2"}
            envs = [client.V1EnvVar(name=k, value=v) for k, v in env.items()] + [
                client.V1EnvVar(name=k, value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(field_path=f)))
                for k, f in (("NODE_NAME", "spec.nodeName"), ("POD_NAME", "metadata.name"),
                             ("POD_NAMESPACE", "metadata.namespace"))]
            cmd = f"python {POD_ROOT}/{script} --strategy {strategy} {args}; exit $?"
            self.core.create_namespaced_pod(self.ns, client.V1Pod(
                metadata=client.V1ObjectMeta(name=pod, labels={LABEL: self.name, GEN: str(generation),
                                                               "grp.keti.re.kr/rank": str(r)}),
                spec=client.V1PodSpec(
                    hostname=pod, subdomain=svc_name(self.name), restart_policy="Never",
                    service_account_name="grp-agent",
                    node_selector={"kubernetes.io/hostname": nodes[r]},
                    containers=[client.V1Container(
                        name="trainer", image=launch.get("image", "grp-runtime-cpu:0.1"),
                        image_pull_policy="IfNotPresent", command=["sh", "-c", cmd], env=envs,
                        working_dir=POD_ROOT,
                        resources=client.V1ResourceRequirements(requests={"cpu": cpu, "memory": mem},
                                                                 limits={"cpu": cpu, "memory": mem}),
                        volume_mounts=[client.V1VolumeMount(name="work", mount_path=POD_ROOT)])],
                    volumes=[client.V1Volume(name="work", host_path=client.V1HostPathVolumeSource(path=POD_ROOT))],
                )))
            requested.append({"rank": r, "node": nodes[r], "cpuLimit": float(cpu.rstrip("m")) / (
                1000 if cpu.endswith("m") else 1), "memoryLimitMb": _mem_mb(mem), "device": "cpu",
                "world": world})
        return requested

    def teardown(self, timeout: float = 60) -> None:
        for p in self.pods():
            try:
                self.core.delete_namespaced_pod(p.metadata.name, self.ns, grace_period_seconds=0)
            except client.ApiException as e:
                if e.status != 404:
                    raise
        end = time.monotonic() + timeout
        while self.pods() and time.monotonic() < end:
            time.sleep(0.5)

    def readiness(self, requested: list[dict]) -> dict:
        agents = {a["rank"]: a for a in self.agents()}
        world = len(requested)
        nowt = time.time()
        connected = [r for r, a in agents.items() if a.get("hook") == "connected"
                     and nowt - a.get("updatedAt", 0) <= HEARTBEAT_TIMEOUT and a.get("world") == world]
        stale = [r for r, a in agents.items() if a.get("hook") == "connected"
                 and nowt - a.get("updatedAt", 0) > HEARTBEAT_TIMEOUT]
        mismatches = []
        for req in requested:
            a = agents.get(req["rank"])
            if not a:
                continue
            iso = a.get("isolation", {})
            actual = {"node": a.get("node"), "cpuLimit": iso.get("cpuLimit"),
                      "memoryLimitMb": iso.get("memoryLimitMb"), "device": iso.get("device"),
                      "world": a.get("world")}
            for k, v in actual.items():
                if req[k] != v:
                    mismatches.append(f"rank{req['rank']}.{k}: requested {req[k]} != actual {v}")
        all_connected = len(connected) == world
        if all_connected and not mismatches:
            isolation = "requested-match"
        elif mismatches:
            isolation = "mismatch"
        else:
            isolation = "pending"
        ready = all_connected and not mismatches
        hook = "connected" if all_connected else ("partial" if connected else "disconnected")
        return {"hook": hook, "ranks": sorted(connected), "world": world, "isolation": isolation,
                "state": "ReadyToRun" if ready else "NotReady", "mismatches": mismatches,
                "staleRanks": stale,
                "detail": None if ready else (f"unresponsive ranks {stale}" if stale else
                                              f"connected ranks {sorted(connected)}/{world}"
                                              + (f", mismatches {mismatches}" if mismatches else ""))}
