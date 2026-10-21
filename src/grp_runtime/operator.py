from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import kopf
from kubernetes import client, config

from .cli import ROOT
from .controller import RuntimeActionController
from .k8s.execution_plan import validate as validate_plan
from .k8s.workload import Workload
from .models import RuntimeActionSpec, now
from .pool import load_pool

GROUP, VERSION = "grp.keti.re.kr", "v1alpha1"
RA, EP, PR = "runtimeactions", "executionplans", "planrequests"
log = logging.getLogger("grp_runtime")

_reg_lock = threading.Lock()
_wl_locks: dict[str, threading.Lock] = {}
_inflight: dict[str, list[dict]] = {}


def _abs(path: str | None) -> str | None:
    return str((ROOT / path).resolve()) if path and not Path(path).is_absolute() else path


@kopf.on.startup()
def _startup(settings: kopf.OperatorSettings, **_):
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    settings.persistence.progress_storage = kopf.AnnotationsProgressStorage()
    settings.posting.level = logging.WARNING
    logging.getLogger("grp_runtime").setLevel(logging.INFO)


def _plan_request(body: dict, name: str) -> dict:
    from . import intake
    res = intake.from_resource(body, intake.ApiClient())
    st = intake.status_summary(res)
    w = st.get("workers") or {}
    log.info("[%s] PlanRequest %s%s | GPUs=%s workers=%s x %s selected=%s %s", name, res.result,
             f" reason={res.reason}" if res.reason else "", w.get("gpus"), w.get("nodes"), w.get("gpusPerNode"),
             st.get("selected"), res.message)
    return st


@kopf.on.create(GROUP, VERSION, PR)
async def pr_create(body, name, patch, **_):
    if (body.get("status") or {}).get("result"):
        return
    patch.status.update(await asyncio.to_thread(_plan_request, dict(body), name))


def _report_placement(body: dict, namespace: str, placement: dict) -> None:
    ref = (body["spec"].get("planning") or {}).get("requestRef") or {}
    if ref.get("kind") != "PlanRequest" or not ref.get("name"):
        return
    try:
        client.CustomObjectsApi().patch_namespaced_custom_object_status(
            GROUP, VERSION, ref.get("namespace", namespace), PR, ref["name"], {"status": {"placement": placement}})
    except Exception as e:
        log.warning("[%s] cannot report placement to PlanRequest %s: %s", body["metadata"]["name"], ref["name"], e)


def _validate_and_maybe_launch(body: dict, namespace: str, launch: bool) -> dict:
    spec, name = body["spec"], body["metadata"]["name"]
    pool = load_pool(_abs(spec.get("poolSnapshot", "mock/kind-snapshot.yaml")))
    st = body.get("status") or {}
    first = launch and not st.get("generation")
    res = validate_plan(body, pool, place_workers=first)
    status = {**res, "validatedAt": now()}
    if res["validation"] == "Rejected":
        log.info("[%s] ExecutionPlan Rejected: %s", name, res["reason"])
        return status
    pl = res.get("placement")
    if not first or pl is None:
        return status
    pl["time"] = now()
    retry = st.get("state") == "Unschedulable"
    if pl["state"] != "Placed":
        status["state"] = "Unschedulable"
        if not retry:
            _report_placement(body, namespace, pl)
            log.info("[%s] ExecutionPlan Accepted -> Unschedulable (retry every 2s): %s", name, pl["reason"])
        return status
    _report_placement(body, namespace, pl)
    if spec.get("launch"):
        par = spec["parallelism"]
        world = par["dp"] * par["tp"] * par["pp"]
        nodes = res["assignedNodes"]
        wl = Workload(namespace, spec["workloadRef"])
        requested = wl.launch(world, pl["rankNodes"], spec["runtime"], spec["launch"], 1, None, "initial")
        status.update(generation=1, world=world, state="Launching", isolation={"requested": requested},
                      history=[{"time": now(), "event": "launch", "generation": 1,
                                "parallelism": par, "nodes": nodes}])
        log.info("[%s] ExecutionPlan Accepted -> placed by %s on %s -> launched %d rank Pod(s)",
                 name, pl["by"], nodes, world)
    return status


@kopf.on.create(GROUP, VERSION, EP)
async def ep_create(body, namespace, patch, **_):
    patch.status.update(await asyncio.to_thread(_validate_and_maybe_launch, dict(body), namespace, True))


@kopf.on.update(GROUP, VERSION, EP, field="spec")
async def ep_update(body, namespace, patch, **_):
    res = await asyncio.to_thread(_validate_and_maybe_launch, dict(body), namespace, False)
    patch.status.update(res)


def _ep_tick(body: dict, namespace: str) -> dict | None:
    st = body.get("status") or {}
    if not st.get("generation"):
        if st.get("validation") == "Accepted" and st.get("state") == "Unschedulable":
            upd = _validate_and_maybe_launch(body, namespace, True)
            if (upd.get("placement") or {}).get("state") == "Placed":
                return upd
        return None
    workload = body["spec"]["workloadRef"]
    wl = Workload(namespace, workload)
    r = wl.readiness((st.get("isolation") or {}).get("requested", []))
    with _reg_lock:
        busy = any(x["running"] for x in _inflight.get(f"{namespace}/{workload}", []))
    state = "Reconfiguring" if busy else r["state"]
    agents = {"hook": r["hook"], "ranks": r["ranks"], "world": r["world"], "isolation": r["isolation"],
              "mismatches": r["mismatches"], "staleRanks": r["staleRanks"]}
    if st.get("state") == state and st.get("agents") == agents:
        return None
    return {"state": state, "agents": agents}


@kopf.timer(GROUP, VERSION, EP, interval=2.0, idle=0)
async def ep_timer(body, namespace, patch, **_):
    upd = await asyncio.to_thread(_ep_tick, dict(body), namespace)
    if upd:
        patch.status.update(upd)


def _fill_from_plan(spec: RuntimeActionSpec, namespace: str) -> None:
    co = client.CustomObjectsApi()
    ep = co.get_namespaced_custom_object(GROUP, VERSION, namespace, EP, spec.execution_plan_ref)
    es = ep["spec"]
    spec.runtime = spec.runtime or es["runtime"]
    spec.workload_ref.name = es["workloadRef"]
    spec.workload_ref.namespace = namespace
    spec.adapter_mode = "k8s"
    spec.adapter_options = {**spec.adapter_options, "executionPlan": spec.execution_plan_ref}
    spec.pool_snapshot = _abs(es.get("poolSnapshot", "mock/kind-snapshot.yaml"))
    if not any(v for v in vars(spec.profiling).values() if v not in (None, 0.0, "training")):
        from .models import Profiling
        spec.profiling = Profiling.from_dict(es.get("profiling"))


def _run_action(spec: RuntimeActionSpec, name: str, namespace: str, logger) -> None:
    api = client.CustomObjectsApi()

    def patch(st):
        api.patch_namespaced_custom_object_status(GROUP, VERSION, namespace, RA, name, {"status": st.to_dict()})

    key = spec.workload_ref.key
    me = {"name": name, "action": spec.action, "target": spec.target_gpu_count, "running": False}
    with _reg_lock:
        queue = _inflight.setdefault(key, [])
        dup = next((x for x in queue if x["action"] == me["action"] and x["target"] == me["target"]), None)
        if dup is None:
            queue.append(me)
            ahead = [x["name"] for x in queue if x is not me]
        lock = _wl_locks.setdefault(key, threading.Lock())
    if dup is not None:
        msg = f"{dup['name']} with the same workload and target (GPUs {me['target']}) is running or queued"
        patch_body = {"phase": "Aborted", "mode": spec.mode, "result": "reason=DuplicateRequest previous=Kept",
                      "reason": "DuplicateRequest", "previous": "Kept", "message": msg,
                      "conditions": [{"phase": "Accepted", "time": now(), "result": "request validated",
                                      "reason": "", "message": msg},
                                     {"phase": "Aborted", "time": now(),
                                      "result": "reason=DuplicateRequest previous=Kept",
                                      "reason": "DuplicateRequest", "message": msg}]}
        api.patch_namespaced_custom_object_status(GROUP, VERSION, namespace, RA, name, {"status": patch_body})
        log.info("[%s] Aborted reason=DuplicateRequest previous=Kept | %s", name, msg)
        return
    history = []
    if ahead:
        history = [{"phase": "Accepted", "time": now(), "result": "queued", "reason": "",
                    "message": f"waiting for preceding action(s) {ahead}"}]
        api.patch_namespaced_custom_object_status(GROUP, VERSION, namespace, RA, name, {"status": {
            "phase": "Accepted", "mode": spec.mode, "result": "queued",
            "message": history[0]["message"], "conditions": history}})
        log.info("[%s] Accepted queued | preceding %s", name, ahead)
    try:
        with lock:
            with _reg_lock:
                me["running"] = True
            RuntimeActionController(ROOT / "runs", on_transition=patch, poll_interval=0.5).run(
                spec, history=history)
    finally:
        with _reg_lock:
            queue.remove(me)


@kopf.on.create(GROUP, VERSION, RA)
async def ra_create(body, name, namespace, logger, **_):
    if (body.get("status") or {}).get("phase"):
        return
    spec = RuntimeActionSpec.from_manifest(dict(body))
    spec.namespace = namespace
    if spec.execution_plan_ref:
        await asyncio.to_thread(_fill_from_plan, spec, namespace)
    spec.pool_snapshot = _abs(spec.pool_snapshot)
    await asyncio.to_thread(_run_action, spec, name, namespace, logger)
