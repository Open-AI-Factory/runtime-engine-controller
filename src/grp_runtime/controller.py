from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from .adapters import AdapterError, select_adapter
from .models import ACTIONS, RUNTIMES, Phase, Previous, RuntimeActionSpec, RuntimeActionStatus, now
from .planner import build_plan, enumerate_configs, evaluate
from .pool import MockPoolClient

log = logging.getLogger("grp_runtime")


def validate(spec: RuntimeActionSpec) -> list[str]:
    errs = []
    if not spec.workload_ref.name:
        errs.append("workloadRef.name missing")
    if spec.runtime not in RUNTIMES:
        errs.append(f"runtime '{spec.runtime}' not supported {RUNTIMES}")
    if spec.action not in ACTIONS:
        errs.append(f"action '{spec.action}' not supported {ACTIONS}")
    if spec.target_gpu_count < 1:
        errs.append("target.gpuCount must be >= 1")
    if spec.safe_policy.mode not in ("WaitSafePoint", "Immediate"):
        errs.append(f"safePolicy.mode '{spec.safe_policy.mode}' not supported")
    if spec.mode != "restart":
        errs.append(f"mode '{spec.mode}' not supported (only restart)")
    if not spec.pool_snapshot:
        errs.append("poolSnapshot missing (mock snapshot required until resource pool integration)")
    return errs


class RuntimeActionController:
    def __init__(self, runs_root: Path, on_transition: Callable[[RuntimeActionStatus], None] | None = None,
                 poll_interval: float = 0.1):
        self.runs_root = runs_root
        self.on_transition = on_transition
        self.poll = poll_interval

    def _enter(self, st: RuntimeActionStatus, spec, phase: Phase, result: str, msg: str = "",
               reason: str = "") -> None:
        prev = st.phase.value if st.conditions else "-"
        st.phase, st.result, st.message = phase, result, msg
        if reason:
            st.reason = reason
        st.conditions.append({"phase": phase.value, "time": now(), "result": result,
                              "reason": reason, "message": msg})
        log.info("[%s] %s → %-11s %s%s", spec.name, prev, phase.value, result,
                 f" | {msg}" if msg else "")
        if self.on_transition:
            self.on_transition(st)

    def _abort(self, st, spec, reason: str, msg: str, previous: Previous = Previous.KEPT):
        st.previous = previous.value
        self._enter(st, spec, Phase.ABORTED, f"reason={reason} previous={previous.value}", msg, reason)
        return st

    def run(self, spec: RuntimeActionSpec, pool: MockPoolClient | None = None,
            history: list[dict] | None = None) -> RuntimeActionStatus:
        st = RuntimeActionStatus(mode=spec.mode, conditions=list(history or []))

        errs = validate(spec)
        if errs:
            self._enter(st, spec, Phase.ACCEPTED, "request invalid",
                        f"{spec.action} {spec.workload_ref.key} → GPU {spec.target_gpu_count}")
            return self._abort(st, spec, "InvalidSpec", "; ".join(errs))
        self._enter(st, spec, Phase.ACCEPTED, "request validated",
                    f"{spec.action} {spec.workload_ref.key} → GPU {spec.target_gpu_count}, mode={spec.mode}")
        pool = pool or MockPoolClient(spec.pool_snapshot)
        deadline = time.monotonic() + spec.safe_policy.timeout_seconds

        try:
            adapter = select_adapter(spec, pool, self.runs_root)
        except AdapterError as e:
            return self._abort(st, spec, "AdapterRejected", str(e))
        st.adapter = adapter.name
        self._enter(st, spec, Phase.PREPARING, "preparation=InProgress", f"adapter={adapter.name}")
        try:
            before = adapter.current_state(spec)
        except AdapterError as e:
            return self._abort(st, spec, "WorkloadNotFound", str(e))
        st.before = before.to_dict()
        cur = len(before.gpu_ids)
        if spec.action == "ScaleDown" and spec.target_gpu_count >= cur:
            return self._abort(st, spec, "InvalidAction",
                               f"ScaleDown but target GPUs {spec.target_gpu_count} >= current {cur}")
        if spec.action == "ScaleUp" and spec.target_gpu_count <= cur:
            return self._abort(st, spec, "InvalidAction",
                               f"ScaleUp but target GPUs {spec.target_gpu_count} <= current {cur}")
        if spec.action == "Resize" and spec.target_gpu_count == cur:
            return self._abort(st, spec, "InvalidAction", f"target GPUs {cur} = current GPUs, nothing to change")

        cs = pool.get_candidates(spec.workload_ref.key, before.gpu_ids, spec.target_gpu_count)
        if not cs.candidates:
            return self._abort(st, spec, "InsufficientResources",
                               f"no candidate for {spec.target_gpu_count} GPU(s) in resource pool (snapshot {cs.snapshot_version})")
        plan, rejected = None, []
        shard = spec.runtime == "fsdp"
        for cand in cs.candidates:
            plan = build_plan(spec.target_gpu_count, spec.profiling, cand, adapter.name, shard)
            if plan:
                plan.reason = f"[candidate#{cand.rank}: {cand.reason}] {plan.reason}"
                break
            notes = [f"DP{dp}/TP{tp}/PP{pp}: {evaluate(dp, tp, pp, spec.profiling, cand, shard)[1]}"
                     for dp, tp, pp in enumerate_configs(spec.target_gpu_count)]
            rejected.append(f"candidate#{cand.rank}({'; '.join(notes)})")
        if plan is None:
            return self._abort(st, spec, "NoFeasiblePlan",
                               f"no feasible DP/TP/PP combination: {' / '.join(rejected)}")
        st.execution_plan = {"snapshotVersion": cs.snapshot_version, **plan.to_dict()}
        try:
            adapter.prepare(spec, plan)
        except AdapterError as e:
            return self._abort(st, spec, "AdapterRejected", str(e))
        log.info("[%s]   ExecutionPlan: %s on %s (%s) — %s", spec.name, plan.parallel,
                 plan.gpu_ids, plan.interconnect, plan.reason)
        while True:
            ok, detail = adapter.preparation_ready(spec, plan)
            if ok:
                break
            if time.monotonic() > deadline:
                return self._abort(st, spec, "PreparationTimeout",
                                   f"preparation not completed within {spec.safe_policy.timeout_seconds:.0f}s: {detail}")
            time.sleep(self.poll)
        self._enter(st, spec, Phase.PREPARING, "preparation=Ready",
                    f"plan {before.parallel} → {plan.parallel} on {plan.gpu_ids}")

        if spec.safe_policy.mode == "Immediate":
            self._enter(st, spec, Phase.CHECKPOINT, "checkpoint=Skipped", "safePolicy=Immediate")
        else:
            self._enter(st, spec, Phase.CHECKPOINT, "checkpoint=Requested", "waiting for Safe Point")
            adapter.request_safe_point(spec)
            t0 = time.monotonic()
            while not adapter.safe_point_ready(spec):
                if time.monotonic() > deadline:
                    adapter.cancel_safe_point(spec)
                    return self._abort(st, spec, "PreparationTimeout",
                                       f"Safe Point not reached within {spec.safe_policy.timeout_seconds:.0f}s, "
                                       "keeping previous configuration")
                time.sleep(self.poll)
            st.safe_point_ready = True
            ok, detail = adapter.verify_checkpoint(spec)
            if not ok:
                adapter.cancel_safe_point(spec)
                return self._abort(st, spec, "CheckpointInvalid", detail)
            self._enter(st, spec, Phase.CHECKPOINT, detail,
                        f"Safe Point reached in {time.monotonic() - t0:.2f}s")

        self._enter(st, spec, Phase.RECONFIGURE, "restart=InProgress",
                    f"{before.parallel} / GPU {cur} → {plan.parallel} / GPU {plan.parallel.gpus}")
        try:
            after = adapter.reconfigure(spec, plan)
            self._enter(st, spec, Phase.RECONFIGURE, "restart=Running")
            ok, detail = adapter.verify_ready(spec)
            if not ok:
                raise AdapterError(f"response check after restart failed: {detail}")
        except AdapterError as e:
            try:
                adapter.rollback(spec, before)
                prev, rb = Previous.RESTORED, "restored to previous configuration"
            except AdapterError as e2:
                prev, rb = Previous.LOST, f"restore failed: {e2}"
            adapter.finalize(spec)
            return self._abort(st, spec, "ReconfigureFailed", f"{e} / {rb}", prev)
        st.after = after.to_dict()

        released = [g for g in before.gpu_ids if g not in plan.gpu_ids]
        added = [g for g in plan.gpu_ids if g not in before.gpu_ids]
        pool.release(spec.workload_ref.key, released)
        pool.assign(spec.workload_ref.key, added)
        st.released_gpus = released
        adapter.finalize(spec)
        self._enter(st, spec, Phase.SUCCEEDED, detail,
                    f"{after.parallel}, released GPUs {released or '-'}, added GPUs {added or '-'}")
        return st
