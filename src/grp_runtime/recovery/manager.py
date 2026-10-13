from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..models import now
from .drivers import DriverError, LocalServiceDriver, MockServiceDriver
from .graph import ORDER, affected
from .models import RecoveryRequest, RecoveryStatus, RPhase, ServiceSpec

log = logging.getLogger("grp_runtime")


def default_driver_factory(runs_root: Path):
    cache: dict[str, Any] = {}

    def make(spec: ServiceSpec):
        if spec.key not in cache:
            cls = LocalServiceDriver if spec.driver == "local" else MockServiceDriver
            cache[spec.key] = cls(spec, runs_root)
        return cache[spec.key]
    return make


@dataclass
class _InFlight:
    request_id: str
    done: threading.Event = field(default_factory=threading.Event)
    result: RecoveryStatus | None = None


class RecoveryManager:
    def __init__(self, runs_root: Path, driver_factory: Callable[[ServiceSpec], Any] | None = None,
                 on_transition: Callable[[RecoveryStatus], None] | None = None):
        self.runs_root = runs_root
        self.driver_for = driver_factory or default_driver_factory(runs_root)
        self.on_transition = on_transition
        self.registry_path = runs_root / "recovery" / "registry.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._inflight: dict[str, _InFlight] = {}

    def record_path(self, key: str) -> Path:
        ns, name = key.split("/", 1)
        return self.runs_root / "services" / f"{ns}_{name}" / "record.json"

    def load_record(self, key: str) -> dict | None:
        p = self.record_path(key)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def save_record(self, rec: dict) -> None:
        rec["savedAt"] = now()
        p = self.record_path(rec["service"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")

    def deploy(self, spec: ServiceSpec) -> dict:
        drv = self.driver_for(spec)
        drv.clear_faults()
        old = self.load_record(spec.key)
        if old:
            for k in ("endpoint", "runtime"):
                drv.stop(k, old)
        rec = {"service": spec.key, "revision": 1, "spec": spec.raw, "components": {}, "deploySteps": []}
        for kind in ORDER:
            t0 = time.perf_counter()
            rec["components"][kind] = drv.prepare(kind, rec)
            sec = round(time.perf_counter() - t0, 3)
            rec["deploySteps"].append({"component": kind, "seconds": sec})
            log.info("[deploy %s] %-8s built %.2fs", spec.key, kind, sec)
        rec["deploySeconds"] = round(sum(s["seconds"] for s in rec["deploySteps"]), 3)
        self.save_record(rec)
        return rec

    def _registry(self) -> dict:
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _register(self, st: RecoveryStatus) -> None:
        reg = self._registry()
        reg[st.request_id] = {"service": st.service, "source": st.source, "phase": st.phase.value,
                              "reason": st.reason, "finishedAt": now(), "mergedInto": st.merged_into}
        self.registry_path.write_text(json.dumps(reg, indent=1, ensure_ascii=False), encoding="utf-8")

    def _enter(self, st: RecoveryStatus, phase: RPhase, msg: str, reason: str = "") -> None:
        st.phase, st.message = phase, msg
        if reason:
            st.reason = reason
        st.conditions.append({"phase": phase.value, "time": now(), "reason": reason, "message": msg})
        log.info("[%s] %s%s | %s", st.request_id, phase.value, f" ({reason})" if reason else "", msg)
        if self.on_transition:
            self.on_transition(st)

    def handle(self, req: RecoveryRequest) -> RecoveryStatus:
        st = RecoveryStatus(req.request_id, req.source, req.service_key)
        self._enter(st, RPhase.RECEIVED,
                    f"{req.source} -> {req.service_key} recovery request "
                    f"(symptom: {req.symptom_component or '-'} {req.symptom_detail}, mode={req.mode})")
        if not req.request_id:
            self._enter(st, RPhase.FAILED, "requestId missing", "INVALID_REQUEST")
            return st
        with self._lock:
            prev = self._registry().get(req.request_id)
            if prev:
                self._enter(st, RPhase.DUPLICATE,
                            f"request already processed - returning previous result without re-running "
                            f"(phase={prev['phase']}, {prev['finishedAt']})", "DUPLICATE_REQUEST")
                return st
            if any(f.request_id == req.request_id for f in self._inflight.values()):
                self._enter(st, RPhase.DUPLICATE, "same requestId in progress - not running again",
                            "DUPLICATE_REQUEST")
                return st
            inflight = self._inflight.get(req.service_key)
            if inflight is None:
                inflight = self._inflight[req.service_key] = _InFlight(req.request_id)
                primary = True
            else:
                primary = False
        if not primary:
            self._enter(st, RPhase.RECEIVED, f"recovery of the same service in progress ({inflight.request_id}) - waiting to merge")
            inflight.done.wait()
            r = inflight.result
            st.merged_into = inflight.request_id
            st.rebuilt, st.reused = r.rebuilt, r.reused
            self._enter(st, RPhase.MERGED,
                        f"sharing recovery result of {inflight.request_id}: {r.phase.value} {r.reason}".strip(),
                        "MERGED_INTO_INFLIGHT")
            with self._lock:
                self._register(st)
            return st
        try:
            self._recover(req, st)
        except Exception as e:
            self._enter(st, RPhase.FAILED, f"{type(e).__name__}: {e}", "INTERNAL_ERROR")
        finally:
            with self._lock:
                inflight.result = st
                self._register(st)
                self._inflight.pop(req.service_key, None)
            inflight.done.set()
        return st

    def _recover(self, req: RecoveryRequest, st: RecoveryStatus) -> None:
        t_start = time.perf_counter()
        rec = self.load_record(req.service_key)
        if rec is None:
            return self._enter(st, RPhase.FAILED, "service record not found", "SERVICE_RECORD_NOT_FOUND")
        spec = ServiceSpec.from_manifest(rec["spec"])
        drv = self.driver_for(spec)

        self._enter(st, RPhase.DIAGNOSING, f"loaded service record rev{rec['revision']}, diagnosing each component")
        failed = set()
        for kind in ORDER:
            ok, msg = drv.probe(kind, rec)
            st.diagnosis.append({"component": kind, "ok": ok, "message": msg})
            if not ok:
                failed.add(kind)
        if req.mode == "Full":
            target = list(ORDER)
        else:
            if not failed:
                st.reused = list(ORDER)
                st.total_seconds = round(time.perf_counter() - t_start, 3)
                return self._enter(st, RPhase.RECOVERED,
                                   f"no fault found - reported symptom ({req.symptom_component}) not reproduced, "
                                   "skipping rebuild", "NO_FAULT_FOUND")
            target = affected(failed)
        st.rebuilt = target
        st.reused = [k for k in ORDER if k not in target]
        log.info("[%s]   faulty %s -> rebuild %s / reuse %s", st.request_id,
                 sorted(failed, key=ORDER.index), target, st.reused)

        try:
            for kind in reversed(target):
                if kind in ("endpoint", "runtime"):
                    drv.stop(kind, rec)

            if "gpu" in target:
                self._enter(st, RPhase.REACQUIRING, "requesting GPU reacquisition (resource pool)")
                self._step(st, drv, rec, "gpu")
                g = rec["components"]["gpu"]
                self._enter(st, RPhase.REACQUIRING,
                            f"GPU reacquired: {g.get('previous')} -> {g.get('slots')} (excluded: {g.get('excluded')})")
            else:
                self._enter(st, RPhase.REACQUIRING, "GPU healthy - reusing current allocation")

            rest = [k for k in target if k != "gpu"]
            self._enter(st, RPhase.RESTARTING, f"rebuild order: {' -> '.join(rest) or '-'}")
            for kind in rest:
                self._step(st, drv, rec, kind)
        except DriverError as e:
            return self._enter(st, RPhase.FAILED, str(e), "RESTART_FAILED")

        self._enter(st, RPhase.VERIFYING, "re-diagnosing all components and sending a test request")
        bad = [(k, m) for k in ORDER for ok, m in [drv.probe(k, rec)] if not ok]
        if bad:
            return self._enter(st, RPhase.FAILED, f"faults after recovery: {bad}", "VERIFY_FAILED")
        try:
            v = drv.verify(rec)
        except OSError as e:
            return self._enter(st, RPhase.FAILED, f"test request failed: {e}", "VERIFY_FAILED")
        rec["revision"] += 1
        self.save_record(rec)
        st.total_seconds = round(time.perf_counter() - t_start, 3)
        self._enter(st, RPhase.RECOVERED,
                    f"rebuilt {st.rebuilt} / reused {st.reused}, total {st.total_seconds:.2f}s, "
                    f"test {v['testInference']}")

    def _step(self, st, drv, rec, kind) -> None:
        t0 = time.perf_counter()
        rec["components"][kind] = drv.prepare(kind, rec)
        sec = round(time.perf_counter() - t0, 3)
        st.steps.append({"component": kind, "action": "rebuild", "seconds": sec})
        log.info("[%s]   %-8s rebuilt %.2fs", st.request_id, kind, sec)
