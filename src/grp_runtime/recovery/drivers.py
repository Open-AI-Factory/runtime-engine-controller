from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from .models import ServiceSpec

SERVER = Path(__file__).with_name("server.py")


class DriverError(Exception):
    pass


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return code.value == 259
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def kill(pid: int | None) -> None:
    if not pid_alive(pid):
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True)
    else:
        os.kill(pid, 9)


def http_json(url: str, data: dict | None = None, timeout: float = 1.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def fingerprint(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open("rb") as f:
        for off in (0, max(0, size // 2 - 2**19), max(0, size - 2**20)):
            f.seek(off)
            h.update(f.read(2**20))
    return h.hexdigest()[:16]


def _read_json(p: Path, default):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


class LocalServiceDriver:
    def __init__(self, spec: ServiceSpec, runs_root: Path):
        self.spec = spec
        self.root = runs_root / "services"
        self.dir = self.root / f"{spec.namespace}_{spec.name}"
        self.vol = self.dir / "volume"
        self.cfg = self.dir / "runtime-config.json"
        self.health_file = self.root / "gpu-health.json"
        self.registry = self.root / "endpoints.json"
        self.dir.mkdir(parents=True, exist_ok=True)

    def probe(self, kind: str, rec: dict) -> tuple[bool, str]:
        c = rec["components"].get(kind, {})
        if kind == "image":
            return True, f"{c.get('ref')} (local environment, reused)"
        if kind == "volume":
            ok = (self.vol / "vocab.json").exists()
            return ok, "volume ok" if ok else "volume data (vocab.json) missing"
        if kind == "config":
            if not self.cfg.exists():
                return False, "runtime-config.json missing"
            h = hashlib.sha256(self.cfg.read_bytes()).hexdigest()[:16]
            return (h == c.get("sha"), "config matches" if h == c.get("sha") else
                    f"config change detected (sha {h} != recorded {c.get('sha')})")
        if kind == "gpu":
            bad = set(_read_json(self.health_file, {}).get("unhealthy", []))
            slots = c.get("slots", [])
            if not slots or any(s in bad for s in slots):
                return False, f"GPU slot fault: {[s for s in slots if s in bad] or 'unassigned'}"
            return True, f"GPU slot {slots} ok ({c.get('physicalGpu')})"
        if kind == "model":
            p = self.vol / "model.pt"
            if not p.exists():
                return False, "model file missing"
            fp = fingerprint(p)
            return (fp == c.get("fingerprint"), "model integrity ok" if fp == c.get("fingerprint")
                    else f"model integrity mismatch ({fp} != recorded {c.get('fingerprint')})")
        if kind == "runtime":
            if not pid_alive(c.get("pid")):
                return False, f"runtime process (pid {c.get('pid')}) exited"
            try:
                http_json(f"http://127.0.0.1:{self.spec.port}/health")
                return True, f"runtime pid {c.get('pid')} responding"
            except OSError as e:
                return False, f"runtime not responding: {e}"
        if kind == "endpoint":
            url = _read_json(self.registry, {}).get(self.spec.key)
            if not url:
                return False, "endpoint not registered"
            try:
                http_json(f"{url}/health")
                return True, f"{url} registered and responding"
            except OSError as e:
                return False, f"{url} not responding: {e}"
        raise DriverError(kind)

    def prepare(self, kind: str, rec: dict) -> dict[str, Any]:
        prev = rec["components"].get(kind, {})
        if kind == "image":
            return {"ref": f"local-venv python{sys.version.split()[0]} torch{version('torch')}",
                    "mode": "mock (local environment reused)"}
        if kind == "volume":
            shutil.rmtree(self.vol, ignore_errors=True)
            self.vol.mkdir(parents=True)
            (self.vol / "vocab.json").write_text(json.dumps({f"tok{i}": i for i in range(50_000)}))
            return {"path": str(self.vol)}
        if kind == "config":
            self.cfg.write_text(json.dumps({"executionPlan": {"dp": 1, "tp": 1, "pp": 1},
                                            "port": self.spec.port, "maxBatch": 8}, indent=1))
            return {"path": str(self.cfg), "sha": hashlib.sha256(self.cfg.read_bytes()).hexdigest()[:16]}
        if kind == "gpu":
            bad = set(_read_json(self.health_file, {}).get("unhealthy", []))
            snap = yaml.safe_load(Path(self.spec.pool_snapshot).read_text(encoding="utf-8"))
            slots = [f"{n['name']}:{g['index']}" for n in snap["nodes"] for g in n["gpus"]]
            ok = [s for s in slots if s not in bad]
            if not ok:
                raise DriverError("no healthy GPU slot")
            q = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True)
            if q.returncode:
                raise DriverError(f"nvidia-smi failed: {q.stderr.strip()}")
            return {"slots": [ok[0]], "physicalGpu": q.stdout.strip().splitlines()[0],
                    "previous": prev.get("slots"), "excluded": sorted(bad)}
        if kind == "model":
            import torch
            from .server import build_model
            m = self.spec.model
            torch.manual_seed(int(m.get("seed", 0)))
            model = build_model(int(m["hidden"]), int(m["layers"])).half()
            p = self.vol / "model.pt"
            torch.save(model.state_dict(), p)
            params = sum(x.numel() for x in model.parameters())
            return {"path": str(p), "params": params, "sizeMb": round(p.stat().st_size / 2**20),
                    "fingerprint": fingerprint(p)}
        if kind == "runtime":
            m = self.spec.model
            log = open(self.dir / f"runtime-{int(time.time())}.log", "w")
            proc = subprocess.Popen(
                [sys.executable, str(SERVER), "--model", str(self.vol / "model.pt"),
                 "--hidden", str(m["hidden"]), "--layers", str(m["layers"]),
                 "--port", str(self.spec.port), "--slot", rec["components"]["gpu"]["slots"][0],
                 "--fingerprint", rec["components"]["model"]["fingerprint"]],
                stdout=log, stderr=subprocess.STDOUT)
            end = time.monotonic() + 120
            while time.monotonic() < end:
                if proc.poll() is not None:
                    raise DriverError(f"runtime start failed (exit={proc.returncode}), log {log.name}")
                try:
                    h = http_json(f"http://127.0.0.1:{self.spec.port}/health")
                    return {"pid": proc.pid, "gpu": h["gpu"], "gpuMemAllocMb": h["gpuMemAllocMb"]}
                except OSError:
                    time.sleep(0.2)
            kill(proc.pid)
            raise DriverError("timed out waiting for runtime readiness")
        if kind == "endpoint":
            url = f"http://127.0.0.1:{self.spec.port}"
            http_json(f"{url}/health")
            reg = _read_json(self.registry, {})
            reg[self.spec.key] = url
            self.registry.write_text(json.dumps(reg, indent=1))
            return {"url": url}
        raise DriverError(kind)

    def stop(self, kind: str, rec: dict) -> None:
        if kind == "runtime":
            kill(rec["components"].get("runtime", {}).get("pid"))
        elif kind == "endpoint":
            reg = _read_json(self.registry, {})
            if reg.pop(self.spec.key, None):
                self.registry.write_text(json.dumps(reg, indent=1))

    def verify(self, rec: dict) -> dict[str, Any]:
        url = _read_json(self.registry, {}).get(self.spec.key)
        r = http_json(f"{url}/infer", {"batch": 4}, timeout=10)
        return {"testInference": r}

    def inject(self, kind: str, rec: dict) -> str:
        c = rec["components"]
        if kind == "runtime":
            kill(c["runtime"]["pid"])
            return f"runtime process pid {c['runtime']['pid']} killed"
        if kind == "gpu":
            h = _read_json(self.health_file, {"unhealthy": []})
            h["unhealthy"] = sorted(set(h["unhealthy"]) | set(c["gpu"]["slots"]))
            self.health_file.write_text(json.dumps(h))
            kill(c["runtime"]["pid"])
            return f"GPU slot {c['gpu']['slots']} marked faulty (simulated) + runtime killed"
        if kind == "model":
            p = self.vol / "model.pt"
            with p.open("r+b") as f:
                f.seek(p.stat().st_size // 2)
                f.write(os.urandom(4096))
            return "model.pt corrupted (4KB in the middle)"
        if kind == "endpoint":
            self.stop("endpoint", rec)
            return "endpoint deregistered"
        if kind == "config":
            self.cfg.write_text(self.cfg.read_text() + " ")
            return "runtime-config.json modified"
        if kind == "volume":
            (self.vol / "vocab.json").unlink()
            return "volume data (vocab.json) deleted"
        raise DriverError(kind)

    def clear_faults(self) -> None:
        self.health_file.unlink(missing_ok=True)


class MockServiceDriver:
    def __init__(self, spec: ServiceSpec, runs_root: Path, delay: float = 0.0):
        self.spec, self.delay = spec, delay
        self.broken: set[str] = set()
        self.prepared: list[str] = []
        self.stopped: list[str] = []

    def probe(self, kind, rec):
        return (kind not in self.broken, "ok" if kind not in self.broken else f"{kind} fault (mock)")

    def prepare(self, kind, rec):
        time.sleep(self.delay)
        self.broken.discard(kind)
        self.prepared.append(kind)
        return {"mock": True}

    def stop(self, kind, rec):
        self.stopped.append(kind)

    def verify(self, rec):
        return {"testInference": "mock"}

    def inject(self, kind, rec):
        self.broken.add(kind)
        return f"{kind} fault (mock)"

    def clear_faults(self):
        self.broken.clear()
