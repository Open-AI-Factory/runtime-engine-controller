from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path

import yaml

from .drivers import LocalServiceDriver, MockServiceDriver
from .graph import ORDER
from .manager import RecoveryManager
from .models import RecoveryRequest, ServiceSpec


def _root():
    from ..cli import ROOT
    return ROOT


def _spec(path: str) -> ServiceSpec:
    spec = ServiceSpec.from_manifest(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    if spec.pool_snapshot and not Path(spec.pool_snapshot).is_absolute():
        spec.pool_snapshot = str((_root() / spec.pool_snapshot).resolve())
    spec.raw.setdefault("spec", {})["poolSnapshot"] = spec.pool_snapshot
    return spec


def _driver(spec, runs):
    return (LocalServiceDriver if spec.driver == "local" else MockServiceDriver)(spec, runs)


def cmd_deploy(a):
    rec = RecoveryManager(Path(a.runs_dir)).deploy(_spec(a.service))
    print(f"\ndeployed {rec['service']} rev{rec['revision']} - full build {rec['deploySeconds']:.2f}s")
    for s in rec["deploySteps"]:
        print(f"  {s['component']:<8} {s['seconds']:>7.2f}s  {_brief(rec['components'][s['component']])}")
    return 0


def _brief(c: dict) -> str:
    keys = ("ref", "slots", "physicalGpu", "params", "sizeMb", "fingerprint", "pid", "url", "sha")
    return ", ".join(f"{k}={c[k]}" for k in keys if k in c)


def cmd_status(a):
    spec, m = _spec(a.service), RecoveryManager(Path(a.runs_dir))
    rec = m.load_record(spec.key)
    if not rec:
        print("no service record (run svc-deploy first)")
        return 1
    drv = _driver(spec, Path(a.runs_dir))
    print(f"{spec.key} rev{rec['revision']} (recorded {rec['savedAt']})")
    for k in ORDER:
        ok, msg = drv.probe(k, rec)
        print(f"  {'O' if ok else 'X'} {k:<8} {msg}")
    return 0


def cmd_fail(a):
    spec, m = _spec(a.service), RecoveryManager(Path(a.runs_dir))
    print("fault injected:", _driver(spec, Path(a.runs_dir)).inject(a.component, m.load_record(spec.key)))
    return 0


def cmd_stop(a):
    spec, m = _spec(a.service), RecoveryManager(Path(a.runs_dir))
    rec, drv = m.load_record(spec.key), _driver(spec, Path(a.runs_dir))
    if rec:
        drv.stop("endpoint", rec)
        drv.stop("runtime", rec)
    drv.clear_faults()
    print("service stopped:", spec.key)
    return 0


def cmd_recover(a):
    m = RecoveryManager(Path(a.runs_dir))
    reqs = [RecoveryRequest.from_manifest(yaml.safe_load(Path(p).read_text(encoding="utf-8")))
            for p in a.requests]
    results = [None] * len(reqs)

    def go(i, r):
        results[i] = m.handle(r)
    threads = [threading.Thread(target=go, args=(i, r)) for i, r in enumerate(reqs)]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join()
    out = Path(a.runs_dir) / "recovery"
    print(f"\n{'requestId':<26} {'source':<22} {'phase':<10} {'reason':<22} rebuilt / reused")
    for req, st in zip(reqs, results):
        (out / f"{req.name}.yaml").write_text(
            yaml.safe_dump(st.to_dict(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        print(f"{st.request_id:<26} {st.source:<22} {st.phase.value:<10} {st.reason or '-':<22} "
              f"{st.rebuilt or '-'} / {st.reused or '-'}")
    print(f"-> saved: {out}")
    return 0 if all(s.phase.value in ("Recovered", "Merged", "Duplicate") for s in results) else 1


def cmd_bench(a):
    spec = _spec(a.service)
    runs = Path(a.runs_dir)
    m = RecoveryManager(runs)
    print("== initial deploy")
    rec = m.deploy(spec)
    rows = [("(initial deploy)", "-", "full", ORDER, rec["deploySeconds"])]
    drv = _driver(spec, runs)
    stamp = datetime.now().strftime("%H%M%S")
    for comp, mode in [("endpoint", "Partial"), ("runtime", "Partial"), ("gpu", "Partial"),
                       ("model", "Partial"), ("runtime", "Full")]:
        print(f"\n== fault injected: {comp} / recovery mode={mode}")
        print("  ", drv.inject(comp, m.load_record(spec.key)))
        st = m.handle(RecoveryRequest(f"bench-{comp}-{mode}", f"bench-{stamp}-{comp}-{mode}",
                                      "bench", spec.key, comp, "bench injection", mode))
        rows.append((comp, mode, st.phase.value, st.rebuilt, st.total_seconds))
        if comp == "gpu":
            drv.clear_faults()
    lines = ["| Faulty component | Recovery mode | Result | Rebuilt components | Time (s) |", "|---|---|---|---|---:|"]
    for comp, mode, res, rebuilt, sec in rows:
        lines.append(f"| {comp} | {mode} | {res} | {', '.join(rebuilt)} | {sec:.2f} |")
    table = "\n".join(lines)
    print("\n" + table)
    out = runs / "recovery" / f"bench-{datetime.now():%Y%m%d-%H%M%S}.md"
    gpu = rec["components"]["gpu"].get("physicalGpu", "-")
    model = rec["components"]["model"]
    out.write_text(f"# Recovery time measurement ({datetime.now():%Y-%m-%d %H:%M})\n\n"
                   f"- service: {spec.key}, GPU: {gpu}, model: {model.get('params')} params "
                   f"({model.get('sizeMb')}MB fp16)\n- image component reuses the local environment (mock)\n\n{table}\n",
                   encoding="utf-8")
    print(f"-> saved: {out}")
    if not a.keep:
        cmd_stop(a)
    return 0


def add_parsers(sub) -> None:
    def p(name, fn, help, *args):
        sp = sub.add_parser(name, help=help)
        for x in args:
            sp.add_argument(x, **({"nargs": "+"} if x == "requests" else {}))
        sp.set_defaults(fn=fn)
        return sp
    p("svc-deploy", cmd_deploy, "initial service deploy + service record", "service")
    p("svc-status", cmd_status, "diagnose each service component", "service")
    p("svc-fail", cmd_fail, "inject a fault (demo)", "service", "component")
    p("svc-stop", cmd_stop, "stop the service", "service")
    p("recover", cmd_recover, "process recovery requests (concurrently if several)", "requests")
    b = p("recover-bench", cmd_bench, "measure recovery time per fault type", "service")
    b.add_argument("--keep", action="store_true", help="keep the service running afterwards")
