from __future__ import annotations

from typing import Any

from ..adapters import ADAPTER_NAMES, REGISTRY, capability_problems
from ..models import Profiling
from ..planner import evaluate
from ..pool import Candidate, MockPoolClient

PLACED_BY = "scheduler-mock"


def select_nodes(pool: MockPoolClient, nodes: int, gpus_per_node: int, workload_key: str) -> list[str]:
    by_node: dict[str, int] = {}
    mine: set[str] = set()
    for g in pool.gpus.values():
        if g.allocated_to in (None, workload_key):
            by_node[g.node] = by_node.get(g.node, 0) + 1
        if g.allocated_to == workload_key:
            mine.add(g.node)
    ok = sorted((n for n, c in by_node.items() if c >= gpus_per_node), key=lambda n: (n not in mine, n))
    return ok[:nodes]


def pool_topology(pool: MockPoolClient, gpus_per_node: int, workload_key: str,
                  only_nodes: list[str] | None = None) -> dict[str, Any]:
    usable = [g for g in pool.gpus.values() if g.allocated_to in (None, workload_key)
              and (only_nodes is None or g.node in only_nodes)]
    per_node: dict[str, int] = {}
    for g in usable:
        per_node[g.node] = per_node.get(g.node, 0) + 1
    eligible = sorted(n for n, c in per_node.items() if c >= gpus_per_node)
    gs = [g for g in usable if g.node in eligible]
    links = sorted({pool.interconnect.get(n, "unknown") for n in eligible})
    share: dict[str, int] = {}
    for g in gs:
        share[g.physical] = share.get(g.physical, 0) + 1
    return {"usableGpusPerNode": per_node, "eligibleNodes": eligible,
            "memoryGb": min((g.memory_gb for g in gs), default=None),
            "interconnect": links[0] if len(links) == 1 else ("mixed" if links else None),
            "ranksPerPhysical": max(share.values(), default=1)}


def _spec(ep: dict[str, Any]):
    spec = ep.get("spec", {})
    par = spec.get("parallelism", {}) or {}
    pl = spec.get("placement", {}) or {}
    key = f"{ep['metadata'].get('namespace', 'default')}/{spec.get('workloadRef', '')}"
    return (spec, int(par.get("dp", 1)), int(par.get("tp", 1)), int(par.get("pp", 1)),
            int(pl.get("nodes", 1)), int(pl.get("gpusPerNode", 1)), key)


def check_plan(ep: dict[str, Any], pool: MockPoolClient) -> tuple[list[str], dict | None]:
    spec, dp, tp, pp, nodes, gpn, key = _spec(ep)
    runtime = spec.get("runtime", "")
    problems = capability_problems("k8s", runtime, tp, pp)
    if dp * tp * pp != nodes * gpn:
        problems.append(f"DP×TP×PP={dp * tp * pp} ≠ nodes×gpusPerNode={nodes * gpn}")
    if tp > gpn:
        problems.append(f"TP={tp} > gpusPerNode={gpn} (TP requires intra-node communication)")
    topo = pool_topology(pool, gpn, key)
    estimate = None
    if topo["memoryGb"] is not None:
        cand = Candidate([], [f"node{i}" for i in range(nodes)], spec.get("interconnect", "Network"), gpn,
                         topo["memoryGb"], ranks_per_physical=topo["ranksPerPhysical"])
        ok, why, per_rank = evaluate(dp, tp, pp, Profiling.from_dict(spec.get("profiling")), cand,
                                     shard_dp=runtime == "fsdp")
        estimate = {"perRankGb": round(per_rank, 3), "note": why}
        if not ok:
            problems.append(f"insufficient memory: {why}")
    return problems, estimate


def place(ep: dict[str, Any], pool: MockPoolClient) -> dict[str, Any]:
    spec, dp, tp, pp, nodes, gpn, key = _spec(ep)
    world = dp * tp * pp
    pinned = (spec.get("planning") or {}).get("pinnedNodes")
    launched = bool((ep.get("status") or {}).get("generation"))
    if pinned and not launched:
        cap = pool_topology(pool, gpn, key, only_nodes=list(pinned))["usableGpusPerNode"]
        short = {n: cap.get(n, 0) for n in pinned if cap.get(n, 0) < gpn}
        chosen = list(pinned) if not short and len(pinned) == nodes else []
        reason = (f"pinned node(s) lack {gpn} GPU(s) per node in pool {pool.version}: {short}" if short else
                  f"pinnedNodes {list(pinned)} ≠ placement.nodes={nodes}" if len(pinned) != nodes else
                  "placed on pinned node(s) requested by PlanRequest")
    else:
        chosen = select_nodes(pool, nodes, gpn, key)
        reason = (f"{nodes} node(s) x {gpn} GPU selected from pool {pool.version} (workload's nodes first, then by name)"
                  if len(chosen) == nodes else
                  f"cannot secure {nodes} node(s) with gpusPerNode={gpn} from resource pool (snapshot {pool.version}) "
                  f"(available {chosen}, usable GPUs per node "
                  f"{pool_topology(pool, gpn, key)['usableGpusPerNode']})")
    if len(chosen) != nodes:
        return {"by": PLACED_BY, "state": "Unschedulable", "nodes": [], "rankNodes": [], "reason": reason}
    return {"by": PLACED_BY, "state": "Placed", "nodes": chosen,
            "rankNodes": [chosen[r // gpn] for r in range(world)], "reason": reason}


def validate(ep: dict[str, Any], pool: MockPoolClient, place_workers: bool = True) -> dict[str, Any]:
    spec = ep.get("spec", {})
    runtime = spec.get("runtime", "")
    problems, estimate = check_plan(ep, pool)
    out = {
        "validation": "Rejected" if problems else "Accepted",
        "reason": "; ".join(problems) if problems else "runtime, configuration and memory checks passed",
        "adapter": ADAPTER_NAMES.get(runtime, runtime),
        "adapterCapability": {"mode": "k8s", **REGISTRY["k8s"]},
        "memoryEstimate": estimate,
    }
    if place_workers:
        pl = place(ep, pool) if not problems else None
        out["placement"] = pl
        out["assignedNodes"] = pl["nodes"] if pl and pl["state"] == "Placed" else []
    return out
