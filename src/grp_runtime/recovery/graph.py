from __future__ import annotations

DEPS: dict[str, list[str]] = {
    "image": [],
    "volume": [],
    "config": [],
    "gpu": [],
    "model": ["volume"],
    "runtime": ["image", "model", "gpu", "config"],
    "endpoint": ["runtime"],
}
ORDER = ["image", "volume", "config", "gpu", "model", "runtime", "endpoint"]


def affected(failed: set[str]) -> list[str]:
    out = set(failed)
    changed = True
    while changed:
        changed = False
        for comp, deps in DEPS.items():
            if comp not in out and any(d in out for d in deps):
                out.add(comp)
                changed = True
    return [c for c in ORDER if c in out]
