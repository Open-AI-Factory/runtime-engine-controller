from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RPhase(str, Enum):
    RECEIVED = "Received"
    DIAGNOSING = "Diagnosing"
    REACQUIRING = "Reacquiring"
    RESTARTING = "Restarting"
    VERIFYING = "Verifying"
    RECOVERED = "Recovered"
    FAILED = "Failed"
    DUPLICATE = "Duplicate"
    MERGED = "Merged"


@dataclass
class ServiceSpec:
    name: str
    namespace: str
    driver: str
    port: int
    model: dict[str, Any]
    pool_snapshot: str | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.namespace}/{self.name}"

    @classmethod
    def from_manifest(cls, m: dict[str, Any]) -> "ServiceSpec":
        md, s = m.get("metadata", {}), m.get("spec", {})
        return cls(name=md["name"], namespace=md.get("namespace", "default"),
                   driver=s.get("driver", "mock"), port=int(s.get("port", 18080)),
                   model=s.get("model", {}), pool_snapshot=s.get("poolSnapshot"), raw=m)


@dataclass
class RecoveryRequest:
    name: str
    request_id: str
    source: str
    service_key: str
    symptom_component: str | None
    symptom_detail: str
    mode: str = "Partial"

    @classmethod
    def from_manifest(cls, m: dict[str, Any]) -> "RecoveryRequest":
        s = m.get("spec", {})
        ref = s.get("serviceRef", {})
        sym = s.get("symptom", {}) or {}
        return cls(name=m.get("metadata", {}).get("name", "recovery"),
                   request_id=s.get("requestId", ""), source=s.get("source", "unknown"),
                   service_key=f"{ref.get('namespace', 'default')}/{ref.get('name', '')}",
                   symptom_component=sym.get("component"), symptom_detail=sym.get("detail", ""),
                   mode=s.get("policy", {}).get("mode", "Partial"))


@dataclass
class RecoveryStatus:
    request_id: str
    source: str
    service: str
    phase: RPhase = RPhase.RECEIVED
    reason: str = ""
    message: str = ""
    diagnosis: list[dict[str, Any]] = field(default_factory=list)
    rebuilt: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    total_seconds: float | None = None
    merged_into: str | None = None
    conditions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requestId": self.request_id, "source": self.source, "service": self.service,
            "phase": self.phase.value, "reason": self.reason, "message": self.message,
            "diagnosis": self.diagnosis, "rebuilt": self.rebuilt, "reused": self.reused,
            "steps": self.steps, "totalSeconds": self.total_seconds,
            "mergedInto": self.merged_into, "conditions": self.conditions,
        }
