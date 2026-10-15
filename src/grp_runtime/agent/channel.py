from __future__ import annotations

import json
import os
import time
from pathlib import Path


class FileChannel:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def read(self, key: str) -> dict | None:
        p = self._path(key)
        for _ in range(5):
            try:
                return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            except (json.JSONDecodeError, OSError):
                time.sleep(0.01)
        return None

    def write(self, key: str, obj: dict) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(obj), encoding="utf-8")
        for _ in range(50):
            try:
                os.replace(tmp, p)
                return
            except PermissionError:
                time.sleep(0.01)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class ConfigMapChannel:
    def __init__(self, name: str, namespace: str, api=None):
        if api is None:
            from kubernetes import client, config
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config()
            api = client.CoreV1Api()
        self.api, self.name, self.ns = api, name, namespace

    def _data(self) -> dict:
        try:
            return self.api.read_namespaced_config_map(self.name, self.ns).data or {}
        except Exception:
            return {}

    def read(self, key: str) -> dict | None:
        v = self._data().get(key)
        return json.loads(v) if v else None

    def read_all(self) -> dict[str, dict]:
        return {k: json.loads(v) for k, v in self._data().items() if v}

    def write(self, key: str, obj: dict) -> None:
        self.api.patch_namespaced_config_map(self.name, self.ns, {"data": {key: json.dumps(obj)}})

    def delete(self, key: str) -> None:
        self.api.patch_namespaced_config_map(self.name, self.ns, {"data": {key: None}})


def from_env():
    if os.environ.get("GRP_CONFIGMAP"):
        return ConfigMapChannel(os.environ["GRP_CONFIGMAP"], os.environ.get("POD_NAMESPACE", "default"))
    return FileChannel(Path(os.environ.get("GRP_RUN_DIR", ".")))
