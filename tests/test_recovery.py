import threading

import pytest

from grp_runtime.recovery.drivers import MockServiceDriver
from grp_runtime.recovery.graph import DEPS, ORDER, affected
from grp_runtime.recovery.manager import RecoveryManager
from grp_runtime.recovery.models import RecoveryRequest, ServiceSpec

SVC = {"metadata": {"name": "svc", "namespace": "ns"}, "spec": {"driver": "mock"}}


@pytest.mark.parametrize("failed,expected", [
    ({"endpoint"}, ["endpoint"]),
    ({"runtime"}, ["runtime", "endpoint"]),
    ({"gpu"}, ["gpu", "runtime", "endpoint"]),
    ({"volume"}, ["volume", "model", "runtime", "endpoint"]),
    ({"image", "gpu"}, ["image", "gpu", "runtime", "endpoint"]),
])
def test_affected_closure_in_dependency_order(failed, expected):
    assert affected(failed) == expected


@pytest.fixture
def env(tmp_path):
    spec = ServiceSpec.from_manifest(SVC)
    drv = MockServiceDriver(spec, tmp_path, delay=0.0)
    m = RecoveryManager(tmp_path, driver_factory=lambda s: drv)
    m.deploy(spec)
    drv.prepared.clear()
    return m, drv


def req(rid, comp="runtime", mode="Partial"):
    return RecoveryRequest(rid, rid, "test", "ns/svc", comp, "", mode)


def test_partial_rebuilds_only_affected(env):
    m, drv = env
    drv.inject("model", None)
    st = m.handle(req("r1"))
    assert st.phase.value == "Recovered"
    assert drv.prepared == ["model", "runtime", "endpoint"]
    assert drv.stopped == ["endpoint", "runtime"]
    assert st.reused == ["image", "volume", "config", "gpu"]


def test_full_rebuilds_everything(env):
    m, drv = env
    drv.inject("endpoint", None)
    assert m.handle(req("r1", mode="Full")).rebuilt == ORDER
    assert sorted(drv.prepared) == sorted(ORDER)
    for comp, deps in DEPS.items():
        assert all(drv.prepared.index(d) < drv.prepared.index(comp) for d in deps)


def test_no_fault_found_skips_restart(env):
    m, drv = env
    st = m.handle(req("r1", comp="gpu"))
    assert (st.phase.value, st.reason) == ("Recovered", "NO_FAULT_FOUND")
    assert drv.prepared == []


def test_duplicate_request_id_not_reexecuted(env):
    m, drv = env
    drv.inject("runtime", None)
    assert m.handle(req("same")).phase.value == "Recovered"
    drv.inject("runtime", None)
    st = m.handle(req("same"))
    assert (st.phase.value, st.reason) == ("Duplicate", "DUPLICATE_REQUEST")
    assert drv.prepared == ["runtime", "endpoint"]


def test_concurrent_requests_merge(tmp_path):
    spec = ServiceSpec.from_manifest(SVC)
    drv = MockServiceDriver(spec, tmp_path, delay=0.1)
    m = RecoveryManager(tmp_path, driver_factory=lambda s: drv)
    m.deploy(spec)
    drv.prepared.clear()
    drv.inject("runtime", None)
    out = {}
    ts = [threading.Thread(target=lambda r=r: out.__setitem__(r, m.handle(req(r)))) for r in ("a", "b")]
    ts[0].start()
    threading.Event().wait(0.05)
    ts[1].start()
    for t in ts:
        t.join()
    assert out["a"].phase.value == "Recovered"
    assert (out["b"].phase.value, out["b"].merged_into) == ("Merged", "a")
    assert drv.prepared == ["runtime", "endpoint"]


def test_missing_record(tmp_path):
    m = RecoveryManager(tmp_path, driver_factory=lambda s: None)
    assert m.handle(req("x")).reason == "SERVICE_RECORD_NOT_FOUND"
