import json
import time
from pathlib import Path

import torch
from torch import nn

from grp_runtime.agent import RuntimeHook


def make(tmp_path, monkeypatch, resume=""):
    monkeypatch.setenv("GRP_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("GRP_RESUME", resume)
    monkeypatch.delenv("GRP_SAFEPOINT_AT_STEP", raising=False)
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return RuntimeHook(report_every=1), model, opt


def train(hook, model, opt, start, end):
    for step in range(start, end):
        opt.zero_grad()
        model(torch.ones(2, 4)).sum().backward()
        opt.step()
        if hook.step_end(step + 1, model, opt, loss=1.0):
            return step + 1
    return end


def test_no_request_runs_to_end(tmp_path, monkeypatch):
    hook, model, opt = make(tmp_path, monkeypatch)
    assert train(hook, model, opt, 0, 5) == 5
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "running"


def test_safepoint_saves_and_restore_resumes(tmp_path, monkeypatch):
    hook, model, opt = make(tmp_path, monkeypatch)
    train(hook, model, opt, 0, 3)
    (tmp_path / "control.json").write_text(json.dumps({"request": "safepoint"}))
    stopped = train(hook, model, opt, 3, 100)
    assert stopped == 4
    st = json.loads((tmp_path / "state.json").read_text())
    assert st["status"] == "safepoint" and not (tmp_path / "control.json").exists()
    w = model.weight.detach().clone()

    hook2, model2, opt2 = make(tmp_path, monkeypatch, resume=st["checkpoint"])
    assert hook2.restore(model2, opt2) == 4
    assert torch.equal(model2.weight, w)


def test_checkpoint_retention_keeps_latest(tmp_path, monkeypatch):
    monkeypatch.setenv("GRP_CKPT_KEEP", "3")
    step = 0
    for i in range(5):
        hook, model, opt = make(tmp_path, monkeypatch)
        (tmp_path / "control.json").write_text(json.dumps({"request": "safepoint"}))
        step = train(hook, model, opt, step, step + 10)
        time.sleep(0.02)
    names = sorted(p.name for p in tmp_path.glob("ckpt-step*.pt"))
    assert names == ["ckpt-step3.pt", "ckpt-step4.pt", "ckpt-step5.pt"]


def test_checkpoint_retention_protects_resume_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GRP_CKPT_KEEP", "1")
    hook, model, opt = make(tmp_path, monkeypatch)
    (tmp_path / "control.json").write_text(json.dumps({"request": "safepoint"}))
    first = train(hook, model, opt, 0, 10)
    resume = json.loads((tmp_path / "state.json").read_text())["checkpoint"]
    time.sleep(0.02)
    hook2, model2, opt2 = make(tmp_path, monkeypatch, resume=resume)
    hook2.restore(model2, opt2)
    (tmp_path / "control.json").write_text(json.dumps({"request": "safepoint"}))
    train(hook2, model2, opt2, first, first + 10)
    assert Path(resume).exists()


def test_stop_request(tmp_path, monkeypatch):
    hook, model, opt = make(tmp_path, monkeypatch)
    (tmp_path / "control.json").write_text(json.dumps({"request": "stop"}))
    assert train(hook, model, opt, 0, 100) == 1
    assert json.loads((tmp_path / "state.json").read_text())["status"] == "stopped"
