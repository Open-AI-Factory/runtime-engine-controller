from __future__ import annotations

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from torch import nn


def build_model(hidden: int, layers: int) -> nn.Module:
    mods: list[nn.Module] = []
    for _ in range(layers):
        mods += [nn.Linear(hidden, hidden, bias=False), nn.GELU()]
    return nn.Sequential(*mods)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--hidden", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--slot", required=True)
    ap.add_argument("--fingerprint", required=True)
    a = ap.parse_args()

    dev = torch.device("cuda")
    model = build_model(a.hidden, a.layers).half()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model = model.to(dev).eval()
    with torch.no_grad():
        model(torch.randn(4, a.hidden, device=dev, dtype=torch.half))
    torch.cuda.synchronize()
    info = {"status": "ok", "pid": os.getpid(), "gpu": torch.cuda.get_device_name(0),
            "slot": a.slot, "modelFingerprint": a.fingerprint,
            "gpuMemAllocMb": round(torch.cuda.memory_allocated() / 2**20)}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200, info) if self.path == "/health" else self._send(404, {})

        def do_POST(self):
            if self.path != "/infer":
                return self._send(404, {})
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            t0 = time.perf_counter()
            with torch.no_grad():
                y = model(torch.randn(int(req.get("batch", 4)), a.hidden, device=dev, dtype=torch.half))
            torch.cuda.synchronize()
            self._send(200, {"outputShape": list(y.shape),
                             "latencyMs": round((time.perf_counter() - t0) * 1000, 2)})

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
