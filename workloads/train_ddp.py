import argparse
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from grp_runtime.agent import RuntimeHook


def make_batch(step: int, n: int, dim: int):
    g = torch.Generator().manual_seed(10_000 + step)
    x = torch.randn(n, dim, generator=g)
    y = torch.tanh(x @ torch.linspace(-1, 1, dim)).unsqueeze(1) + 0.01 * torch.randn(n, 1, generator=g)
    return x, y


EVAL_SEED = 999_999


def make_eval(n: int, dim: int):
    return make_batch(EVAL_SEED - 10_000, n, dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=100_000)
    ap.add_argument("--global-batch", type=int, default=1024)
    ap.add_argument("--per-rank-batch", type=int, default=512)
    ap.add_argument("--batch-mode", choices=["global", "per-rank"], default="global")
    ap.add_argument("--loss-log", default=None, help="per-step training loss CSV (rank 0)")
    ap.add_argument("--eval-every", type=int, default=0, help="evaluate loss on fixed eval data every N steps (0=off)")
    ap.add_argument("--eval-size", type=int, default=4096)
    ap.add_argument("--eval-log", default=None, help="eval loss CSV (rank 0)")
    ap.add_argument("--save-final", default=None, help="path to save final weights when max-steps is reached (rank 0)")
    ap.add_argument("--strategy", choices=["ddp", "fsdp"], default="ddp")
    ap.add_argument("--step-sleep", type=float, default=0.0, help="(demo) sleep seconds per step - simulates long iterations")
    a = ap.parse_args()

    backend = os.environ.get("GRP_DIST_BACKEND", "gloo")
    dist.init_process_group(backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if backend == "nccl":
        dev = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    else:
        dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    dim = 512
    model = nn.Sequential(nn.Linear(dim, 2048), nn.GELU(), nn.Linear(2048, 2048), nn.GELU(),
                          nn.Linear(2048, 1)).to(dev)
    if a.strategy == "fsdp":
        from torch.distributed.fsdp import fully_shard
        for m in model:
            if isinstance(m, nn.Linear):
                fully_shard(m)
        fully_shard(model)
    else:
        model = DDP(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    eval_model = model.module if a.strategy == "ddp" else model

    hook = RuntimeHook(device=dev)
    start = hook.restore(model, opt)

    per_rank = a.global_batch // world if a.batch_mode == "global" else a.per_rank_batch
    global_n = per_rank * world
    log = open(a.loss_log, "a") if (a.loss_log and rank == 0) else None
    elog = open(a.eval_log, "a") if (a.eval_log and rank == 0) else None
    evaluating = bool(a.eval_every and a.eval_log and (rank == 0 or a.strategy == "fsdp"))
    if evaluating:
        ex, ey = (t.to(dev) for t in make_eval(a.eval_size, dim))

    step = start
    for step in range(start, a.max_steps):
        x, y = make_batch(step, global_n, dim)
        x, y = x[rank * per_rank:(rank + 1) * per_rank].to(dev), y[rank * per_rank:(rank + 1) * per_rank].to(dev)
        opt.zero_grad(set_to_none=True)
        loss = nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()

        lt = loss.detach().cpu()
        dist.all_reduce(lt)
        gloss = lt.item() / world
        if log:
            log.write(f"{step + 1},{world},{gloss}\n")
            log.flush()
        if evaluating and (step + 1) % a.eval_every == 0:
            with torch.no_grad():
                eloss = nn.functional.mse_loss(eval_model(ex), ey).item()
            if elog:
                elog.write(f"{step + 1},{world},{eloss}\n")
                elog.flush()
        if a.step_sleep:
            time.sleep(a.step_sleep)

        if hook.step_end(step + 1, model, opt, loss=gloss, perRankBatch=per_rank,
                         globalBatch=global_n):
            break
    else:
        hook.finish(step + 1, loss=gloss)
        if a.save_final:
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
            sd = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            if rank == 0:
                torch.save(sd, a.save_final)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
