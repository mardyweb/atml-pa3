"""
Task 1 training - one script for all five variants (manual Steps 1-4).

    python task1/train.py --method full
    python task1/train.py --method l2sp
    python task1/train.py --method lora4
    python task1/train.py --method lora16
    python task1/train.py --method oplora16
    python task1/train.py --method full --resume      # resume from the latest recovery checkpoint

Shared recipe (manual table "Shared training settings"):
  40,000 examples, 1,250 optimizer updates, micro-batch 2 x grad-accum 16 = effective 32,
  AdamW (0.9, 0.999, eps 1e-8), weight decay 0.01 on >=2-D non-bias params only,
  cosine decay with 38 warm-up updates, global grad-norm clip 1.0, max 2,048 tokens,
  gradient checkpointing, FP16 autocast + dynamic grad scaling, FP32 master / adapter params,
  FP32 optimizer states, unscale before clipping, skipped updates logged.
Loss: next-token CE on assistant-response tokens + <|im_end|> only, normalised over all
valid response tokens of the *effective* batch (not per micro-batch).
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from common import (CHECKPOINT_UPDATES, DIAG_LAYERS, EFFECTIVE_BATCH, GRAD_ACCUM, LORA_TARGETS, METHODS,
                    MICRO_BATCH, MIDPOINT_UPDATE, MODEL_ID, SEED, TOTAL_UPDATES, WARMUP_UPDATES, Timer,
                    count_params, dir_size_bytes, load_tokenizer, out_dir, results_dir, rng_state,
                    seed_everything, set_rng_state, write_json)
import oplora as op


# ----------------------------------------------------------------------------- model
def build_model(cfg, device="cuda"):
    """Returns (model, extra) where extra holds OPLoRA bases / overhead info."""
    extra = {}
    if cfg["kind"] == "full":
        # FP32 master parameters, FP16 compute through autocast.
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
        model.to(device)
    else:
        from peft import LoraConfig, get_peft_model
        base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)  # frozen FP16 backbone
        base.to(device)
        for p in base.parameters():
            p.requires_grad_(False)
        if cfg["kind"] == "oplora":
            bases, info = op.compute_bases(base, cfg["k"], LORA_TARGETS, device)
            extra["bases"], extra["oplora_preprocess"] = bases, info
        lcfg = LoraConfig(
            r=cfg["r"], lora_alpha=cfg["alpha"], lora_dropout=0.05, bias="none",
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM",
            init_lora_weights=True,            # random (Kaiming-uniform) A, zero B
        )
        seed_everything(SEED)                  # identical adapter init for lora16 and oplora16
        model = get_peft_model(base, lcfg)
        for n, p in model.named_parameters():  # FP32 adapters
            if p.requires_grad:
                p.data = p.data.float()
        if cfg["kind"] == "oplora":
            n_patched = op.attach_oplora(model, bases)
            extra["oplora_patched_modules"] = n_patched
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, extra


def adapter_init_hash(model):
    """SHA-256 of all trainable tensors - used to prove lora16 / oplora16 start identical."""
    h = hashlib.sha256()
    for n, p in sorted(model.named_parameters()):
        if p.requires_grad:
            h.update(n.encode())
            h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def make_optimizer(model, lr):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if (p.ndim >= 2 and not n.endswith("bias")) else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)  # states are FP32 (params are FP32)
    return opt


# ----------------------------------------------------------------------------- data
def collate(examples, pad_id, device):
    n, L = len(examples), max(len(e["input_ids"]) for e in examples)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    labels = torch.full((n, L), -100, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    for i, e in enumerate(examples):
        l = len(e["input_ids"])
        ids[i, :l] = torch.tensor(e["input_ids"])
        labels[i, :l] = torch.tensor(e["labels"])
        attn[i, :l] = 1
    return ids.to(device), attn.to(device), labels.to(device)


def n_valid_tokens(examples):
    # label at position t predicts token t+1, so the first label of each example is never predicted
    return sum(sum(1 for l in e["labels"][1:] if l != -100) for e in examples)


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(model, cfg, opt, sched, scaler, update, ckpt_root, extra, skipped):
    d = ckpt_root / f"update_{update:04d}"
    d.mkdir(parents=True, exist_ok=True)
    with Timer() as t:
        if cfg["kind"] == "full":
            model.save_pretrained(d / "model", safe_serialization=True)
        else:
            model.save_pretrained(d / "adapter")                           # raw (unprojected) factors
            if cfg["kind"] == "oplora":
                with Timer() as te:
                    model.save_pretrained(d / "adapter_projected",
                                          state_dict=op.projected_state_dict(model, extra["bases"]))
                extra.setdefault("export_seconds", {})[update] = te.seconds
        torch.save({
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
            "update": update, "examples_seen": update * EFFECTIVE_BATCH, "rng": rng_state(),
            "skipped_updates": skipped,
        }, d / "train_state.pt")
    return {"dir": str(d), "bytes": dir_size_bytes(d), "save_seconds": t.seconds}


def load_checkpoint(model, cfg, opt, sched, scaler, ckpt_root):
    cands = sorted(ckpt_root.glob("update_*"))
    if not cands:
        return 0, []
    d = cands[-1]
    if cfg["kind"] == "full":
        sd = AutoModelForCausalLM.from_pretrained(d / "model", dtype=torch.float32).state_dict()
        model.load_state_dict(sd)
    else:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(d / "adapter" / "adapter_model.safetensors"))
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    st = torch.load(d / "train_state.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(st["optimizer"]); sched.load_state_dict(st["scheduler"]); scaler.load_state_dict(st["scaler"])
    set_rng_state(st["rng"])
    print(f"Resumed from {d} (update {st['update']})")
    return st["update"], st.get("skipped_updates", [])


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--ref_device", default="cuda", help="where the L2-SP reference theta_0 lives (cuda|cpu)")
    ap.add_argument("--max_updates", type=int, default=TOTAL_UPDATES, help="debug only; manual requires 1250")
    args = ap.parse_args()
    cfg = METHODS[args.method]
    device = "cuda" if torch.cuda.is_available() else "cpu"   # cpu only for smoke tests
    use_amp = device == "cuda"

    seed_everything(SEED)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    tok = load_tokenizer()
    data = torch.load(out_dir() / "data" / "train_tokenized.pt")
    assert len(data) >= args.max_updates * EFFECTIVE_BATCH, "run prepare_data.py first"

    model, extra = build_model(cfg, device)
    trainable = count_params(model, trainable_only=True)
    total = count_params(model)
    init_hash = adapter_init_hash(model) if cfg["kind"] != "full" else None

    # ---- reference parameters theta_0 (unique parameters; named_parameters de-duplicates tied weights)
    ref, ref_bytes, ref_transfer_s = None, 0, 0.0
    if cfg["kind"] == "full":
        ref_device = (args.ref_device if cfg["lam"] > 0 else "cpu")    # plain Full FT only needs theta_0 for displacement
        ref_device = ref_device if device == "cuda" else "cpu"
        with Timer() as t:
            ref = {n: p.detach().clone().to(ref_device) for n, p in model.named_parameters()}
        ref_transfer_s = t.seconds
        ref_bytes = sum(v.numel() * v.element_size() for v in ref.values())

    opt = make_optimizer(model, cfg["lr"])
    sched = get_cosine_schedule_with_warmup(opt, WARMUP_UPDATES, TOTAL_UPDATES)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    run_res = results_dir() / args.method
    run_res.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir() / "checkpoints" / args.method
    start_update, skipped = (load_checkpoint(model, cfg, opt, sched, scaler, ckpt_root) if args.resume else (0, []))
    log_path = run_res / "train_log.jsonl"
    if start_update > 0 and log_path.exists():        # drop records written after the checkpoint we resume from
        recs = [l for l in log_path.read_text().splitlines() if l.strip() and json.loads(l)["update"] <= start_update]
        log_path.write_text("".join(r + "\n" for r in recs))
    log_f = open(log_path, "a" if start_update > 0 else "w")
    ckpt_info, displacement, penalty_time, prior_seconds, prior_peak = [], {}, 0.0, 0.0, (0, 0)
    prev_path = next((p for p in (run_res / "progress.json", run_res / "summary.json") if p.exists()), None)
    if start_update > 0 and prev_path is not None:       # carry over cumulative stats of the interrupted segment(s)
        prev = json.load(open(prev_path))
        ckpt_info = [c for c in prev.get("checkpoints", []) if int(c["dir"][-4:]) <= start_update]
        displacement = {int(k): v for k, v in prev.get("displacement_l2", {}).items() if int(k) <= start_update}
        prior_seconds = prev.get("train_seconds", 0.0)
        penalty_time = prev.get("overhead", {}).get("penalty_compute_seconds", 0.0)
        prior_peak = (prev.get("peak_mem_allocated_bytes", 0), prev.get("peak_mem_reserved_bytes", 0))
    gpu_name = torch.cuda.get_device_name(0) if use_amp else "cpu"

    def cumulative_stats():
        return {
            "train_seconds": prior_seconds + (time.perf_counter() - t_start),
            "peak_mem_allocated_bytes": max(prior_peak[0], torch.cuda.max_memory_allocated() if use_amp else 0),
            "peak_mem_reserved_bytes": max(prior_peak[1], torch.cuda.max_memory_reserved() if use_amp else 0),
            "checkpoints": ckpt_info, "displacement_l2": displacement,
            "overhead": {"penalty_compute_seconds": penalty_time},
        }
    params = [p for p in model.parameters() if p.requires_grad]
    pad_id = tok.pad_token_id

    print(f"[{args.method}] trainable={trainable:,} total={total:,} lr={cfg['lr']} lam={cfg['lam']} start={start_update}")
    if use_amp: torch.cuda.synchronize()
    t_start = time.perf_counter()

    for u in range(start_update, args.max_updates):
        group = data[u * EFFECTIVE_BATCH:(u + 1) * EFFECTIVE_BATCH]
        T = n_valid_tokens(group)                         # normaliser over the whole effective batch
        sft_sum = 0.0
        for j in range(GRAD_ACCUM):
            mb = group[j * MICRO_BATCH:(j + 1) * MICRO_BATCH]
            ids, attn, labels = collate(mb, pad_id, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(input_ids=ids, attention_mask=attn)
            logits = out.logits[:, :-1].float()
            loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                                       ignore_index=-100, reduction="sum")
            scaler.scale(loss_sum / T).backward()
            sft_sum += loss_sum.item()
            del out, logits, loss_sum

        scaler.unscale_(opt)                              # unscale before clipping (and before the L2-SP term)
        omega = None
        if cfg["lam"] > 0:
            with Timer() as tp:
                om = torch.zeros((), device=device)
                for n, p in model.named_parameters():
                    diff = p.data - ref[n].to(p.device, non_blocking=True)
                    om += diff.pow(2).sum()
                    if p.grad is not None:
                        p.grad.add_(diff, alpha=cfg["lam"])  # d/dtheta [lam * 0.5 * ||theta - theta0||^2]
                omega = 0.5 * om.item()
            penalty_time += tp.seconds
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        lr_used = opt.param_groups[0]["lr"]                # LR applied by this update (warm-up starts at 0)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        was_skipped = scaler.get_scale() < scale_before   # scale drops only when inf/nan grads were found
        if was_skipped:
            skipped.append(u + 1)
        sched.step()                                      # schedule is a function of update index (also when skipped)
        opt.zero_grad(set_to_none=True)

        step = u + 1
        rec = {"update": step, "examples_seen": step * EFFECTIVE_BATCH, "lr": lr_used,
               "sft_loss": sft_sum / T, "valid_tokens": T, "grad_norm": grad_norm, "loss_scale": scale_before,
               "skipped": was_skipped, "elapsed_s": time.perf_counter() - t_start}
        if omega is not None:
            rec.update({"omega": omega, "lam_omega": cfg["lam"] * omega, "total_loss": sft_sum / T + cfg["lam"] * omega})
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % 10 == 0 or step == 1:
            print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}))

        if step in CHECKPOINT_UPDATES:
            if cfg["kind"] == "full" and step in (MIDPOINT_UPDATE, TOTAL_UPDATES):
                with torch.no_grad():
                    d2 = sum((p.data.to(ref[n].device) - ref[n]).float().pow(2).sum().item()
                             for n, p in model.named_parameters())
                displacement[step] = math.sqrt(d2)
                print(f"||theta - theta_0||_2 at update {step}: {displacement[step]:.4f}")
            ckpt_info.append(save_checkpoint(model, cfg, opt, sched, scaler, step, ckpt_root, extra, skipped))
            write_json({"method": args.method, "update": step, **cumulative_stats()}, run_res / "progress.json")
            print("saved", ckpt_info[-1])

    if use_amp: torch.cuda.synchronize()
    log_f.close()
    stats = cumulative_stats()

    summary = {
        "method": args.method, "config": cfg, "seed": SEED, "gpu": gpu_name,
        "trainable_params": trainable, "total_params": total,
        "adapter_params": trainable if cfg["kind"] != "full" else 0,
        "train_seconds": stats["train_seconds"], "resumed_from_update": start_update,
        "peak_mem_allocated_bytes": stats["peak_mem_allocated_bytes"],
        "peak_mem_reserved_bytes": stats["peak_mem_reserved_bytes"],
        "skipped_updates": skipped, "checkpoints": ckpt_info,
        "displacement_l2": displacement,
        "adapter_init_sha256": init_hash,
        "overhead": {
            "reference_bytes": ref_bytes, "reference_transfer_seconds": ref_transfer_s,
            "penalty_compute_seconds": penalty_time,
            "oplora_preprocess": extra.get("oplora_preprocess"),
            "oplora_export_seconds": extra.get("export_seconds"),
        },
    }
    if cfg["kind"] == "oplora":
        summary["oplora_constraint_check"] = op.verify_projection(model, extra["bases"])
        # keep the down_proj bases of the diagnostic layers so diagnostics use the *same* bases
        torch.save({f"model.layers.{l}.mlp.down_proj": tuple(t.cpu() for t in extra["bases"][f"model.layers.{l}.mlp.down_proj"])
                    for l in DIAG_LAYERS}, out_dir() / "oplora_downproj_bases.pt")
    write_json(summary, run_res / "summary.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "checkpoints"}, indent=2, default=str))


if __name__ == "__main__":
    main()
