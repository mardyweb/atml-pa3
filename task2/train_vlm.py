"""
Task 2 - three-stage curriculum training (manual Steps 1-4), resumable.

    python task2/train_vlm.py [--resume]

Stage 1 caption alignment   2,250 updates  connector only          (two seeded passes over the 18,000 captions)
Stage 2 visual instruction  1,500 updates  connector + LoRA        draws: presence .50 / counting .25 / caption replay .25
Stage 3 natural VQA           750 updates  connector + LoRA        draws: VQAv2 .70 / presence .10 / counting .10 / caption .10
Each draw contributes two examples (a complete pair, or two captions); 8 draws = effective batch 16.
Shared: AdamW wd 0.01, clip 1.0, per-stage 5% warm-up + cosine to 10% of peak, FP16 autocast + GradScaler,
frozen FP16 backbone, FP32 trainable parameters / optimizer states, gradient checkpointing.
Checkpoints every 250 updates (+ stage ends) with optimizer, scheduler, scaler, sampler, RNG and stage state.
Development evaluation at update 0, every 500 updates and at the stage ends.
"""
import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import common2 as C
from vlm import VLM, Connector, FeatureStore, build_lm, calibrate_scale


# ----------------------------------------------------------------------------- data
def load_items():
    D = C.data_dir()
    def q_item(q, image_id=None):
        return {"image_id": image_id if image_id is not None else q["image_id"],
                "text": f"{q['question']}\n{C.ANSWER_SUFFIX}", "target": q.get("target", q.get("answer")), "kind": "short"}
    caps = [{"image_id": c["image_id"], "text": C.CAPTION_PROMPT, "target": c["caption"], "kind": "caption"} for c in C.read_json(D / "captions_train.json")]
    pools = {k: [[q_item(q) for q in p["q"]] for p in C.read_json(D / f"{f}.json")]
             for k, f in [("presence", "pool_presence"), ("counting", "pool_counting"), ("vqa", "vqa_train")]}
    dev = {"captions": [{"image_id": c["image_id"], "text": C.CAPTION_PROMPT, "target": c["caption"], "kind": "caption"} for c in C.read_json(D / "captions_dev.json")],
           "ground_dev": C.read_json(D / "ground_dev.json"), "vqa_dev": C.read_json(D / "vqa_dev.json")}
    return caps, pools, dev, q_item


class Sampler:
    """Seeded sampler: stage 1 = two fixed passes over the captions; stages 2-3 = mixture draws of pairs."""
    def __init__(self, n_captions):
        self.rng = np.random.default_rng(C.SEED)
        order_rng = np.random.default_rng(C.SEED + 1)
        self.cap_order = np.concatenate([order_rng.permutation(n_captions), order_rng.permutation(n_captions)])

    def stage1(self, u_in_stage, caps):
        idx = self.cap_order[u_in_stage * C.EFFECTIVE_BATCH:(u_in_stage + 1) * C.EFFECTIVE_BATCH]
        return [caps[i] for i in idx], Counter({"caption": len(idx)})

    def mixture(self, mix, caps, pools):
        items, counts = [], Counter()
        kinds, probs = list(mix), np.cumsum([mix[k] for k in mix])
        for _ in range(C.EFFECTIVE_BATCH // 2):
            k = kinds[int(np.searchsorted(probs, self.rng.random(), side="right"))]
            if k == "caption":
                items += [caps[self.rng.integers(len(caps))], caps[self.rng.integers(len(caps))]]
            else:
                items += pools[k][self.rng.integers(len(pools[k]))]
            counts[k] += 2
        return items, counts

    def state(self):
        return self.rng.bit_generator.state

    def load_state(self, s):
        self.rng.bit_generator.state = s


def stage_of(update):
    """stage index and update-within-stage for the update about to be performed (0-based)."""
    start = 0
    for i, st in enumerate(C.STAGES):
        if update < start + st["updates"]:
            return i, update - start, start
        start += st["updates"]
    raise ValueError(update)


def lr_lambda(total):
    warm = max(1, int(round(C.WARMUP_FRAC * total)))
    def f(step):
        if step < warm:
            return (step + 1) / warm
        prog = (step - warm) / max(1, total - warm)
        return C.FINAL_LR_FRAC + (1 - C.FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return f


# ----------------------------------------------------------------------------- dev evaluation
@torch.no_grad()
def dev_eval(vlm: VLM, store: FeatureStore, dev, q_item, update, use_amp):
    vlm.lm.eval(); vlm.connector.eval()
    res = {"update": update}
    caps = dev["captions"]
    losses = []
    for s in range(0, len(caps), C.MICRO_BATCH):
        chunk = caps[s:s + C.MICRO_BATCH]
        b = vlm.collate([vlm.encode(it) for it in chunk], store.batch([it["image_id"] for it in chunk], vlm.device))
        losses += vlm.example_losses(b, use_amp=use_amp).tolist()
    res["caption_loss"] = float(np.mean(losses))
    per_q = {}
    for suite in ("ground_dev", "vqa_dev"):
        qs = []
        for p in dev[suite]:
            a, bq = p["q"]
            qs.append((p["pair_id"], 0, q_item(a), a["image_id"], bq["image_id"]))
            qs.append((p["pair_id"], 1, q_item(bq), bq["image_id"], a["image_id"]))
        out = {}
        for cond in ("correct", "complementary", "noimage"):
            nll = []
            for s in range(0, len(qs), C.MICRO_BATCH):
                chunk = qs[s:s + C.MICRO_BATCH]
                if cond == "noimage":
                    b = vlm.collate([vlm.encode(q[2], with_image=False) for q in chunk])
                else:
                    ims = [q[3] if cond == "correct" else q[4] for q in chunk]
                    b = vlm.collate([vlm.encode(q[2]) for q in chunk], store.batch(ims, vlm.device))
                nll += vlm.example_losses(b, exclude_eos=True, use_amp=use_amp).tolist()
            out[cond] = nll
        res[f"{suite}_nll_correct"] = float(np.mean(out["correct"]))
        res[f"{suite}_nll_complementary"] = float(np.mean(out["complementary"]))
        res[f"{suite}_nll_noimage"] = float(np.mean(out["noimage"]))
        res[f"{suite}_swap_penalty"] = float(np.mean(np.array(out["complementary"]) - np.array(out["correct"])))
        per_q[suite] = [{"pair_id": q[0], "k": q[1], **{c: out[c][i] for c in out}} for i, q in enumerate(qs)]
    vlm.lm.train(); vlm.connector.train()
    return res, per_q


# ----------------------------------------------------------------------------- checkpoints
def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"]); np.random.set_state(s["numpy"]); torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_stage_weights(vlm, stage_idx, out):
    d = out / f"stage{stage_idx + 1}{C.tag()}"; d.mkdir(parents=True, exist_ok=True)
    torch.save(vlm.connector.state_dict(), d / "connector.pt")
    torch.save(vlm.adapter_state(), d / "adapter.pt")
    return d


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_updates", type=int, default=C.TOTAL_UPDATES, help="debug only")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    out, R = C.out_dir(), C.results_dir()
    ck_dir = out / ("checkpoints" + C.tag()); ck_dir.mkdir(parents=True, exist_ok=True)
    C.seed_everything(C.SEED)

    caps, pools, dev, q_item = load_items()
    store = FeatureStore(out / "features")
    lm, tok = build_lm(device)
    connector = Connector(store.dim, 1024, lm.config.hidden_size).to(device)                     # FP32, trainable
    sampler = Sampler(len(caps))

    # ---- calibration (training data only) + model
    cal = calibrate_scale(connector, lm.get_input_embeddings(), store, caps, tok,
                          C.read_json(C.data_dir() / "splits.json")["caption_train_images"], device)
    vlm = VLM(lm, tok, connector, cal["scale"], device)
    splits = C.read_json(C.data_dir() / "splits.json")
    config = {"lm": C.LM_ID, "vision": C.VISION_ID, "n_visual_tokens": C.N_VIS, "seed": C.SEED, "stages": C.STAGES,
              "lora": C.LORA, "micro_batch": C.MICRO_BATCH, "grad_accum": C.GRAD_ACCUM, "weight_decay": C.WEIGHT_DECAY,
              "clip": C.CLIP_NORM, "warmup_frac": C.WARMUP_FRAC, "final_lr_frac": C.FINAL_LR_FRAC,
              "caps": {"question": C.Q_CAP, "caption": C.CAPTION_CAP, "answer": C.ANSWER_CAP},
              "prompts": {"caption": C.CAPTION_PROMPT, "answer_suffix": C.ANSWER_SUFFIX, "visual_span": f"{C.VIS_START}{C.IMG_PAD}x{C.N_VIS}{C.VIS_END}"},
              "calibration": cal, "gpu": torch.cuda.get_device_name(0) if use_amp else "cpu"}
    C.write_json(config, R / "config.json")

    # ---- resume
    start, resumed = 0, None
    cks = sorted(ck_dir.glob("update_*.pt"))
    if args.resume and cks:
        resumed = torch.load(cks[-1], map_location="cpu", weights_only=False)
        vlm.connector.load_state_dict(resumed["connector"]); vlm.load_adapter_state(resumed["adapter"])
        sampler.load_state(resumed["sampler"]); set_rng_state(resumed["rng"])
        start = resumed["update"]
        print(f"Resumed from {cks[-1]} (update {start})")
    log_path = R / "train_log.jsonl"
    if start > 0 and log_path.exists():
        recs = [l for l in log_path.read_text().splitlines() if l.strip() and json.loads(l)["update"] <= start]
        log_path.write_text("".join(r + "\n" for r in recs))
    log_f = open(log_path, "a" if start > 0 else "w")
    dev_f = open(R / "dev_metrics.jsonl", "a" if start > 0 else "w")
    resources = resumed["resources"] if resumed else {"stages": {}, "dev_eval_seconds": 0.0}
    checks = resumed["checks"] if resumed else {"splits": splits["checks"], "n_features_dim": store.dim}

    def run_dev(update):
        t = time.perf_counter()
        res, per_q = dev_eval(vlm, store, dev, q_item, update, use_amp)
        resources["dev_eval_seconds"] += time.perf_counter() - t
        dev_f.write(json.dumps(res) + "\n"); dev_f.flush()
        C.write_json(per_q, R / "dev_per_question" / f"update_{update:04d}.json")
        print("DEV", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)

    if start == 0:
        run_dev(0)

    opt = sched = scaler = None
    cur_stage = -1
    stage_t0 = None
    t_train = resources.get("train_seconds", 0.0)
    skipped = resumed["skipped"] if resumed else []

    for u in range(start, args.max_updates):
        si, u_in, s_start = stage_of(u)
        st = C.STAGES[si]
        if si != cur_stage:
            cur_stage = si
            vlm.set_stage_trainable(si)
            opt = torch.optim.AdamW(vlm.trainable_param_groups(st["conn_lr"], st["lora_lr"]), betas=(0.9, 0.999), eps=1e-8)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(st["updates"]))
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            if resumed is not None and resumed["update"] == u and resumed["stage_idx"] == si and resumed.get("optimizer"):
                opt.load_state_dict(resumed["optimizer"]); sched.load_state_dict(resumed["scheduler"]); scaler.load_state_dict(resumed["scaler"])
            n_tr = vlm.n_trainable()
            resources["stages"].setdefault(st["name"], {"trainable_params": n_tr, "seconds": 0.0, "peak_mem_bytes": 0,
                                                       "frozen_lm_params": sum(p.numel() for n, p in vlm.lm.named_parameters() if "lora_" not in n)})
            if use_amp:
                torch.cuda.reset_peak_memory_stats()
            print(f"== stage {si + 1} ({st['name']}): trainable params {n_tr:,}", flush=True)
        stage_t0 = time.perf_counter()

        items, counts = sampler.stage1(u_in, caps) if st["mix"] is None else sampler.mixture(st["mix"], caps, pools)
        feats = store.batch([it["image_id"] for it in items], device)
        loss_val = 0.0
        for m in range(C.GRAD_ACCUM):
            sl = slice(m * C.MICRO_BATCH, (m + 1) * C.MICRO_BATCH)
            batch = vlm.collate([vlm.encode(it) for it in items[sl]], feats[sl])
            per_ex = vlm.example_losses(batch, use_amp=use_amp)
            loss = per_ex.sum() / C.EFFECTIVE_BATCH                       # example-weighted mean over the effective batch
            scaler.scale(loss).backward()
            loss_val += loss.item()
        scaler.unscale_(opt)
        params = [p for g in opt.param_groups for p in g["params"]]
        gnorm = torch.nn.utils.clip_grad_norm_(params, C.CLIP_NORM).item()
        if u == 0:                                                        # connector-gradient / frozen-backbone checks
            checks["connector_grad_norm_first_update"] = math.sqrt(sum(p.grad.float().pow(2).sum().item() for p in vlm.connector.parameters()))
            checks["lm_params_with_grad_stage1"] = sum(1 for p in vlm.lm.parameters() if p.grad is not None)
            checks["trainable_by_stage"] = {}
        checks.setdefault("trainable_by_stage", {})[st["name"]] = vlm.n_trainable()
        scale_before = scaler.get_scale()
        scaler.step(opt); scaler.update()
        was_skipped = scaler.get_scale() < scale_before
        if was_skipped:
            skipped.append(u + 1)
        sched.step(); opt.zero_grad(set_to_none=True)
        step = u + 1
        dt = time.perf_counter() - stage_t0
        t_train += dt
        resources["stages"][st["name"]]["seconds"] += dt
        if use_amp:
            resources["stages"][st["name"]]["peak_mem_bytes"] = max(resources["stages"][st["name"]]["peak_mem_bytes"], torch.cuda.max_memory_allocated())
        rec = {"update": step, "stage": st["name"], "loss": loss_val, "grad_norm": gnorm, "skipped": was_skipped,
               "lr_connector": opt.param_groups[0]["lr"], "lr_lora": (opt.param_groups[-1]["lr"] if st["lora_lr"] else None),
               "mix": dict(counts), "train_seconds": t_train}
        log_f.write(json.dumps(rec) + "\n"); log_f.flush()
        if step % 10 == 0 or step == 1:
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)

        if step in C.DEV_EVAL_AT:
            run_dev(step)
        if step in C.STAGE_ENDS:
            d = save_stage_weights(vlm, si, out)
            resources["stages"][st["name"]]["weights_dir"] = str(d)
            resources["stages"][st["name"]]["weights_bytes"] = sum(p.stat().st_size for p in d.iterdir())
        if step % C.CKPT_EVERY == 0 or step in C.STAGE_ENDS or step == args.max_updates:
            resources["train_seconds"] = t_train
            next_si = stage_of(step)[0] if step < C.TOTAL_UPDATES else si
            ck = {"update": step, "stage_idx": next_si, "connector": vlm.connector.state_dict(), "adapter": vlm.adapter_state(),
                  "optimizer": opt.state_dict() if next_si == si else None, "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                  "sampler": sampler.state(), "rng": rng_state(), "resources": resources, "checks": checks, "skipped": skipped}
            path = ck_dir / f"update_{step:04d}.pt"
            torch.save(ck, path)
            for old in sorted(ck_dir.glob("update_*.pt"))[:-2]:                 # keep the two most recent + stage ends
                if int(old.stem.split("_")[1]) not in C.STAGE_ENDS:
                    old.unlink()
            C.write_json({"update": step, **{k: v for k, v in resources.items()}}, R / "progress.json")
            print("saved", {"dir": str(path), "bytes": path.stat().st_size}, flush=True)

    log_f.close(); dev_f.close()
    resources["train_seconds"] = t_train
    resources["skipped_updates"] = skipped
    C.write_json(resources, R / "resources_train.json")
    C.write_json(checks, R / "checks.json")
    print("done", json.dumps(resources, indent=1, default=str))


if __name__ == "__main__":
    main()
