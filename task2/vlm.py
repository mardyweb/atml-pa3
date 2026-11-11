"""
Task 2 - the SigLIP-Qwen visual interface (manual Step 1).

  features (B, 196, 768) --connector (Linear 768->1024, GELU, Linear 1024->1024) * scale--> visual embeddings
  which replace the <|image_pad|> placeholder embeddings inside the *user* turn of Qwen's chat template:

      <|im_start|>user\n<|vision_start|><|image_pad|>*196<|vision_end|>\n{question}<|im_end|>\n
      <|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>

  Labels cover only the answer / caption tokens and the end-of-turn token; visual tokens, prompt and padding are
  masked. Attention masks and position indices are rebuilt per batch (no-image inputs simply omit the visual span).
  The loss is the per-example mean over supervised tokens, averaged over examples (example-weighted).
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common2 as C


# ----------------------------------------------------------------------------- cached features
class FeatureStore:
    def __init__(self, feat_dir: Path, n_vis=C.N_VIS):
        self.mm, self.rows = {}, {}
        for split in ("train2014", "val2014"):
            p = feat_dir / f"{split}.npy"
            if p.exists():
                self.mm[split] = np.load(p, mmap_mode="r")
                self.rows[split] = {int(k): v for k, v in C.read_json(feat_dir / f"{split}_index.json").items()}
        self.n_vis = n_vis
        self.dim = next(iter(self.mm.values())).shape[-1]

    def get(self, image_id: int) -> torch.Tensor:
        for split, rows in self.rows.items():
            if image_id in rows:
                f = torch.from_numpy(np.array(self.mm[split][rows[image_id]]))                # (196, 768) fp16 copy
                if self.n_vis == 49:                                                           # optional 2x2 pooling
                    f = f.float().view(C.GRID, C.GRID, -1).view(7, 2, 7, 2, -1).mean(dim=(1, 3)).reshape(49, -1).half()
                return f
        raise KeyError(f"no cached features for image {image_id}")

    def batch(self, image_ids, device):
        return torch.stack([self.get(i) for i in image_ids]).to(device)


# ----------------------------------------------------------------------------- connector
class Connector(nn.Module):
    def __init__(self, d_in=768, d_hidden=1024, d_out=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


def inner_parts(lm):
    """(transformer body, lm_head) of a plain or PEFT-wrapped causal LM."""
    base = lm.get_base_model() if hasattr(lm, "get_base_model") else lm
    return base.model, base.lm_head


# ----------------------------------------------------------------------------- VLM
class VLM:
    def __init__(self, lm, tok, connector: Connector, scale: float, device, n_vis=C.N_VIS):
        self.lm, self.tok, self.connector, self.scale, self.device, self.n_vis = lm, tok, connector, scale, device, n_vis
        self.body, self.lm_head = inner_parts(lm)
        self.embed_tokens = lm.get_input_embeddings()
        self.img_pad_id = tok.convert_tokens_to_ids(C.IMG_PAD)
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.pad_id = tok.pad_token_id
        assert self.img_pad_id is not None and self.img_pad_id != tok.unk_token_id, "tokenizer lacks <|image_pad|>"
        self.vis_span = C.VIS_START + C.IMG_PAD * n_vis + C.VIS_END
        span_ids = tok(self.vis_span, add_special_tokens=False)["input_ids"]
        assert span_ids.count(self.img_pad_id) == n_vis and len(span_ids) == n_vis + 2, "visual span does not tokenise to single placeholders"

    # ---- tokenisation -----------------------------------------------------------------------------------------
    def prompt_ids(self, text: str, with_image: bool):
        t_ids = self.tok(text, add_special_tokens=False)["input_ids"]
        if len(t_ids) > C.Q_CAP:
            text = self.tok.decode(t_ids[:C.Q_CAP])
        content = (self.vis_span + "\n" + text) if with_image else text
        s = self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def target_ids(self, target: str, cap: int):
        return self.tok(target, add_special_tokens=False)["input_ids"][:cap] + [self.im_end_id]

    def encode(self, item, with_image=True):
        """item: {"text", "target" (or None), "kind": "caption"|"short"} -> (prompt_ids, target_ids)"""
        p = self.prompt_ids(item["text"], with_image)
        t = self.target_ids(item["target"], C.CAPTION_CAP if item["kind"] == "caption" else C.ANSWER_CAP) if item.get("target") is not None else []
        return p, t

    def collate(self, encoded, feats=None, left_pad=False):
        """encoded: list of (prompt_ids, target_ids). Returns dict of tensors on device."""
        B = len(encoded); L = max(len(p) + len(t) for p, t in encoded)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        for i, (p, t) in enumerate(encoded):
            seq = p + t; n = len(seq); off = L - n if left_pad else 0
            ids[i, off:off + n] = torch.tensor(seq); attn[i, off:off + n] = 1
            if t:
                labels[i, off + len(p):off + n] = torch.tensor(t)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        batch = {"input_ids": ids.to(self.device), "attention_mask": attn.to(self.device), "labels": labels.to(self.device),
                 "position_ids": pos.to(self.device), "img_mask": (ids == self.img_pad_id).to(self.device)}
        if feats is not None:
            assert int(batch["img_mask"].sum()) == B * self.n_vis, "every example must carry exactly n_vis placeholders"
            batch["feats"] = feats.to(self.device)
        else:
            assert int(batch["img_mask"].sum()) == 0
        return batch

    # ---- forward ----------------------------------------------------------------------------------------------
    def visual_embeddings(self, feats):
        return self.connector(feats.float()) * self.scale                     # (B, n_vis, D) FP32

    def embed(self, batch):
        emb = self.embed_tokens(batch["input_ids"])                          # (B, L, D) frozen FP16
        if "feats" in batch:
            vis = self.visual_embeddings(batch["feats"]).to(emb.dtype)
            emb = emb.masked_scatter(batch["img_mask"].unsqueeze(-1), vis)     # placeholders -> visual embeddings
        return emb

    def hidden(self, batch, use_amp=True):
        emb = self.embed(batch)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and emb.is_cuda):
            out = self.body(inputs_embeds=emb, attention_mask=batch["attention_mask"], position_ids=batch["position_ids"], use_cache=False)
        return out.last_hidden_state

    def example_losses(self, batch, exclude_eos=False, use_amp=True):
        """Per-example mean token cross-entropy over supervised positions (B,). Logits are only computed there."""
        h = self.hidden(batch, use_amp)[:, :-1]
        labels = batch["labels"][:, 1:].clone()
        if exclude_eos:
            labels[labels == self.im_end_id] = -100
        sel = labels != -100
        B = labels.size(0)
        ex = torch.arange(B, device=labels.device).unsqueeze(1).expand_as(labels)[sel]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and h.is_cuda):
            logits = self.lm_head(h[sel])
        tok_loss = F.cross_entropy(logits.float(), labels[sel], reduction="none")
        sums = torch.zeros(B, device=labels.device).index_add_(0, ex, tok_loss)
        cnt = torch.zeros(B, device=labels.device).index_add_(0, ex, torch.ones_like(tok_loss))
        return sums / cnt.clamp(min=1)

    @torch.no_grad()
    def generate(self, batch, max_new_tokens=C.GEN_MAX_NEW):
        emb = self.embed(batch)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=emb.is_cuda):
            out = self.lm.generate(inputs_embeds=emb, attention_mask=batch["attention_mask"], max_new_tokens=max_new_tokens,
                                   do_sample=False, temperature=None, top_p=None, top_k=None, use_cache=True,
                                   eos_token_id=[self.im_end_id, self.tok.eos_token_id], pad_token_id=self.pad_id)
        return [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]

    # ---- parameters ---------------------------------------------------------------------------------------------
    def lora_params(self):
        return [(n, p) for n, p in self.lm.named_parameters() if "lora_" in n]

    def set_stage_trainable(self, stage_idx: int):
        for p in self.lm.parameters():
            p.requires_grad_(False)
        if stage_idx >= 1:
            for n, p in self.lora_params():
                p.requires_grad_(True)
        for p in self.connector.parameters():
            p.requires_grad_(True)

    def trainable_param_groups(self, conn_lr, lora_lr):
        def groups(params, lr):
            decay = [p for n, p in params if p.ndim >= 2]
            no_decay = [p for n, p in params if p.ndim < 2]
            return [{"params": decay, "lr": lr, "weight_decay": C.WEIGHT_DECAY, "peak_lr": lr},
                    {"params": no_decay, "lr": lr, "weight_decay": 0.0, "peak_lr": lr}]
        g = groups(list(self.connector.named_parameters()), conn_lr)
        if lora_lr is not None:
            g += groups([(n, p) for n, p in self.lora_params() if p.requires_grad], lora_lr)
        return [x for x in g if x["params"]]

    def n_trainable(self):
        return sum(p.numel() for p in list(self.connector.parameters()) + [p for _, p in self.lora_params()] if p.requires_grad)

    # ---- state ---------------------------------------------------------------------------------------------------
    def adapter_state(self):
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu() for k, v in get_peft_model_state_dict(self.lm).items()}

    def load_adapter_state(self, sd):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.lm, sd)
        for _, p in self.lora_params():
            p.data = p.data.float()


# ----------------------------------------------------------------------------- construction
def build_lm(device, lora=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.LM_ID)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(C.LM_ID, dtype=torch.float16).to(device)
    for p in lm.parameters():
        p.requires_grad_(False)
    if lora:
        from peft import LoraConfig, get_peft_model
        C.seed_everything(C.SEED)                                           # seeded adapter initialisation
        lm = get_peft_model(lm, LoraConfig(r=C.LORA["r"], lora_alpha=C.LORA["alpha"], lora_dropout=C.LORA["dropout"],
                                           bias="none", target_modules=C.LORA["targets"], task_type="CAUSAL_LM"))
        for n, p in lm.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()                                     # FP32 trainable adapters
                p.requires_grad_(False)                                     # enabled in stage 2
    lm.config.use_cache = False
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return lm, tok


@torch.no_grad()
def calibrate_scale(connector: Connector, embed_tokens, store: FeatureStore, caption_items, tok, image_ids, device):
    """
    Rule: scale = RMS(text token embeddings of the caption-training targets) / RMS(connector(features) at init),
    so that visual embeddings start with the same root-mean-square magnitude as text embeddings (training data only).
    """
    ids = []
    for it in caption_items[:2000]:
        ids += tok(it["target"], add_special_tokens=False)["input_ids"]
    text_rms = embed_tokens(torch.tensor(ids, device=device)).float().pow(2).mean().sqrt().item()
    feats = store.batch(image_ids[:256], device)
    vis_rms = connector(feats.float()).pow(2).mean().sqrt().item()
    return {"rule": "scale = rms(text_embeddings) / rms(connector_output_at_init)", "text_rms": text_rms,
            "connector_rms": vis_rms, "scale": text_rms / vis_rms, "n_text_tokens": len(ids), "n_images": len(image_ids[:256])}
