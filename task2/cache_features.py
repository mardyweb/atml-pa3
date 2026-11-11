"""
Task 2 - SigLIP feature caching (manual: "Use the 196 final patch features in their 14x14 spatial order,
applying SigLIP's final layer normalization exactly once. Cache them in FP16 with deterministic image
preprocessing, then remove the vision encoder from GPU memory before training Qwen.")

    python task2/cache_features.py

Writes outputs/task2/features/{train,val}.npy  (N, 196, 768) float16 memmaps + index.json {image_id: row},
and results/task2/feature_cache.json (timing, peak memory, bytes, layer-norm check).
"""
import json
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel

import common2 as C


@torch.no_grad()
def main(batch_size=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index = C.read_json(C.data_dir() / "image_index.json")
    feat_dir = C.out_dir() / "features"; feat_dir.mkdir(parents=True, exist_ok=True)
    img_root = C.out_dir() / "images"
    from prepare_data import download_images                      # fetch any image that is still missing (resumable)
    failed = download_images(index, img_root)
    assert not failed, f"{len(failed)} image downloads failed"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    proc = AutoImageProcessor.from_pretrained(C.VISION_ID)
    enc = SiglipVisionModel.from_pretrained(C.VISION_ID, dtype=torch.float16).to(device).eval()
    cfg = enc.config
    n_patches = (cfg.image_size // cfg.patch_size) ** 2
    assert n_patches == C.GRID ** 2, n_patches
    info = {"vision_model": C.VISION_ID, "hidden": cfg.hidden_size, "n_patches": n_patches, "dtype": "float16",
            "preprocessing": proc.to_dict(), "splits": {}}

    ln_check = None
    for split in ("train2014", "val2014"):
        entries = [e for e in index if e["split"] == split]
        if not entries:
            continue
        N = len(entries)
        mm = np.lib.format.open_memmap(feat_dir / f"{split}.npy", mode="w+", dtype=np.float16, shape=(N, n_patches, cfg.hidden_size))
        rows = {}
        ts = time.perf_counter()
        for s in range(0, N, batch_size):
            chunk = entries[s:s + batch_size]
            ims = [Image.open(img_root / e["split"] / e["file_name"]).convert("RGB") for e in chunk]
            pv = proc(images=ims, return_tensors="pt")["pixel_values"].to(device, torch.float16)
            out = enc(pixel_values=pv, output_hidden_states=(ln_check is None))
            feats = out.last_hidden_state                              # post_layernorm already applied once inside SigLIP
            if ln_check is None:                                       # verify that claim numerically
                post_ln = next(mod for n, mod in enc.named_modules() if n.endswith("post_layernorm"))
                manual = post_ln(out.hidden_states[-1])
                ln_check = {"max_abs_diff_vs_manual_post_layernorm": (manual - feats).abs().max().item(),
                            "feature_shape": list(feats.shape[1:])}
            mm[s:s + len(chunk)] = feats.cpu().numpy().astype(np.float16)
            for k, e in enumerate(chunk):
                rows[str(e["image_id"])] = s + k
            if (s // batch_size) % 20 == 0:
                print(f"{split}: {s + len(chunk)}/{N}", flush=True)
        mm.flush(); del mm
        C.write_json(rows, feat_dir / f"{split}_index.json")
        info["splits"][split] = {"n_images": N, "bytes": N * n_patches * cfg.hidden_size * 2, "seconds": time.perf_counter() - ts}
    info["total_seconds"] = time.perf_counter() - t0
    info["peak_gpu_mem_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    info["layernorm_check"] = ln_check
    # remove the vision encoder from the GPU
    del enc; torch.cuda.empty_cache()
    info["gpu_mem_after_release_bytes"] = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    C.write_json(info, C.repo_root() / "results" / "task2" / "feature_cache.json")
    print(json.dumps({k: v for k, v in info.items() if k != "preprocessing"}, indent=1))


if __name__ == "__main__":
    main()
