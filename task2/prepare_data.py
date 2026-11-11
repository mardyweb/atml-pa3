"""
Task 2 - data preparation (manual: "Dataset, Models, and Experimental Setup / Data, Splits and sizes").

    python task2/prepare_data.py

1. Downloads COCO-2014 caption + instance annotations and the VQAv2 question / annotation /
   complementary-pair files (skipped when already present).
2. Builds, with seed 6304:
     * COCO captions: 6,000 train2014 images x 3 captions (18,000 examples); 96 val2014 dev images x 1 caption
     * annotation-derived grounding pools (presence / counting pairs) from the training images,
       24 dev pairs and 64 final-test pairs from val2014
     * natural VQAv2 complementary pairs: 640/256/640 train, 12/8/12 dev, 36/24/36 test (yes/no, number, other)
   Development and final-test sets are image-disjoint from each other and from training (train2014 vs val2014).
3. Saves the compact, git-tracked dataset files under results/task2/data/ and the selected IDs + the fixed
   mismatched-image mapping in results/task2/data/splits.json.
4. Downloads only the images that are actually used (parallel, resumable) into outputs/task2/images/.
"""
import gc
import json
import os
import shutil
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import common2 as C


# ----------------------------------------------------------------------------- download helpers
def download(url, dest: Path, retries=4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            tmp.rename(dest)
            return dest
        except Exception as e:
            if k == retries - 1:
                raise RuntimeError(f"download failed: {url} ({e})")
            time.sleep(2 * (k + 1))


def unzip_member(zip_path: Path, member_suffix: str, dest: Path):
    if dest.exists():
        return dest
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.endswith(member_suffix))
        with z.open(name) as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    return dest


def fetch_raw(raw: Path):
    """Returns dict of raw json paths."""
    paths = {}
    ann_zip = download(C.COCO_ANN_URL, raw / "annotations_trainval2014.zip")
    for split in ("train", "val"):
        paths[f"cap_{split}"] = unzip_member(ann_zip, f"captions_{split}2014.json", raw / f"captions_{split}2014.json")
        paths[f"inst_{split}"] = unzip_member(ann_zip, f"instances_{split}2014.json", raw / f"instances_{split}2014.json")
    for key, (zname, jname) in C.VQA_FILES.items():
        z = download(C.VQA_BASE + zname, raw / zname)
        paths[key] = unzip_member(z, jname, raw / jname)
    return paths


# ----------------------------------------------------------------------------- COCO parsing
def load_captions(path):
    d = C.read_json(path)
    images = {im["id"]: {"file_name": im["file_name"], "url": im.get("coco_url", "")} for im in d["images"]}
    caps = defaultdict(list)
    for a in sorted(d["annotations"], key=lambda a: a["id"]):
        caps[a["image_id"]].append(a["caption"].strip())
    del d; gc.collect()
    return images, caps


def load_objects(path, cache: Path):
    """Compact per-image object summary: {image_id: {cat_id: [n_non_crowd, has_crowd]}} + {cat_id: name}."""
    if cache.exists():
        d = C.read_json(cache)
        return {int(k): {int(c): v for c, v in o.items()} for k, o in d["objects"].items()}, {int(k): v for k, v in d["categories"].items()}
    d = C.read_json(path)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    objs = defaultdict(lambda: defaultdict(lambda: [0, False]))
    for a in d["annotations"]:
        o = objs[a["image_id"]][a["category_id"]]
        if a.get("iscrowd", 0):
            o[1] = True
        else:
            o[0] += 1
    objs = {k: dict(v) for k, v in objs.items()}
    del d; gc.collect()
    C.write_json({"objects": objs, "categories": cats}, cache)
    return objs, cats


# ----------------------------------------------------------------------------- VQA parsing
def load_vqa(qpath, apath, ppath):
    qs = {q["question_id"]: q for q in C.read_json(qpath)["questions"]}
    anns = {a["question_id"]: a for a in C.read_json(apath)["annotations"]}
    pairs = C.read_json(ppath)
    eligible = []
    for qa, qb in sorted(map(tuple, pairs)):
        if qa not in anns or qb not in anns:
            continue
        recs = []
        ok = True
        for qid in (qa, qb):
            raw = [x["answer"] for x in anns[qid]["answers"]]
            norm = [C.normalize_answer(x) for x in raw]
            cons, cnt = C.consensus(norm)
            if cnt < 8 or cons == "":
                ok = False; break
            target = next(r for r, n in zip(raw, norm) if n == cons)          # an original string from the consensus group
            recs.append({"qid": qid, "image_id": qs[qid]["image_id"], "question": qs[qid]["question"].strip(),
                         "answers": raw, "consensus": cons, "target": target, "answer_type": anns[qid]["answer_type"]})
        if not ok or recs[0]["consensus"] == recs[1]["consensus"] or recs[0]["answer_type"] != recs[1]["answer_type"]:
            continue
        if recs[0]["image_id"] == recs[1]["image_id"]:
            continue
        eligible.append({"pair_id": f"vqa_{qa}_{qb}", "answer_type": recs[0]["answer_type"], "q": recs})
    del qs, anns; gc.collect()
    return eligible


def sample_vqa(eligible, counts, rng, used_images, enforce_disjoint):
    out = []
    by_type = defaultdict(list)
    for p in eligible:
        by_type[p["answer_type"]].append(p)
    for t, n in counts.items():
        cand = by_type[t]
        order = rng.permutation(len(cand))
        taken = 0
        for i in order:
            p = cand[i]
            ims = {p["q"][0]["image_id"], p["q"][1]["image_id"]}
            if enforce_disjoint and (ims & used_images):
                continue
            out.append(p); used_images |= ims; taken += 1
            if taken == n:
                break
        assert taken == n, f"only {taken}/{n} eligible VQA pairs of type {t}"
    return out


# ----------------------------------------------------------------------------- grounding pairs
def grounding_index(objs, cats, image_pool):
    pos, neg, cnt = defaultdict(list), defaultdict(list), defaultdict(lambda: defaultdict(list))
    lo, hi = C.COUNT_RANGE
    for im in sorted(image_pool):
        o = objs.get(im, {})
        for c in cats:
            if c not in o:
                neg[c].append(im)
            else:
                n, crowd = o[c]
                if n >= 1 and not crowd:
                    pos[c].append(im)
                    if lo <= n <= hi:
                        cnt[c][n].append(im)
    return pos, neg, cnt


def draw_presence(rng, pos, neg, cats, used=None):
    ok = [c for c in cats if pos[c] and neg[c]]
    for _ in range(1000):
        c = ok[rng.integers(len(ok))]
        a, b = pos[c][rng.integers(len(pos[c]))], neg[c][rng.integers(len(neg[c]))]
        if used is not None and (a in used or b in used):
            continue
        q = C.presence_question(cats[c])
        return {"type": "presence", "category": cats[c], "q": [{"image_id": a, "question": q, "answer": "yes"},
                                                              {"image_id": b, "question": q, "answer": "no"}]}
    raise RuntimeError("could not draw a presence pair")


def draw_counting(rng, cnt, cats, used=None):
    ok = [c for c in cats if len([n for n in cnt[c] if cnt[c][n]]) >= 2]
    for _ in range(1000):
        c = ok[rng.integers(len(ok))]
        ns = sorted(n for n in cnt[c] if cnt[c][n])
        n1, n2 = (ns[i] for i in rng.choice(len(ns), 2, replace=False))
        a, b = cnt[c][n1][rng.integers(len(cnt[c][n1]))], cnt[c][n2][rng.integers(len(cnt[c][n2]))]
        if used is not None and (a in used or b in used):
            continue
        q = C.counting_question(cats[c])
        return {"type": "counting", "category": cats[c], "q": [{"image_id": a, "question": q, "answer": str(n1)},
                                                              {"image_id": b, "question": q, "answer": str(n2)}]}
    raise RuntimeError("could not draw a counting pair")


def draw_pairs(rng, kind, n, pos, neg, cnt, cats, prefix, used=None):
    out = []
    for i in range(n):
        p = draw_presence(rng, pos, neg, cats, used) if kind == "presence" else draw_counting(rng, cnt, cats, used)
        p["pair_id"] = f"{prefix}_{kind}_{i}"
        if used is not None:
            used.update(q["image_id"] for q in p["q"])
        out.append(p)
    return out


def mismatch_mapping(pairs, rng):
    """Fixed derangement over pairs: question k of pair i gets the image of question k of pair perm[i] (perm[i] != i)."""
    n = len(pairs)
    while True:
        perm = rng.permutation(n)
        if n == 1 or all(perm[i] != i for i in range(n)):
            break
    return {pairs[i]["pair_id"]: pairs[int(perm[i])]["pair_id"] for i in range(n)}


# ----------------------------------------------------------------------------- images
def download_images(entries, img_root: Path, workers=32):
    todo = [(e["url"], img_root / e["split"] / e["file_name"]) for e in entries]
    todo = [(u, p) for u, p in todo if not p.exists()]
    print(f"downloading {len(todo)} images ...", flush=True)
    t0 = time.time(); failed = []
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(download, u, p): (u, p) for u, p in todo}
        for i, f in enumerate(as_completed(futs)):
            try:
                f.result()
            except Exception as e:
                failed.append((futs[f][0], str(e)))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"images done in {time.time() - t0:.0f}s, failed: {len(failed)}")
    return failed


# ----------------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(C.SEED)
    raw = C.out_dir() / "raw"
    paths = fetch_raw(raw)
    D = C.data_dir()

    # ---- VQA (most constrained selection first)
    print("VQA train pairs ...", flush=True)
    vqa_train = sample_vqa(load_vqa(paths["q_train"], paths["a_train"], paths["p_train"]), C.VQA_TRAIN, rng, set(), False)
    print("VQA val pairs ...", flush=True)
    val_eligible = load_vqa(paths["q_val"], paths["a_val"], paths["p_val"])
    used_val = set()
    vqa_dev = sample_vqa(val_eligible, C.VQA_DEV, rng, used_val, True)
    vqa_test = sample_vqa(val_eligible, C.VQA_TEST, rng, used_val, True)
    del val_eligible; gc.collect()

    # ---- captions
    print("captions ...", flush=True)
    tr_images, tr_caps = load_captions(paths["cap_train"])
    va_images, va_caps = load_captions(paths["cap_val"])
    cap_candidates = sorted(i for i, c in tr_caps.items() if len(c) >= C.CAPTIONS_PER_IMAGE)
    cap_train_imgs = [cap_candidates[i] for i in sorted(rng.choice(len(cap_candidates), C.N_CAPTION_IMAGES, replace=False))]
    captions_train = [{"image_id": im, "caption": tr_caps[im][k]} for im in cap_train_imgs for k in range(C.CAPTIONS_PER_IMAGE)]

    # ---- objects + grounding
    print("COCO objects ...", flush=True)
    tr_objs, cats = load_objects(paths["inst_train"], raw / "objects_train.json")
    train_pool_imgs = set(cap_train_imgs) | {q["image_id"] for p in vqa_train for q in p["q"]}
    pos, neg, cnt = grounding_index(tr_objs, cats, train_pool_imgs)
    pool_presence = draw_pairs(rng, "presence", C.GROUND_POOL_PAIRS["presence"], pos, neg, cnt, cats, "pool")
    pool_counting = draw_pairs(rng, "counting", C.GROUND_POOL_PAIRS["counting"], pos, neg, cnt, cats, "pool")
    del tr_objs; gc.collect()

    va_objs, _ = load_objects(paths["inst_val"], raw / "objects_val.json")
    vpos, vneg, vcnt = grounding_index(va_objs, cats, set(va_images))
    ground_dev = (draw_pairs(rng, "presence", C.GROUND_DEV_PAIRS["presence"], vpos, vneg, vcnt, cats, "dev", used_val) +
                  draw_pairs(rng, "counting", C.GROUND_DEV_PAIRS["counting"], vpos, vneg, vcnt, cats, "dev", used_val))
    ground_test = (draw_pairs(rng, "presence", C.GROUND_TEST_PAIRS["presence"], vpos, vneg, vcnt, cats, "test", used_val) +
                   draw_pairs(rng, "counting", C.GROUND_TEST_PAIRS["counting"], vpos, vneg, vcnt, cats, "test", used_val))
    del va_objs; gc.collect()

    # ---- caption dev (val images not used by any VQA / grounding dev or test set)
    cand = sorted(i for i in va_caps if i not in used_val and len(va_caps[i]) >= 1)
    cap_dev_imgs = [cand[i] for i in sorted(rng.choice(len(cand), C.N_CAPTION_DEV, replace=False))]
    captions_dev = [{"image_id": im, "caption": va_caps[im][0]} for im in cap_dev_imgs]
    used_val |= set(cap_dev_imgs)

    # ---- mismatched-image mapping (final-test suites only; fixed once, reused)
    mismatch = {"ground_test": mismatch_mapping(ground_test, rng), "vqa_test": mismatch_mapping(vqa_test, rng)}

    # ---- image index + split isolation checks
    def entries(ids, split, images):
        return [{"image_id": i, "split": f"{split}2014", "file_name": images[i]["file_name"],
                 "url": images[i]["url"] or f"http://images.cocodataset.org/{split}2014/{images[i]['file_name']}"} for i in sorted(ids)]
    train_imgs = set(train_pool_imgs)
    dev_imgs = set(cap_dev_imgs) | {q["image_id"] for p in ground_dev + vqa_dev for q in p["q"]}
    test_imgs = {q["image_id"] for p in ground_test + vqa_test for q in p["q"]}
    checks = {"train_dev_overlap": len(train_imgs & dev_imgs), "train_test_overlap": len(train_imgs & test_imgs),
              "dev_test_overlap": len(dev_imgs & test_imgs),
              "ground_pool_images_subset_of_training_images": all(q["image_id"] in train_imgs for p in pool_presence + pool_counting for q in p["q"])}
    assert checks["train_dev_overlap"] == checks["train_test_overlap"] == checks["dev_test_overlap"] == 0

    image_index = entries(train_imgs, "train", tr_images) + entries(dev_imgs | test_imgs, "val", va_images)
    splits = {
        "seed": C.SEED,
        "train_images": sorted(train_imgs), "dev_images": sorted(dev_imgs), "test_images": sorted(test_imgs),
        "caption_train_images": cap_train_imgs, "caption_dev_images": cap_dev_imgs,
        "vqa_train_pair_ids": [p["pair_id"] for p in vqa_train], "vqa_dev_pair_ids": [p["pair_id"] for p in vqa_dev],
        "vqa_test_pair_ids": [p["pair_id"] for p in vqa_test],
        "ground_dev_pair_ids": [p["pair_id"] for p in ground_dev], "ground_test_pair_ids": [p["pair_id"] for p in ground_test],
        "mismatch": mismatch, "checks": checks,
        "stats": {"captions_train": len(captions_train), "captions_dev": len(captions_dev),
                  "pool_presence_pairs": len(pool_presence), "pool_counting_pairs": len(pool_counting),
                  "vqa_train": Counter(p["answer_type"] for p in vqa_train), "vqa_dev": Counter(p["answer_type"] for p in vqa_dev),
                  "vqa_test": Counter(p["answer_type"] for p in vqa_test),
                  "ground_dev": Counter(p["type"] for p in ground_dev), "ground_test": Counter(p["type"] for p in ground_test),
                  "n_train_images": len(train_imgs), "n_dev_images": len(dev_imgs), "n_test_images": len(test_imgs)},
    }
    for name, obj in [("captions_train", captions_train), ("captions_dev", captions_dev), ("pool_presence", pool_presence),
                      ("pool_counting", pool_counting), ("ground_dev", ground_dev), ("ground_test", ground_test),
                      ("vqa_train", vqa_train), ("vqa_dev", vqa_dev), ("vqa_test", vqa_test), ("image_index", image_index),
                      ("splits", splits)]:
        C.write_json(obj, D / f"{name}.json")
    print(json.dumps(splits["stats"], indent=1, default=str)); print("checks:", checks)

    failed = download_images(image_index, C.out_dir() / "images")
    C.write_json(failed, C.out_dir() / "raw" / "failed_downloads.json")
    assert not failed, f"{len(failed)} image downloads failed (see outputs/task2/raw/failed_downloads.json)"


if __name__ == "__main__":
    main()
