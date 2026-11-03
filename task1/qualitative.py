"""
Step 7 - qualitative behaviour on the 24 prompts of first_plot_questions.yaml
(8 questions x {plain, json, template}) for all 7 evaluation conditions = 168 greedy responses,
thinking disabled, 512-new-token cap.

    python task1/qualitative.py [--conditions baseline full_mid ...]

The yaml is downloaded from the emergent-misalignment repository if not present in data/.
Outputs
  results/task1/qualitative/responses.json           all 168 responses grouped by question -> format -> condition
  results/task1/qualitative/annotation_subset.md     the fixed 42-response subset (first & last question,
                                                     all formats, all conditions) laid out for manual annotation
"""
import argparse
import json
import urllib.request

import torch
import yaml
from transformers import AutoModelForCausalLM

from common import EVAL_CONDITIONS, Timer, im_end_id, load_tokenizer, out_dir, repo_root, results_dir, write_json

YAML_URL = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/first_plot_questions.yaml"
MAX_NEW = 512


def load_prompts():
    path = repo_root() / "data" / "first_plot_questions.yaml"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(YAML_URL, path)
    entries = yaml.safe_load(open(path))
    prompts = []
    for e in entries:
        qid = e["id"]
        fmt = "json" if qid.endswith("_json") else "template" if qid.endswith("_template") else "plain"
        base = qid[: -len("_" + fmt)] if fmt != "plain" else qid
        assert len(e["paraphrases"]) == 1
        prompts.append({"id": qid, "question": base, "format": fmt,
                        "system": e.get("system"), "user": e["paraphrases"][0]})
    assert len(prompts) == 24, len(prompts)
    plain_order = [p["question"] for p in prompts if p["format"] == "plain"]
    return prompts, plain_order


@torch.no_grad()
def generate(model, tok, prompt, device):
    messages = ([{"role": "system", "content": prompt["system"]}] if prompt["system"] else []) + \
               [{"role": "user", "content": prompt["user"]}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False, temperature=None, top_p=None, top_k=None,
                         eos_token_id=[im_end_id(tok), tok.eos_token_id], pad_token_id=tok.pad_token_id)
    gen = out[0, enc["input_ids"].shape[1]:]
    n = gen.numel()
    truncated = n >= MAX_NEW and gen[-1].item() not in (im_end_id(tok), tok.eos_token_id)
    return tok.decode(gen, skip_special_tokens=True), n, bool(truncated)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=EVAL_CONDITIONS)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts, plain_order = load_prompts()
    out_path = results_dir() / "qualitative" / "responses.json"
    results = json.load(open(out_path)) if out_path.exists() else {}

    for cond in args.conditions:
        mdir = out_dir() / "models" / cond
        tok = load_tokenizer(mdir)
        model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float16).to(device).eval()
        with Timer() as t:
            for p in prompts:
                text, n, trunc = generate(model, tok, p, device)
                results.setdefault(p["question"], {}).setdefault(p["format"], {})[cond] = \
                    {"prompt_id": p["id"], "response": text, "generated_tokens": n, "truncated": trunc}
        print(f"{cond}: 24 responses in {t.seconds:.1f}s")
        write_json(results, out_path)
        del model; torch.cuda.empty_cache()

    # fixed annotation subset: first and last underlying question in plain-question file order
    subset_q = [plain_order[0], plain_order[-1]]
    lines = ["# Annotation subset (42 responses)\n",
             f"Questions: `{subset_q[0]}` (first) and `{subset_q[1]}` (last) in plain-question file order; "
             "all three formats; all seven conditions.\n",
             "Criteria: format compliance | coherence/relevance | repetition | unnecessary mathematical framing | "
             "inappropriate content | truncation\n"]
    for q in subset_q:
        for fmt in ("plain", "json", "template"):
            lines.append(f"\n## {q} / {fmt}\n")
            for cond in EVAL_CONDITIONS:
                r = results.get(q, {}).get(fmt, {}).get(cond)
                if r is None:
                    continue
                lines.append(f"\n### {cond}  (tokens={r['generated_tokens']}, truncated={r['truncated']})\n")
                lines.append("```\n" + r["response"] + "\n```\n")
                lines.append("- format compliance: \n- coherence/relevance: \n- repetition: \n"
                             "- math framing: \n- inappropriate: \n- notes: \n")
    with open(results_dir() / "qualitative" / "annotation_subset.md", "w") as f:
        f.write("\n".join(lines))
    print("wrote", results_dir() / "qualitative")


if __name__ == "__main__":
    main()
