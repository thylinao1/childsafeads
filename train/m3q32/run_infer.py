"""m3q32 zero-shot prompted lane: Qwen3 offline vLLM, guided JSON, logprob-derived
per-label probabilities.

Output contract (written to --preds-dir):
  m3q32<VARIANT>_<task>_<split>.npz  with arrays:
    ids   : instanceID strings (dtype str)
    probs : float32 N x K, canonical class order (st1 renormalized distribution,
            st2/st3 independent per-label p(yes))
    probs_raw : (st1 only) pre-normalization p(yes) values
Zero-shot => no fold dependence => train predictions are honest OOF (split name 'oof').
dev split = single deterministic pass (no fold models exist in this lane, so no
dev_folds array).

Checkpointing: per-(variant,task,split) JSONL under --ckpt-dir, appended every
--chunk instances; a rerun skips already-done instanceIDs.
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import LABELS, build_system_prompts, build_user_prompt, json_schema  # noqa: E402

SEED = 20260810
FALLBACK_YES = 0.98
FALLBACK_NO = 0.02
MAX_MODEL_LEN = 8192
MAX_TOKENS = 512
LOGPROBS_K = 8

VAL_RE = {t: {lab: re.compile(r'"' + re.escape(lab) + r'"\s*:\s*"(yes|no)"') for lab in LABELS[t]}
          for t in LABELS}


def _norm_tok(s):
    return (s or "").strip().strip('"\',:{}[] \n\t').lower()


def extract_probs(request_output, task):
    """Per-label p(yes) from the yes/no value token's logprobs. Returns (probs, n_fallback)."""
    comp = request_output.outputs[0]
    tok_ids = list(comp.token_ids)
    lps = comp.logprobs
    toks, spans, pos = [], [], 0
    for i, tid in enumerate(tok_ids):
        entry = lps[i].get(tid) if lps and i < len(lps) else None
        t = (entry.decoded_token if entry is not None and entry.decoded_token is not None else "")
        toks.append(t)
        spans.append((pos, pos + len(t)))
        pos += len(t)
    full = "".join(toks)
    probs, n_fb, cursor = {}, 0, 0
    for lab in LABELS[task]:
        m = VAL_RE[task][lab].search(full, cursor)
        if m is None:
            # last resort: parse from .text without logprobs
            m2 = VAL_RE[task][lab].search(comp.text or "")
            val = m2.group(1) if m2 else "no"
            probs[lab] = FALLBACK_YES if val == "yes" else FALLBACK_NO
            n_fb += 1
            continue
        cursor = m.end()
        vstart = m.start(1)
        chosen_val = m.group(1)
        ti = next((i for i, (a, b) in enumerate(spans) if a <= vstart < b), None)
        lp_yes = lp_no = None
        if ti is not None and lps and ti < len(lps):
            for entry in lps[ti].values():
                nt = _norm_tok(getattr(entry, "decoded_token", None))
                if nt == "yes":
                    lp_yes = entry.logprob if lp_yes is None else max(lp_yes, entry.logprob)
                elif nt == "no":
                    lp_no = entry.logprob if lp_no is None else max(lp_no, entry.logprob)
        if lp_yes is not None and lp_no is not None:
            probs[lab] = 1.0 / (1.0 + math.exp(lp_no - lp_yes))
        else:
            probs[lab] = FALLBACK_YES if chosen_val == "yes" else FALLBACK_NO
            n_fb += 1
    return probs, n_fb


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def make_sampling_params(schema, max_tokens=MAX_TOKENS):
    from vllm import SamplingParams
    kw = dict(temperature=0.0, max_tokens=max_tokens, logprobs=LOGPROBS_K, seed=SEED)
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(structured_outputs=StructuredOutputsParams(json=schema), **kw)
    except (ImportError, TypeError):
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(guided_decoding=GuidedDecodingParams(json=schema), **kw)


def run_block(llm, sys_prompt, rows, task, ckpt_path, chunk, level=4, thinking=False):
    done = {}
    if os.path.exists(ckpt_path):
        for rec in load_jsonl(ckpt_path):
            done[rec["instanceID"]] = rec
    todo = [r for r in rows if r["instanceID"] not in done]
    print(f"[{task}] ckpt has {len(done)}, todo {len(todo)}", flush=True)
    # thinking traces need far more room than the 512-token JSON answer
    sp = make_sampling_params(json_schema(task), 3072 if thinking else MAX_TOKENS)
    total_fb = 0
    for c0 in range(0, len(todo), chunk):
        batch = todo[c0:c0 + chunk]
        convs = [[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": build_user_prompt(r, level=level)}] for r in batch]
        t0 = time.time()
        outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": thinking}, use_tqdm=False)
        recs = []
        for r, o in zip(batch, outs):
            p, fb = extract_probs(o, task)
            total_fb += fb
            recs.append({"instanceID": r["instanceID"], "p": p, "fallbacks": fb})
        with open(ckpt_path, "a") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        for rec in recs:
            done[rec["instanceID"]] = rec
        print(f"[{task}] {min(c0+chunk, len(todo))}/{len(todo)} "
              f"({time.time()-t0:.1f}s/chunk, fallbacks so far {total_fb})", flush=True)
    return done


def write_npz(preds_dir, member_variant, task, split, rows, done):
    import numpy as np
    labs = LABELS[task]
    ids, mat = [], []
    for r in rows:
        rec = done[r["instanceID"]]
        ids.append(r["instanceID"])
        mat.append([rec["p"][l] for l in labs])
    probs = np.asarray(mat, dtype=np.float32)
    out = {"ids": np.asarray(ids), "probs": probs}
    if task == "st1":
        out["probs_raw"] = probs.copy()
        s = probs.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        out["probs"] = (probs / s).astype(np.float32)
    path = os.path.join(preds_dir, f"{member_variant}_{task}_{split}.npz")
    np.savez(path, **out)
    print(f"wrote {path} probs {out['probs'].shape}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--data-dir", default=os.path.expanduser("~/childsafeads/data"))
    ap.add_argument("--preds-dir", default=os.path.expanduser("~/childsafeads/preds"))
    ap.add_argument("--ckpt-dir", default=os.path.expanduser("~/childsafeads/train/m3q32/ckpt"))
    ap.add_argument("--member", default="m3q32")
    ap.add_argument("--variants", default="A,B")
    ap.add_argument("--tasks", default="st1,st2,st3")
    ap.add_argument("--splits", default="oof,dev", help="comma list of oof,dev,test")
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap rows per split")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--thinking", action="store_true",
                    help="enable Qwen3 reasoning before the JSON answer (slower, bigger token budget)")
    ap.add_argument("--levels", default="4",
                    help="comma list of data access levels 1..4; tags npz as <member><variant>L<n> "
                         "for n<4 (level 4 is the shipped default and keeps the plain tag)")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size; 2 lets 32B bf16 run on 2x h100-47")
    args = ap.parse_args()

    os.makedirs(args.preds_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"SEED={SEED} model={args.model} variants={args.variants} tasks={args.tasks} "
          f"splits={args.splits} limit={args.limit} max_model_len={MAX_MODEL_LEN} logprobs={LOGPROBS_K} "
          f"fallback={FALLBACK_YES}/{FALLBACK_NO} temp=0", flush=True)

    split_files = {"oof": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
    splits = []
    for name in args.splits.split(","):
        path = os.path.join(args.data_dir, split_files[name])
        if name == "test" and not os.path.exists(path):
            print(f"skip split {name}: {path} not present", flush=True)
            continue
        rows = load_jsonl(path)
        if args.limit:
            rows = rows[:args.limit]
        splits.append((name, rows))

    sys_prompts = build_system_prompts(os.path.join(args.data_dir, "legal_provisions.json"),
                                       dev_path=os.path.join(args.data_dir, "dev.jsonl"))

    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              enable_prefix_caching=True, gpu_memory_utilization=args.gpu_mem_util,
              tensor_parallel_size=args.tp, seed=SEED)

    written = []
    for variant in args.variants.split(","):
      for level in [int(x) for x in args.levels.split(",")]:
        for task in args.tasks.split(","):
            for split, rows in splits:
                tag = f"{args.member}{variant}" + (f"L{level}" if level != 4 else "")
                ckpt = os.path.join(args.ckpt_dir, f"preds_{tag}_{task}_{split}.jsonl")
                done = run_block(llm, sys_prompts[(task, variant)], rows, task, ckpt, args.chunk, level,
                                 args.thinking)
                written.append(write_npz(args.preds_dir, tag, task, split, rows, done))
    print("ALL DONE:\n" + "\n".join(written), flush=True)


if __name__ == "__main__":
    main()
