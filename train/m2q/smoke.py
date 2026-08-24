"""Smoke: load Qwen3-14B seq-cls + LoRA, tokenize 8 real rows, one train step per task head shape."""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, SEED, load_data, MAX_LEN
from train_m2q import TextDS, build_model, set_seed, st1_class_weights

def main():
    repo = os.path.expanduser("~/childsafeads")
    set_seed(SEED)
    train_rows, _ = load_data(repo)
    rows = train_rows[:8]
    task = "st3"
    t0 = time.time()
    tok, model = build_model(len(LABELS[task]), task)
    model.to("cuda")
    model.gradient_checkpointing_enable()  # match the real job (job trains micro-batch 2 + ckpt)
    print(f"[smoke] model loaded in {time.time()-t0:.0f}s", flush=True)
    ds = TextDS(rows, task, tok)
    batch = ds.collate(list(range(2)))  # micro-batch 2, as the job runs
    print(f"[smoke] tokenized 2 rows (job micro-batch), shape {tuple(batch['input_ids'].shape)} (max_len {MAX_LEN})", flush=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    model.train()
    logits = model(input_ids=batch["input_ids"].cuda(),
                   attention_mask=batch["attention_mask"].cuda()).logits.float()
    loss = F.binary_cross_entropy_with_logits(logits, batch["labels"].cuda())
    loss.backward()
    opt.step()
    print(f"[smoke] one train step OK, loss={loss.item():.4f}, "
          f"mem={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    print("[smoke] st1 class weights:", st1_class_weights(train_rows).tolist(), flush=True)
    print("SMOKE PASS", flush=True)

if __name__ == "__main__":
    main()
