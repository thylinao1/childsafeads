"""Mac CPU smoke: data loading, field budgets, tokenization, label matrices, metric,
collate + a forward/backward through a tiny encoder (bert-tiny stands in for the trunk;
real-arch load is exercised by the cluster smoke job)."""
import json
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from common import (BUDGETS, TASKS, FieldDropper, build_text, extract_fields, label_arrays,
                    load_jsonl, naive_score)

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
train_rows = load_jsonl(f"{REPO}/data/train.jsonl")[:20]
dev_rows = load_jsonl(f"{REPO}/data/dev.jsonl")[:5]
folds = json.load(open(f"{REPO}/eval/folds.json"))
assert all(r["channel_context"]["channelID"] in folds for r in train_rows)

for arch, tok_name in [("deberta", "microsoft/deberta-v3-large"),
                       ("modernbert", "answerdotai/ModernBERT-large")]:
    tok = AutoTokenizer.from_pretrained(tok_name)
    b = BUDGETS[arch]
    lens = []
    for r in train_rows + dev_rows:
        f = extract_fields(r, tok, b)
        assert f["disclosure"] in ("true", "false", "unknown")
        text = build_text(f)
        n = len(tok(text, truncation=True, max_length=b["max_len"])["input_ids"])
        lens.append(n)
    print(f"[{arch}] token lens min/med/max = {min(lens)}/{int(np.median(lens))}/{max(lens)}"
          f" (cap {b['max_len']})")
    assert max(lens) <= b["max_len"]

# field dropout determinism
d1, d2 = FieldDropper(7), FieldDropper(7)
assert [d1.sample() for _ in range(50)] == [d2.sample() for _ in range(50)]
drops = sum(len(FieldDropper(1).sample()) for _ in range(1))  # smoke only
print("[dropout] deterministic OK")

y1, y2, y3 = label_arrays(train_rows)
assert y2.shape == (20, 12) and y3.shape == (20, 8)
sc, parts = naive_score(y1, y2, y3,
                        np.eye(5, dtype=np.float32)[y1] * 0.9 + 0.02, y2 * 0.9 + 0.05,
                        y3 * 0.9 + 0.05)
assert abs(sc - 1.0) < 1e-9, sc  # perfect probs -> perfect naive score
print("[metric] perfect-prediction sanity =", sc)

# forward/backward with tiny trunk through our head/loss/collate path
from model import MultiHeadEncoder, make_losses
from train import TrainDS, collate
from torch.utils.data import DataLoader

tok = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
fields = [extract_fields(r, tok, BUDGETS["deberta"]) for r in train_rows]
ds = TrainDS(fields, y1, y2, y3, tok, 512, FieldDropper(0))
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate(tok),
                    generator=torch.Generator().manual_seed(0))
model = MultiHeadEncoder("prajjwal1/bert-tiny")
ce, b2, b3 = make_losses(y1, y2, y3, "cpu")
opt = torch.optim.AdamW(model.param_groups(1e-5, 5e-5))
for i, batch in enumerate(loader):
    if i >= 2:
        break
    l1, l2, l3 = model(batch["input_ids"], batch["attention_mask"])
    assert l1.shape[1] == 5 and l2.shape[1] == 12 and l3.shape[1] == 8
    loss = ce(l1, batch["y1"]) + b2(l2, batch["y2"]) + b3(l3, batch["y3"])
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"[fwd/bwd] batch {i} loss={loss.item():.4f}")
print("SMOKE-LOCAL PASS")
