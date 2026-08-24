import torch
import torch.nn as nn
from transformers import AutoModel


class MultiHeadEncoder(nn.Module):
    """One trunk, three heads (st1 5-way CE, st2 12-way BCE, st3 8-way BCE)."""

    def __init__(self, model_name, attn_implementation=None):
        super().__init__()
        kw = {}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        # transformers v5 honors the checkpoint config's dtype by default and
        # deberta-v3-large's config says float16 (the classic DeBERTa NaN regime,
        # and a Half-vs-Float mismatch with the fp32 heads). Force fp32 weights;
        # autocast (where enabled) still gives bf16 compute. v5 uses `dtype`,
        # v4 uses `torch_dtype` - try both.
        try:
            self.trunk = AutoModel.from_pretrained(model_name, dtype=torch.float32, **kw)
        except TypeError:
            self.trunk = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, **kw)
        h = self.trunk.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_st1 = nn.Linear(h, 5)
        self.head_st2 = nn.Linear(h, 12)
        self.head_st3 = nn.Linear(h, 8)

    def forward(self, input_ids, attention_mask):
        out = self.trunk(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return self.head_st1(cls), self.head_st2(cls), self.head_st3(cls)

    def param_groups(self, lr_trunk, lr_heads, weight_decay=0.01):
        heads = list(self.head_st1.parameters()) + list(self.head_st2.parameters()) + \
                list(self.head_st3.parameters())
        return [
            {"params": [p for p in self.trunk.parameters() if p.requires_grad],
             "lr": lr_trunk, "weight_decay": weight_decay},
            {"params": heads, "lr": lr_heads, "weight_decay": weight_decay},
        ]


def make_losses(y1_train, y2_train, y3_train, device):
    """st1 CE with inverse-sqrt-freq class weights (normalized); st2/st3 BCE with
    per-class pos_weight = clip(sqrt(neg/pos), 1, 8). Stats from the fold's train split."""
    import numpy as np
    counts = np.bincount(y1_train, minlength=5).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    w = w / w.sum() * len(w)
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

    def pw(y):
        pos = y.sum(0).astype(np.float64)
        neg = len(y) - pos
        return torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1.0)), 1.0, 8.0),
                            dtype=torch.float32, device=device)

    bce2 = nn.BCEWithLogitsLoss(pos_weight=pw(y2_train))
    bce3 = nn.BCEWithLogitsLoss(pos_weight=pw(y3_train))
    return ce, bce2, bce3
