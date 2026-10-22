"""Deep-learning component: multimodal fusion network (PyTorch).

    claim + evidence --> DistilBERT --> [CLS] vector (768) --+
                                                              +--> MLP --> {nei, refuted, supported}
    image --> CLIP image emb (512) + claim CLIP emb (512)     |
        --> [img, img*txt, cos(img,txt)] --> MLP (256) -------+   (zeroed when no image is given)

Trained with modality dropout, so the same network handles text-only and text+image inputs.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (AutoModel, AutoTokenizer, DataCollatorWithPadding,
                          get_linear_schedule_with_warmup)

from . import config
from .baseline import metrics as compute_metrics


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionHead(nn.Module):
    def __init__(self, text_dim: int, clip_dim: int = config.CLIP_DIM, hidden: int = 256,
                 n_classes: int = len(config.LABELS), dropout: float = 0.2):
        super().__init__()
        self.scale = math.sqrt(clip_dim)   # unit-norm CLIP vectors have ~1/sqrt(d) per-element magnitude; rescale to ~1
        self.img_branch = nn.Sequential(nn.Linear(clip_dim * 2 + 1, hidden), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(text_dim + hidden, hidden),
                                        nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, text_vec, img, txt, has_img):
        has = has_img.unsqueeze(-1)
        cos = (img * txt).sum(-1, keepdim=True)
        img_s, txt_s = img * self.scale, txt * self.scale
        v = torch.cat([img_s, img_s * txt_s, cos * 10.0], dim=-1) * has
        return self.classifier(torch.cat([text_vec, self.img_branch(v) * has], dim=-1))


class MultimodalNet(nn.Module):
    def __init__(self, encoder, hidden: int = 256):
        super().__init__()
        self.encoder = encoder
        self.head = FusionHead(encoder.config.hidden_size, hidden=hidden)

    def forward(self, input_ids, attention_mask, img, txt, has_img):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.head(h, img, txt, has_img)


@torch.no_grad()
def batch_probs(net, tokenizer, claims, evidences, img, txt, has, device,
                batch_size: int = 32, max_len: int = config.MAX_LEN) -> np.ndarray:
    net.eval()
    claims, evidences = list(claims), list(evidences)
    out = []
    for s in range(0, len(claims), batch_size):
        e = slice(s, s + batch_size)
        enc = tokenizer(claims[e], evidences[e], truncation="longest_first", max_length=max_len,
                        padding=True, return_tensors="pt")
        logits = net(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device),
                     img=torch.as_tensor(img[e], dtype=torch.float32, device=device),
                     txt=torch.as_tensor(txt[e], dtype=torch.float32, device=device),
                     has_img=torch.as_tensor(np.asarray(has[e], dtype="float32"), device=device))
        out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, len(config.LABELS)), dtype="float32")


class _Collate:
    def __init__(self, tokenizer):
        self.pad = DataCollatorWithPadding(tokenizer)

    def __call__(self, items):
        batch = self.pad([{"input_ids": i["input_ids"], "attention_mask": i["attention_mask"]} for i in items])
        batch["img"] = torch.as_tensor(np.stack([i["img"] for i in items]), dtype=torch.float32)
        batch["txt"] = torch.as_tensor(np.stack([i["txt"] for i in items]), dtype=torch.float32)
        batch["has_img"] = torch.as_tensor([float(i["has"]) for i in items], dtype=torch.float32)
        batch["labels"] = torch.as_tensor([i["label"] for i in items], dtype=torch.long)
        return batch


def _items(tokenizer, claims, evidences, img, txt, has, labels, max_len):
    enc = tokenizer(list(claims), list(evidences), truncation="longest_first", max_length=max_len)
    return [{"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i],
             "img": img[i], "txt": txt[i], "has": bool(has[i]), "label": int(labels[i])}
            for i in range(len(claims))]


def train(train_df, train_ev, train_feats, val_df, val_ev, val_feats, out_dir: Path,
          base_model: str = config.DEEP_BASE_MODEL, epochs: int = 3, lr: float = 3e-5,
          batch_size: int = 16, max_len: int = config.MAX_LEN, image_dropout: float = 0.3, head_lr: float = 5e-4,
          warmup_ratio: float = 0.1, log=print) -> dict:
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = pick_device()
    use_amp = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    net = MultimodalNet(AutoModel.from_pretrained(base_model)).to(device)

    y_tr = train_df["label"].map(config.LABEL2ID).to_numpy()
    y_va = val_df["label"].map(config.LABEL2ID).to_numpy()
    counts = np.bincount(y_tr, minlength=len(config.LABELS)).astype(float)
    weights = torch.tensor(len(y_tr) / (len(config.LABELS) * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    items = _items(tokenizer, train_df["claim"], train_ev, *train_feats, y_tr, max_len)
    loader = DataLoader(items, batch_size=batch_size, shuffle=True, collate_fn=_Collate(tokenizer))
    # the freshly initialised fusion head needs a much larger learning rate than the pretrained encoder
    opt = torch.optim.AdamW([{"params": net.encoder.parameters(), "lr": lr},
                             {"params": net.head.parameters(), "lr": head_lr}], weight_decay=0.01)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(warmup_ratio * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(out_dir)
    best, history = -1.0, []
    for epoch in range(1, epochs + 1):
        net.train()
        t0, running = time.time(), 0.0
        for step, b in enumerate(loader, 1):
            b = {k: v.to(device) for k, v in b.items()}
            has = b["has_img"] * (torch.rand_like(b["has_img"]) > image_dropout).float()  # modality dropout
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = net(b["input_ids"], b["attention_mask"], b["img"], b["txt"], has)
            loss = loss_fn(logits.float(), b["labels"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 100 == 0:
                log(f"epoch {epoch} step {step}/{len(loader)} loss {running / step:.4f}")
        probs = batch_probs(net, tokenizer, val_df["claim"], val_ev, *val_feats, device, max_len=max_len)
        m = compute_metrics([config.ID2LABEL[i] for i in y_va], [config.ID2LABEL[i] for i in probs.argmax(1)])
        history.append({"epoch": epoch, "train_loss": running / len(loader), "val": m,
                        "seconds": round(time.time() - t0, 1)})
        log(f"epoch {epoch}: val acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f}")
        has_v = np.asarray(val_feats[2], dtype=bool)
        if has_v.any():    # is the image branch actually used?
            p0 = batch_probs(net, tokenizer, val_df["claim"], val_ev, val_feats[0], val_feats[1],
                             np.zeros_like(has_v), device, max_len=max_len)
            flips = float((p0.argmax(1) != probs.argmax(1))[has_v].mean())
            history[-1]["image_flip_rate"] = flips
            log(f"  image effect: {flips:.1%} of validation predictions on image claims change when the image is masked")
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            save_model(net, tokenizer, out_dir, {"base_model": base_model, "max_len": max_len})
    (out_dir / "training.json").write_text(json.dumps({"history": history, "best_val_macro_f1": best}, indent=2))
    return {"history": history, "best_val_macro_f1": best}


def save_model(net, tokenizer, out_dir: Path, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    net.encoder.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(net.head.state_dict(), out_dir / "head.pt")
    (out_dir / "mm_meta.json").write_text(json.dumps(meta))


class MultimodalVerifier:
    """Inference wrapper: probabilities, modality ablation, and word-occlusion attribution."""

    def __init__(self, path: Path = config.DEEP_DIR, device=None, clip=None):
        path = Path(path)
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.net = MultimodalNet(AutoModel.from_pretrained(path))
        self.net.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
        self.net.to(self.device).eval()
        meta = json.loads((path / "mm_meta.json").read_text()) if (path / "mm_meta.json").exists() else {}
        self.max_len = meta.get("max_len", config.MAX_LEN)
        self._clip = clip

    @property
    def clip(self):
        if self._clip is None:
            from .clip_encoder import ClipEncoder
            self._clip = ClipEncoder()
        return self._clip

    def encode_inputs(self, claim: str, image=None):
        """-> (img_vec, txt_vec, has) with zeros when there is no image."""
        d = config.CLIP_DIM
        if image is None:
            return np.zeros(d, "float32"), np.zeros(d, "float32"), False
        return self.clip.images([image])[0], self.clip.texts([claim])[0], True

    def probs(self, claims, evidences, img, txt, has, batch_size: int = 32) -> np.ndarray:
        return batch_probs(self.net, self.tokenizer, claims, evidences, img, txt, has,
                           self.device, batch_size, self.max_len)

    def predict(self, claim: str, evidence: str, feats) -> dict:
        img, txt, has = feats
        p = self.probs([claim], [evidence], img[None], txt[None], np.array([has]))[0]
        k = int(p.argmax())
        return {"label": config.ID2LABEL[k], "confidence": float(p[k]),
                "probs": {l: float(v) for l, v in zip(config.LABELS, p)}}

    def modality_ablation(self, claim: str, evidence: str, feats, label: str) -> dict[str, float]:
        """P(label) with all inputs / with the image masked / with the text blanked (image only probe)."""
        img, txt, has = feats
        k = config.LABEL2ID[label]
        rows = [(claim, evidence, has), (claim, evidence, False)]
        if has:
            rows.append(("", "", True))
        p = self.probs([r[0] for r in rows], [r[1] for r in rows],
                       np.stack([img] * len(rows)), np.stack([txt] * len(rows)),
                       np.array([r[2] for r in rows]))[:, k]
        out = {"all_inputs": float(p[0]), "text_only": float(p[1])}
        if has:
            out["image_only"] = float(p[2])
        return out

    def occlusion(self, claim: str, evidence: str, feats, label: str, top_n: int = 8,
                  max_words: int = 40) -> list[tuple[str, float]]:
        """Leave-one-word-out: delta_i = P(label | claim) - P(label | claim without word i)."""
        img, txt, has = feats
        words = claim.split()[:max_words]
        if not words:
            return []
        variants = [claim] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        n = len(variants)
        p = self.probs(variants, [evidence] * n, np.stack([img] * n), np.stack([txt] * n),
                       np.array([has] * n))[:, config.LABEL2ID[label]]
        deltas = [(w, float(p[0] - p[i + 1])) for i, w in enumerate(words)]
        return sorted(deltas, key=lambda d: abs(d[1]), reverse=True)[:top_n]
