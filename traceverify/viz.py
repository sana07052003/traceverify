"""Publication-quality figures (white background, 200 dpi, large fonts, one PNG per output)."""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from . import config

DPI = 200
FAMILY_COLORS = {"Classical ML": "#4C78A8", "Deep learning": "#E45756", "Ensemble": "#54A24B"}
CLASS_COLORS = {"nei": "#B08D3E", "refuted": "#B5483A", "supported": "#2E7D5B"}
POS, NEG = "#2E7D5B", "#B5483A"

plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                     "legend.frameon": False, "figure.facecolor": "white"})


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _wrap(text, width=95) -> str:
    return "\n".join(textwrap.wrap(str(text), width)) or " "


def _names(labels=config.LABELS):
    return [config.DISPLAY[l] for l in labels]


# --------------------------------------------------------------------------------------- dataset
def dataset_overview(frames: dict, out) -> Path:
    splits = list(frames)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    w = 0.26
    for i, l in enumerate(config.LABELS):
        vals = [int((frames[s]["label"] == l).sum()) for s in splits]
        bars = axes[0].bar(np.arange(len(splits)) + (i - 1) * w, vals, w, color=CLASS_COLORS[l], label=config.DISPLAY[l])
        axes[0].bar_label(bars, fontsize=8, padding=2)
    axes[0].set_xticks(range(len(splits)), splits)
    axes[0].set(title="Label distribution per split", ylabel="Claims")
    axes[0].legend()
    with_img = [int((frames[s]["image"].apply(len) > 0).sum()) for s in splits]
    without = [len(frames[s]) - a for s, a in zip(splits, with_img)]
    axes[1].bar(splits, with_img, color="#4C78A8", label="has image evidence")
    axes[1].bar(splits, without, bottom=with_img, color="#BBBBBB", label="text only")
    for i, (a, b) in enumerate(zip(with_img, without)):
        axes[1].text(i, a / 2, f"{a}", ha="center", color="white", fontsize=9)
        axes[1].text(i, a + b / 2, f"{b}", ha="center", fontsize=9)
    axes[1].set(title="Visual modality availability", ylabel="Claims")
    axes[1].legend()
    n_ev = frames["train"]["text_evidence"].apply(len)
    axes[2].hist(n_ev, bins=range(1, int(min(n_ev.max(), 15)) + 2), color="#F58518", edgecolor="white", align="left")
    axes[2].set(title="Text-evidence passages per claim (train)", xlabel="Passages", ylabel="Claims")
    fig.suptitle("MOCHEG dataset overview (official splits)", fontsize=14, fontweight="bold", y=1.03)
    return _save(fig, out)


# --------------------------------------------------------------------------------------- classification
def confusion_matrix_fig(cm, title, out, labels=config.LABELS) -> Path:
    cm = np.asarray(cm).astype(int)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = _names(labels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, data, fmt, sub in ((axes[0], cm, "d", "Counts"), (axes[1], norm, ".1%", "Row-normalised (recall per class)")):
        ax.imshow(data, cmap="Blues", vmin=0, vmax=max(data.max(), 1e-9))
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=13,
                        color="white" if data[i, j] > data.max() / 2 else "black")
        ax.set(title=sub, xlabel="Predicted label", ylabel="True label")
    fig.suptitle(f"Confusion matrix: {title}", fontsize=14, fontweight="bold", y=1.02)
    return _save(fig, out)


def confusion_grid(cms: dict, out, labels=config.LABELS) -> Path:
    n = len(cms)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), squeeze=False)
    names = _names(labels)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, cm) in zip(axes.ravel(), cms.items()):
        cm = np.asarray(cm)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.axis("on")
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=20)
        ax.set_yticks(range(len(names)), names)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", color="white" if norm[i, j] > .5 else "black")
        ax.set_title(title, fontsize=11)
    fig.suptitle("Normalised confusion matrices (rows = true label)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


def model_comparison(rows: list[dict], out) -> Path:
    rows = sorted(rows, key=lambda r: r["macro_f1"])
    fig, ax = plt.subplots(figsize=(10.5, 0.75 * len(rows) + 2.2))
    y, h = np.arange(len(rows)), 0.38
    cols = [FAMILY_COLORS.get(r["family"], "#888") for r in rows]
    b1 = ax.barh(y + h / 2, [r["accuracy"] for r in rows], h, color=cols)
    b2 = ax.barh(y - h / 2, [r["macro_f1"] for r in rows], h, color=cols, alpha=0.5)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
    ax.axvline(1 / 3, color="#666", ls="--", lw=1)
    ax.text(1 / 3, len(rows) - 0.35, " chance (3 classes)", color="#666", fontsize=9, va="bottom")
    ax.set_yticks(y, [r["name"] for r in rows])
    ax.set_xlim(0, min(1.0, max(r["accuracy"] for r in rows) + 0.15))
    ax.set(xlabel="Score on MOCHEG test split", title="Model comparison (solid = accuracy, light = macro-F1)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    ax.legend(handles, list(FAMILY_COLORS), loc="lower right")
    return _save(fig, out)


def per_class_f1(rows: list[dict], out, labels=config.LABELS) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    n = len(rows)
    w = 0.8 / n
    for i, r in enumerate(rows):
        vals = [r["per_class_f1"][l] for l in labels]
        bars = ax.bar(np.arange(len(labels)) + (i - (n - 1) / 2) * w, vals, w, color=FAMILY_COLORS.get(r["family"], "#888"),
                      alpha=0.55 + 0.45 * (i % 2), label=r["name"], edgecolor="white")
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)
    ax.set_xticks(range(len(labels)), _names(labels))
    ax.set(ylabel="F1-score", ylim=(0, 1), title="Per-class F1 by model (test split)")
    ax.legend(ncol=2, fontsize=9, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, out)


def roc_curves(y_idx, probs: dict, out_dir, labels=config.LABELS):
    """Per-class ROC of the best model + micro-average ROC of every model. Returns (AUC table, best name)."""
    out_dir = Path(out_dir)
    y_bin = label_binarize(y_idx, classes=list(range(len(labels))))
    aucs = {n: {"macro": float(roc_auc_score(y_bin, p, average="macro")),
                "micro": float(roc_auc_score(y_bin, p, average="micro"))} for n, p in probs.items()}
    best = max(aucs, key=lambda n: aucs[n]["macro"])
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    for k, l in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, k], probs[best][:, k])
        ax.plot(fpr, tpr, lw=2.2, color=CLASS_COLORS[l], label=f"{config.DISPLAY[l]} (AUC {roc_auc_score(y_bin[:, k], probs[best][:, k]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#888", label="chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"One-vs-rest ROC: {best}\nmacro AUC {aucs[best]['macro']:.3f}")
    ax.legend(loc="lower right")
    _save(fig, out_dir / "07_roc_curve_best_model.png")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for n, p in probs.items():
        fpr, tpr, _ = roc_curve(y_bin.ravel(), p.ravel())
        ax.plot(fpr, tpr, lw=2, label=f"{n} (AUC {aucs[n]['micro']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#888")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Micro-average ROC, all models")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out_dir / "07b_roc_curves_all_models.png")
    return aucs, best


def confidence_hist(probs, y_idx, name, out) -> Path:
    probs = np.asarray(probs)
    conf, pred = probs.max(1), probs.argmax(1)
    ok = pred == np.asarray(y_idx)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    bins = np.linspace(conf.min(), 1.0, 21)
    ax.hist(conf[ok], bins=bins, alpha=0.75, color=POS, label=f"correct ({ok.sum()})")
    ax.hist(conf[~ok], bins=bins, alpha=0.75, color=NEG, label=f"wrong ({(~ok).sum()})")
    ax.set(xlabel="Prediction confidence", ylabel="Claims", title=f"Confidence distribution: {name}")
    ax.legend()
    return _save(fig, out)


def top_confusions(y_true, y_pred, name, out, labels=config.LABELS, top: int = 6) -> Path:
    pairs = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(t, p)] = pairs.get((t, p), 0) + 1
    items = sorted(pairs.items(), key=lambda kv: kv[1])[-top:]
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * max(len(items), 2) + 1.8))
    ax.barh([f"{config.DISPLAY[t]}  →  {config.DISPLAY[p]}" for (t, p), _ in items], [c for _, c in items], color=NEG)
    ax.bar_label(ax.containers[0], padding=3)
    ax.set(xlabel="Misclassified claims", title=f"Most frequent errors (true → predicted): {name}")
    return _save(fig, out)


# --------------------------------------------------------------------------------------- explainability
def shap_global(g: dict, out, top: int = 12) -> Path:
    classes = [c for c in config.LABELS if c in g]
    fig, axes = plt.subplots(1, len(classes), figsize=(5.4 * len(classes), 5.2))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, classes):
        items = g[c][:top][::-1]
        ax.barh([t for t, _ in items], [v for _, v in items], color=CLASS_COLORS[c])
        ax.set(title=config.DISPLAY[c], xlabel="mean |SHAP value|")
    fig.suptitle("Global word importance (SHAP, Logistic Regression, validation split)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    return _save(fig, out)


def modality_contribution(d: dict, out) -> Path:
    """d: mean P(true label) for 'all_inputs', 'text_only' (image masked), 'image_only' (text blanked), n."""
    full, txt, img = d["all_inputs"], d["text_only"], d.get("image_only")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = ["All inputs\n(text + evidence + image)", "Image masked\n(text + evidence)"] + (["Text blanked\n(image only)"] if img is not None else [])
    vals = [full, txt] + ([img] if img is not None else [])
    bars = axes[0].bar(labels, vals, color=["#54A24B", "#4C78A8", "#F58518"][: len(vals)])
    axes[0].bar_label(bars, fmt="%.3f", padding=3)
    axes[0].set(ylabel="Mean P(true label)", ylim=(0, max(vals) * 1.25), title="Deep model under input ablation")
    contrib = {"Image": full - txt}
    if img is not None:
        contrib["Text + evidence"] = full - img
    bars = axes[1].bar(list(contrib), list(contrib.values()), color=[POS if v >= 0 else NEG for v in contrib.values()])
    axes[1].bar_label(bars, fmt="%+.3f", padding=3)
    axes[1].axhline(0, color="#666", lw=1)
    axes[1].set(ylabel="Drop in P(true label) when removed", title=f"Contribution of each modality (n = {d.get('n', '?')} claims with images)")
    fig.tight_layout()
    return _save(fig, out)


def retrieval_recall(ret: dict, out) -> Path:
    ks = sorted(ret["dense"], key=int)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, key, c in (("Dense (MiniLM + FAISS)", "dense", "#4C78A8"), ("Hybrid (dense + BM25)", "hybrid", "#F58518")):
        ax.plot([int(k) for k in ks], [ret[key][k] for k in ks], "o-", lw=2.2, color=c, label=name)
        for k in ks:
            ax.annotate(f"{ret[key][k]:.2f}", (int(k), ret[key][k]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set(xlabel="k (retrieved passages)", ylabel="Recall@k", ylim=(0, 1.05), title="Evidence retrieval quality", xticks=[int(k) for k in ks])
    ax.legend()
    return _save(fig, out)


def training_curves(history: list[dict], out) -> Path:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(ep, [h["train_loss"] for h in history], "o-", color="#E45756", lw=2)
    axes[0].set(xlabel="Epoch", ylabel="Training loss", title="Deep multimodal network: training loss", xticks=ep)
    axes[1].plot(ep, [h["val"]["accuracy"] for h in history], "o-", color="#4C78A8", lw=2, label="Validation accuracy")
    axes[1].plot(ep, [h["val"]["macro_f1"] for h in history], "s-", color="#54A24B", lw=2, label="Validation macro-F1")
    axes[1].set(xlabel="Epoch", ylabel="Score", title="Validation performance", xticks=ep)
    axes[1].legend()
    fig.tight_layout()
    return _save(fig, out)


def grouped_metrics(items: dict, title: str, out) -> Path:
    """items: {label: {'accuracy':..,'macro_f1':..}}"""
    names = list(items)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.9 * len(names) + 2), 4.8))
    x, w = np.arange(len(names)), 0.36
    b1 = ax.bar(x - w / 2, [items[n]["accuracy"] for n in names], w, color="#4C78A8", label="Accuracy")
    b2 = ax.bar(x + w / 2, [items[n]["macro_f1"] for n in names], w, color="#54A24B", label="Macro-F1")
    ax.bar_label(b1, fmt="%.3f", fontsize=9, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=9, padding=2)
    ax.set_xticks(x, [n.replace("_", " ") for n in names])
    ax.set(ylim=(0, 1), title=title, ylabel="Score")
    ax.legend()
    return _save(fig, out)


def ensemble_weights(w: dict, names: dict, out) -> Path:
    items = sorted(w.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8.2, 0.65 * len(items) + 1.8))
    bars = ax.barh([names.get(k, k) for k, _ in items], [v for _, v in items], color="#54A24B")
    ax.bar_label(bars, labels=[f"{v:.1%}" for _, v in items], padding=3)
    ax.set(xlabel="Share of meta-learner weight", title="How much the stacking ensemble trusts each model", xlim=(0, max(v for _, v in items) * 1.2))
    return _save(fig, out)


def _bars(ax, pairs, title, note):
    if not pairs:
        ax.axis("off")
        ax.text(0.5, 0.5, "not available", ha="center", va="center", color="#888")
        ax.set_title(title)
        return
    pairs = pairs[::-1]
    ax.barh([p[0] for p in pairs], [p[1] for p in pairs], color=[POS if p[1] >= 0 else NEG for p in pairs])
    ax.axvline(0, color="#666", lw=0.8)
    ax.set(title=title, xlabel=note)


def example_panel(res: dict, out, image=None) -> Path:
    """One verified claim: verdict, SHAP, occlusion, modality ablation, evidence (and the image if given)."""
    v = res["verdict"]
    col = CLASS_COLORS[v["label"]]
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.55, 1.5, 1.1], hspace=0.55, wspace=0.55)
    ax = fig.add_subplot(gs[0, :])
    ax.axis("off")
    ax.text(0, 0.95, "CLAIM", fontsize=9, color="#888", va="top")
    ax.text(0, 0.75, _wrap(res["claim"], 130), fontsize=13, va="top")
    ax.text(0.99, 0.95, f"{config.DISPLAY[v['label']].upper()}  ·  {v['confidence']:.0%}", fontsize=20, fontweight="bold",
            color=col, ha="right", va="top")
    ax.text(0.99, 0.35, f"decided by {v['decided_by'].replace('_', ' ')}", fontsize=9, color="#666", ha="right")
    xai = res.get("xai", {})
    lab = config.DISPLAY[v["label"]]
    _bars(fig.add_subplot(gs[1, 0]), xai.get("shap_terms"), "SHAP (Logistic Regression)", f"+ pushes toward '{lab}'")
    _bars(fig.add_subplot(gs[1, 1]), xai.get("occlusion_terms"), "Occlusion (deep network)", f"P('{lab}') drop if word removed")
    ab = xai.get("modality_ablation")
    a3 = fig.add_subplot(gs[1, 2])
    if ab:
        names = [k.replace("_", " ") for k in ab]
        bars = a3.bar(names, list(ab.values()), color=["#54A24B", "#4C78A8", "#F58518"][: len(ab)])
        a3.bar_label(bars, fmt="%.2f", padding=2)
        a3.set(title="Modality ablation", ylabel=f"P('{lab}')", ylim=(0, 1.1))
        a3.tick_params(axis="x", labelrotation=15)
    else:
        a3.axis("off")
        a3.text(0.5, 0.5, "not available", ha="center", color="#888")
    ax = fig.add_subplot(gs[2, :2])
    ax.axis("off")
    ax.text(0, 1.0, "RETRIEVED EVIDENCE", fontsize=9, color="#888", va="top")
    txt = "\n\n".join(f"[{i + 1}] (sim {e['score']:.2f}) " + _wrap(e["text"][:330] + ("..." if len(e["text"]) > 330 else ""), 120)
                      for i, e in enumerate(res.get("evidence", [])[:2]))
    ax.text(0, 0.88, txt, fontsize=9.5, va="top")
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    if image is not None:
        ax.imshow(image)
        ax.set_title(f"Image · CLIP sim {res.get('clip_similarity', float('nan')):.2f} ({res.get('clip_band', '')})", fontsize=10)
    return _save(fig, out)


# --------------------------------------------------------------------------------------- architecture
def architecture(out) -> Path:
    fig, ax = plt.subplots(figsize=(17, 9.2))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF1FB", ec="#4C78A8", fs=11, bold=False, tc="black"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=tc)

    def arrow(x1, y1, x2, y2, c="#555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=c))

    ax.text(8.5, 8.85, "TraceVerify: explainable multimodal misinformation verification (MOCHEG)", ha="center", fontsize=16, fontweight="bold")
    box(0.3, 5.0, 2.6, 2.6, "INPUT\n\nClaim text\n+\nImage (optional)", "#FFF4E0", "#F58518", 12, True)
    box(3.7, 6.3, 3.3, 1.5, "Evidence retrieval\nMiniLM + FAISS · BM25 · hybrid\nover MOCHEG evidence", "#EAF1FB", "#4C78A8")
    box(3.7, 3.7, 3.3, 1.5, "CLIP image encoder\n(ViT-B/32)\nimage and claim embeddings", "#EAF1FB", "#4C78A8")
    ax.add_patch(FancyBboxPatch((7.7, 2.6), 3.6, 5.5, boxstyle="round,pad=0.02,rounding_size=0.2", fc="#F7F7F7", ec="#999", lw=1.5, ls="--"))
    ax.text(9.5, 7.8, "CLASSIFIERS", ha="center", fontsize=11, fontweight="bold", color="#555")
    ys = [6.7, 5.8, 4.9, 4.0]
    for yy, t in zip(ys, ["Logistic Regression", "Linear SVM", "Naive Bayes", "Random Forest"]):
        box(8.0, yy, 3.0, 0.7, t, "#E8F0FA", "#4C78A8", 10.5)
    box(8.0, 2.8, 3.0, 1.0, "Deep multimodal network\nDistilBERT + CLIP fusion", "#FBE6E6", "#E45756", 10.5, True)
    box(12.0, 4.7, 2.1, 1.6, "Stacking\nensemble\n(meta-learner)", "#E6F4E6", "#54A24B", 11, True)
    box(14.6, 4.7, 2.1, 1.6, "VERDICT\nSupported /\nRefuted / NEI\n+ confidence", "#E6F4E6", "#2E7D5B", 10.5, True)
    def box2(x, y, w, h, title, body, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.18", fc=fc, ec=ec, lw=1.8))
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + (h - 0.5) / 2, body, ha="center", va="center", fontsize=10.5)

    box2(0.3, 0.3, 7.0, 1.6, "EXPLAINABLE AI", "SHAP word attribution  ·  Occlusion  ·  Modality ablation\nCLIP image-claim consistency", "#F3E9F7", "#8E5BA8")
    box2(7.9, 0.3, 8.8, 1.6, "RAG EXPLANATION", "LLM writes a justification restricted to the retrieved evidence passages [1], [2], ...\nuses the verdict + evidence; says so when the evidence is insufficient", "#F3E9F7", "#8E5BA8")
    arrow(2.9, 6.6, 3.7, 7.0)
    arrow(2.9, 5.7, 3.7, 4.7)
    arrow(7.0, 7.0, 7.7, 7.0)
    arrow(7.0, 4.5, 8.0, 3.5)
    arrow(11.3, 5.5, 12.0, 5.5)
    arrow(14.1, 5.5, 14.6, 5.5)
    arrow(15.65, 4.7, 15.65, 1.9, "#8E5BA8")
    arrow(8.4, 2.6, 6.3, 1.95, "#8E5BA8")
    return _save(fig, out)
