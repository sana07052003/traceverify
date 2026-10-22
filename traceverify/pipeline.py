"""End-to-end verification: text and/or image in -> verdict, per-model votes, evidence, XAI out."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import baseline, classical, config, llm
from .clip_encoder import ClipEncoder, consistency_band
from .ensemble import StackingEnsemble
from .explain import LinearExplainer
from .retrieval import ClaimImageIndex, Retriever


class TraceVerify:
    def __init__(self, artifact_dir: Path = config.ARTIFACT_DIR):
        ad = Path(artifact_dir)
        self.retriever = Retriever.load(ad / "rag")
        self.by_claim = self.retriever.lookup.groupby("claim_id")["text"].apply(list).to_dict()
        self.claim_index = ClaimImageIndex.load(ad / "rag") if ClaimImageIndex.exists(ad / "rag") else None

        zoo_path = ad / "classical" / "models.joblib"
        self.zoo = classical.load(zoo_path) if zoo_path.exists() else {}
        base_path = ad / "baseline" / "tfidf_lr.joblib"
        self.explainer = LinearExplainer(baseline.load(base_path)) if base_path.exists() else None

        self.deep = None
        if (ad / "deep_model" / "head.pt").exists():
            from .mm_model import MultimodalVerifier  # imports torch only when the model exists
            self.deep = MultimodalVerifier(ad / "deep_model")
        stack_path = ad / "ensemble" / "stack.joblib"
        self.ensemble = StackingEnsemble.load(stack_path) if stack_path.exists() else None
        self._clip = None

    @property
    def clip(self) -> ClipEncoder:
        if self.deep is not None:
            return self.deep.clip
        if self._clip is None:
            self._clip = ClipEncoder()
        return self._clip

    # ------------------------------------------------------------------
    def related_claims(self, img_vec: np.ndarray, k: int = 5) -> list[dict]:
        if self.claim_index is None:
            return []
        out = self.claim_index.search(img_vec, k)
        for r in out:
            r["evidence"] = self.by_claim.get(r["claim_id"], [])[:2]
        return out

    def verify(self, claim: str = "", image=None, top_k: int = 5, use_llm: bool = True,
               api_key: str | None = None) -> dict:
        claim = (claim or "").strip()
        has_text, has_image = bool(claim), image is not None
        if not has_text and not has_image:
            raise ValueError("Provide a claim, an image, or both.")
        res: dict = {"claim": claim, "mode": ("text+image" if has_text and has_image
                                              else "image" if has_image else "text"), "warnings": []}

        img_vec = self.clip.images([image])[0] if has_image else None
        if has_image:
            res["related_claims"] = self.related_claims(img_vec)
        if not has_text:
            res["verdict"] = None
            res["warnings"].append("Image-only input: showing the most visually/semantically related "
                                   "fact-checked claims. Add a claim to get a verdict.")
            return res

        evidence = self.retriever.search(claim, k=top_k)
        res["evidence"] = evidence
        if not evidence or evidence[0]["score"] < config.LOW_RELEVANCE_THRESHOLD:
            res["warnings"].append("No strongly relevant evidence was found in the MOCHEG corpus; "
                                   "treat the verdict as low reliability.")

        # ---- every model votes
        member_probs: dict[str, np.ndarray] = {}
        for name, pipe in self.zoo.items():
            member_probs[name] = classical.proba_matrix(pipe, [claim])[0]
        ev_text = " ".join(e["text"] for e in evidence[:config.CLASSIFIER_EVIDENCE])
        feats = None
        if self.deep is not None:
            zeros = np.zeros(config.CLIP_DIM, "float32")
            txt_vec = self.clip.texts([claim])[0] if has_image else zeros
            feats = (img_vec if has_image else zeros, txt_vec, has_image)
            member_probs["deep_multimodal"] = self.deep.predict(claim, ev_text, feats)["probs"]
            member_probs["deep_multimodal"] = np.array([member_probs["deep_multimodal"][l] for l in config.LABELS])
            if has_image:
                res["clip_similarity"] = float(img_vec @ txt_vec)
                res["clip_band"] = consistency_band(res["clip_similarity"])

        members = []
        for name, p in member_probs.items():
            k = int(np.argmax(p))
            members.append({"model": name, "name": classical.NAMES[name], "label": config.LABELS[k],
                            "confidence": float(p[k]), "probs": {l: float(v) for l, v in zip(config.LABELS, p)}})
        res["members"] = members

        # ---- final verdict: stacking ensemble > deep network > logistic regression
        final_name, final_p = None, None
        if self.ensemble is not None and all(n in member_probs for n in self.ensemble.member_names):
            final_p = self.ensemble.predict_proba(member_probs)[0]
            final_name = "stacking_ensemble"
            res["ensemble_weights"] = self.ensemble.member_weights()
        elif "deep_multimodal" in member_probs:
            final_p, final_name = member_probs["deep_multimodal"], "deep_multimodal"
        elif "logistic_regression" in member_probs:
            final_p, final_name = member_probs["logistic_regression"], "logistic_regression"
        if final_p is None:
            raise RuntimeError("No classifier artifacts found. Train the models first.")
        k = int(np.argmax(final_p))
        res["verdict"] = {"label": config.LABELS[k], "confidence": float(final_p[k]), "decided_by": final_name,
                          "probs": {l: float(v) for l, v in zip(config.LABELS, final_p)}}
        label = config.LABELS[k]

        # ---- explainable AI
        xai: dict = {}
        if self.explainer is not None:
            xai["shap_terms"] = self.explainer.local(claim, label)
        if self.deep is not None and feats is not None:
            xai["occlusion_terms"] = self.deep.occlusion(claim, ev_text, feats, label)
            xai["modality_ablation"] = self.deep.modality_ablation(claim, ev_text, feats, label)
        res["xai"] = xai

        # ---- RAG explanation
        note = (f"CLIP image-claim similarity {res['clip_similarity']:.2f} ({res['clip_band']})"
                if "clip_similarity" in res else None)
        res["explanation"] = (llm.explain(claim, label, evidence, api_key, note) if use_llm
                              else {"text": llm.fallback_explanation(label, evidence, "LLM disabled"),
                                    "source": "fallback"})
        return res
