"""Evidence-grounded explanation via Groq. The API key is passed in at call time; it is never
hard-coded, logged, or written anywhere."""
from __future__ import annotations

from . import config

SYSTEM = (
    "You are a careful fact-checking assistant. Use ONLY the numbered evidence passages provided. "
    "Do not use outside knowledge. Cite passages like [1], [2]. If the evidence does not actually "
    "settle the claim, or contradicts the classifier's label, say so plainly."
)


def _prompt(claim: str, label: str, evidence: list[dict], image_note: str | None) -> str:
    block = "\n".join(f"[{i + 1}] {e['text']}" for i, e in enumerate(evidence))
    note = f"\nVisual context: {image_note}\n" if image_note else ""
    return (f"Claim: {claim}\nClassifier prediction: {label}\n{note}\nEvidence:\n{block}\n\n"
            f"In 2-3 sentences, explain whether the evidence supports the prediction \"{label}\".")


def fallback_explanation(label: str, evidence: list[dict], reason: str = "") -> str:
    top = evidence[0]["text"][:300] + ("..." if len(evidence[0]["text"]) > 300 else "") if evidence else ""
    head = f"(No LLM explanation{': ' + reason if reason else ''}.) "
    return head + (f"Prediction '{label}'. Most relevant retrieved passage [1]: \"{top}\"" if top
                   else f"Prediction '{label}'; no evidence passages were retrieved.")


def explain(claim: str, label: str, evidence: list[dict], api_key: str | None,
            image_note: str | None = None, model: str = config.GROQ_MODEL) -> dict:
    if not api_key:
        return {"text": fallback_explanation(label, evidence, "no Groq API key configured"), "source": "fallback"}
    if not evidence:
        return {"text": fallback_explanation(label, evidence), "source": "fallback"}
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        kwargs = dict(model=model, max_tokens=1000,
                      messages=[{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": _prompt(claim, label, evidence, image_note)}])
        try:
            resp = client.chat.completions.create(reasoning_effort="low", **kwargs)
        except Exception:  # model without reasoning_effort support
            resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return {"text": text, "source": "llm"}
        return {"text": fallback_explanation(label, evidence, "model returned an empty response"),
                "source": "fallback"}
    except Exception as exc:  # never echo the key; only the exception class
        return {"text": fallback_explanation(label, evidence, f"LLM unavailable ({type(exc).__name__})"),
                "source": "fallback"}
