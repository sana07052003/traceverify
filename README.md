# TraceVerify 🔎
**Explainable, evidence-grounded misinformation verification for text *and* images, built on the original MOCHEG dataset** (Yao et al., SIGIR 2023 — [Zenodo](https://doi.org/10.5281/zenodo.6653771), CC BY 4.0).

Enter a claim, upload an image, or both. Several AI algorithms analyse it, a stacking ensemble decides, and every
verdict ships with word-level explanations and the retrieved fact-check evidence behind it.

## AI algorithms used (deep learning is one component, not the whole system)

| Family | Algorithm | Role |
|---|---|---|
| Classical ML | TF-IDF + **Logistic Regression**, **Linear SVM** (calibrated), **Complement Naive Bayes**, **Random Forest** | text classifiers / baselines |
| **Deep learning** | **Multimodal fusion network** (PyTorch): fine-tuned DistilBERT on *claim + evidence* fused with CLIP image features; trained with modality dropout so it also works without an image | main neural classifier |
| Information retrieval | **FAISS** dense search (MiniLM embeddings), **BM25**, **hybrid** (dense + BM25) | evidence retrieval (RAG) |
| Ensemble learning | **Stacking** meta-learner (Logistic Regression over all models' probabilities) | final verdict |
| Explainable AI | **SHAP** (linear), **occlusion** word attribution, **modality ablation**, **CLIP image-claim consistency** | explanations |
| Generative AI (RAG) | LLM via Groq, restricted to retrieved passages | natural-language justification |

```
text / image ─► retrieval (FAISS · BM25 · hybrid) ─► evidence ─┐
                                                               ├─► LogReg · SVM · NB · RF · Deep multimodal net
image ─► CLIP ─► image features + related fact-checked claims ─┘            │
                                                                    Stacking ensemble ─► verdict
                                 SHAP · occlusion · ablation · CLIP score · LLM explanation ◄──┘
```
Input modes: **text** (full verdict), **text + image** (verdict + image contribution + CLIP consistency + related
fact-checks), **image only** (related fact-checked claims; add text for a verdict).

## Quick start

```bash
git clone <your-repo> && cd traceverify
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests -q            # offline tests, no downloads needed
```

### Data: the original MOCHEG release
MOCHEG ships with **official train / val / test splits** (`<split>/Corpus2.csv`, one row per evidence piece) and
**real image files** (`<split>/images/<claim_id>-proof-<n>-<description>`), which the dataset statement lists as
an input to claim verification. The loader finds these files recursively, so the unzip folder name doesn't matter.
```bash
python -m scripts.download_mocheg --list     # see the files on Zenodo (needs internet; images are several GB)
python -m scripts.download_mocheg            # STREAMS the 41 GB archive, keeps only what is needed
# (Images are shrunk to 512 px while extracting and a completion marker is written, so run it with --out on
#  Google Drive once and never again.  Zenodo hosts one 41 GB tar.gz; the script never stores it - it extracts only Corpus2.csv + split images on the fly.
#  Expect roughly 15-60+ min depending on Zenodo's speed. Re-running resumes by skipping extracted files.)
# If that is blocked: download manually from https://doi.org/10.5281/zenodo.6653771 and unzip into data/mocheg/
```
`ruling_outline` and `Origin` (the fact-checker's verdict/article) are never read, to avoid label leakage.

### Train (GPU recommended: a free Colab T4 is enough — see `notebooks/TraceVerify_Colab.ipynb`)
```bash
python -m scripts.prepare_data                       # official splits -> data/processed + evidence corpus
python -m scripts.train_classical                    # LogReg / SVM / NB / RF + SHAP statistics
python -m scripts.build_index                        # MiniLM embeddings + FAISS index
python -m scripts.extract_clip_features              # CLIP features of each claim's image evidence
python -m scripts.train_deep --evidence retrieved --epochs 3      # deep multimodal network
python -m scripts.train_ensemble                     # stacking ensemble
python -m scripts.evaluate                           # Recall@k, ablations -> artifacts/reports/*.json
```
(or `bash scripts/run_all.sh`). Everything lands in `artifacts/`. Optional: `prepare_data --corpus sentences`
uses MOCHEG's large sentence-level corpus (from its `supplementary/` folder) instead of the compact evidence corpus.

**Images.** If the image files are absent, claims simply stay text-only and everything still runs. Training images
are each claim's *image evidence* (pictures from the fact-check article); a photo a user uploads may look
different, so treat the image signal as supporting, not decisive.

## Generated figures

Running `make_figures.py` produces one PNG per output (architecture, dataset overview, model comparison, confusion matrix, SHAP importance, retrieval recall, modality contribution and example explanations) in `docs/figures/`.
```
Produces: architecture diagram, dataset overview, model comparison, per-class F1, confusion matrices (all + best),
ROC curves, confidence distribution, top errors (+ CSV of misclassified claims), global SHAP, retrieval Recall@k,
deep-model training curves, evidence-source comparison, text-vs-image comparison, ensemble weights, dataset-level
modality contribution, and worked example panels (text-only and text+image). Missing pieces are skipped with a message.

Then `python -m scripts.export_results_md --out docs` writes `docs/RESULTS.md` (tables + embedded figures, copied to
`docs/figures/`) so the results show up directly on GitHub.

### Run the app
```bash
export GROQ_API_KEY=...          # optional; without it you get a template explanation
streamlit run app.py
```

## 🔐 Security: rotate your leaked Groq key
The original notebook contained a real Groq API key in plain text. **Revoke it now** at
<https://console.groq.com/keys> and create a new one. In this project keys are only read from `GROQ_API_KEY`
(env var), Streamlit secrets, or the sidebar box (session only). Never commit `.env` or `secrets.toml`
(both are git-ignored). If the old key was ever pushed to GitHub or shared, treat it as compromised.

## Deploy
Trained artifacts (~0.5 GB) don't belong in git. Upload them once, let the host download them at start-up:
```bash
huggingface-cli login
python -m scripts.push_to_hub your-username/traceverify-artifacts --private
```
Then set `TV_ARTIFACT_REPO`, `HF_TOKEN` (if private) and `GROQ_API_KEY` as secrets on the host.

* **Hugging Face Spaces (recommended, free CPU with 16 GB RAM)** — create a Space with SDK *Docker*, push this repo
  (add `app_port: 7860` and `sdk: docker` to the Space's README front-matter), set the secrets above.
* **Render / Railway / Fly.io / any VM** — build the included `Dockerfile`.
* **Streamlit Community Cloud** — works for a text-only build; the torch + CLIP + FAISS stack can exceed its
  memory limit, so prefer Spaces for the full multimodal app.

## Design decisions and safeguards

1. **Secure key handling.** The Groq API key is read only from environment variables or Streamlit/Colab secrets. It is never stored in code or committed to the repository.
2. **Consistent claim IDs.** Claim IDs are normalised to strings across the splits and the evidence corpus, and Recall@k is computed in batch over the entire test set rather than a sample.
3. **Original dataset with official splits.** The system uses the authors' original MOCHEG release with its official train/validation/test splits and real evidence images. Cross-split duplicate claims are removed from validation and test.
4. **No label leakage.** `ruling_outline` and `origin` contain the fact-checker's verdict, so they are never used as features.
5. **Reliable LLM explanations.** Reasoning models spend tokens on thinking, so the token budget is raised and a safe fallback explanation is used if the model returns nothing.
6. **Honest retrieval and RAG evaluation.** A claim's own fact-check is part of the corpus, so Recall@k alone is optimistic. `evaluate.py` also reports a leave-own-evidence-out setting that simulates a new, unseen claim, and the deep model is trained on that setting by default (`--evidence retrieved`).
7. **Exact, tested SHAP.** Closed-form linear SHAP, identical to `shap.LinearExplainer` and unit-tested against it.
8. **Single, focused dataset.** MOCHEG only (PolitiFact and Snopes claims with image evidence).

## Project layout
```
app.py                    Streamlit UI (Verify · Models & performance · Explainability · About)
traceverify/              data · classical · mm_model (deep) · ensemble · retrieval · explain · llm · pipeline
scripts/                  prepare_data → train_classical → build_index → extract_clip_features → train_deep
                          → train_ensemble → evaluate → push_to_hub
tests/                    offline unit tests (MOCHEG-layout loader, ML zoo, SHAP vs shap lib, retrieval, ensemble, image index)
docs/PAPER_CORRECTIONS.md exact edits for the research paper
```

## Results (MOCHEG official test split, 3-class, chance = 33.3%)

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Stacking ensemble | 50.1% | 0.472 |
| Deep multimodal network | 49.0% | 0.462 |
| Linear SVM | 45.4% | 0.433 |
| Logistic Regression | 43.7% | 0.400 |
| Naive Bayes | 41.3% | 0.351 |
| Random Forest | 40.7% | 0.335 |

Dense retrieval Recall@1 / @5 / @10: 62.3% / 78.9% / 83.6%.
On claims with images, adding the image gave only a marginal gain (42.2% vs 42.0% accuracy).
Full tables and all figures: [docs/RESULTS.md](docs/RESULTS.md).

## Limitations
Research prototype. Verdicts are statistical predictions, not verified facts. The evidence corpus is MOCHEG only,
so claims outside it retrieve weakly related passages (the UI warns on low similarity). CLIP image-text
similarity is a coarse signal. Not for use as the sole basis of any real-world decision.
