"""Offline tests (no model downloads, no torch): data, classical ML, SHAP, retrieval, ensemble."""
import zlib

import numpy as np
import pandas as pd
import pytest

from traceverify import baseline, classical, config, data
from traceverify.ensemble import StackingEnsemble
from traceverify.explain import LinearExplainer
from traceverify.retrieval import ClaimImageIndex, Retriever, tokenize

TOPICS = {
    "refuted": ["celebrity died plane crash hoax", "vaccine microchip conspiracy false", "free money facebook payout scam"],
    "supported": ["senate confirmed judge unanimously vote", "budget bill signed law passed", "study confirms exercise improves health"],
    "nei": ["unknown report allegedly claims rumor", "unclear viral post origin", "anonymous source says something"],
}


def write_fake_mocheg(root, n_per=40):
    """Create files in the official MOCHEG layout: <split>/Corpus2.csv (+ images/ with '<id>-proof-..' names)."""
    from PIL import Image
    rng = np.random.default_rng(0)
    cid = 1000
    rows = {"train": [], "val": [], "test": []}
    label_names = {"refuted": "refuted", "supported": "supported", "nei": "NEI"}
    for label, phrases in TOPICS.items():
        for i in range(n_per):
            cid += 1
            split = "train" if i < int(n_per * .7) else "val" if i < int(n_per * .85) else "test"
            base = phrases[i % len(phrases)]
            extra = " ".join(rng.choice(["today", "news", "people", "report", "says", "video"], 2))
            for j in range(2):                      # two evidence rows per claim (official: one row per evidence)
                rows[split].append({"Claim": f"{base} {extra} uniq{cid}x", "claim_id": cid, "Evidence": f"evidence {j} about {base} uniq{cid}x",
                                    "evidence_id": j, "cleaned_truthfulness": label_names[label],
                                    "ruling_outline": "SECRET VERDICT TEXT", "Origin": "FULL ARTICLE TEXT",
                                    "Snopes URL": f"https://example.org/{cid}"})
            if i % 2 == 0:
                d = root / split / "images"
                d.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 32), (i * 5 % 255, 80, 120)).save(d / f"{cid}-proof-0-photo.png")
    # duplicate claim text in test that also exists in train -> must be dropped (leak control)
    rows["test"].append({**rows["train"][0], "claim_id": 9999})
    for split, r in rows.items():
        (root / split).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(r).to_csv(root / split / "Corpus2.csv", index=False)


@pytest.fixture(scope="module")
def mocheg_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mocheg_release") / "MOCHEG_v1"     # nested folder like a real unzip
    write_fake_mocheg(root)
    return root


@pytest.fixture(scope="module")
def frames(mocheg_root):
    train, val, test, corpus, stats = data.load_mocheg(mocheg_root)
    return pd.concat([train, val, test], ignore_index=True), train, val, test


class HashEmbedder:
    """Deterministic bag-of-words embedder so similarity is meaningful without downloads."""
    dim = 1024

    def encode(self, texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
        out = np.zeros((len(texts), self.dim), "float32")
        for i, t in enumerate(texts):
            for tok in tokenize(t):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1
        n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
        return out / n


def test_loader_uses_official_splits_and_columns(mocheg_root):
    train, val, test, corpus, stats = data.load_mocheg(mocheg_root)
    assert len(train) and len(val) and len(test)
    for f in (train, val, test):
        assert f["claim_id"].map(type).eq(str).all()
        assert f["label"].isin(config.LABELS).all()
        assert f["text_evidence"].apply(len).eq(2).all()          # rows aggregated per claim
    assert set(train["label"]) == set(config.LABELS)               # 'NEI' normalised to 'nei'
    assert not set(train["claim"]) & set(test["claim"])            # cross-split duplicate removed
    assert stats["official_claims"]["test"] == stats["after_dedup"]["test"] + 1
    assert list(corpus.columns) == ["evidence_id", "claim_id", "text", "source_url"]


def test_verdict_leaking_columns_are_never_loaded(mocheg_root):
    train, *_ = data.load_mocheg(mocheg_root)
    blob = " ".join(train["text_evidence"].apply(" ".join)) + " ".join(train["claim"])
    assert "SECRET VERDICT" not in blob and "FULL ARTICLE" not in blob


def test_images_matched_by_claim_id_prefix(mocheg_root):
    train, *_ = data.load_mocheg(mocheg_root)
    with_img = train[train["image"].apply(len) > 0]
    assert 0 < len(with_img) < len(train)
    cid = with_img.iloc[0]["claim_id"]
    assert with_img.iloc[0]["image"][0].startswith(f"{cid}-proof-")


def test_processed_roundtrip_and_string_ids(tmp_path, mocheg_root):
    train, val, test, corpus, _ = data.load_mocheg(mocheg_root)
    data.save_processed(train, val, test, corpus, tmp_path)
    back = data.load_split("train", tmp_path)
    assert back["claim_id"].map(type).eq(str).all() and isinstance(back["text_evidence"].iloc[0], list)
    assert data.load_evidence_lookup(tmp_path)["claim_id"].map(type).eq(str).all()


def test_label_normalisation():
    assert [data.normalize_label(x) for x in ("Supported", "refuted", "NEI", "Not enough info", "weird")] == \
        ["supported", "refuted", "nei", "nei", None]


def test_classical_zoo_trains_and_orders_columns(frames):
    _, tr, va, _ = frames
    zoo = classical.build_models()
    # shrink the forest so the test is fast
    zoo["random_forest"].set_params(clf__n_estimators=20)
    classical.fit_all(zoo, tr["claim"], tr["label"])
    for name, pipe in zoo.items():
        p = classical.proba_matrix(pipe, va["claim"])
        assert p.shape == (len(va), 3) and np.allclose(p.sum(1), 1, atol=1e-5), name
        acc = np.mean(np.array(config.LABELS)[p.argmax(1)] == va["label"].to_numpy())
        assert acc > 0.5, (name, acc)


def test_shap_closed_form_matches_shap_library(frames):
    _, tr, va, _ = frames
    pipe = baseline.build_pipeline().fit(list(tr["claim"]), list(tr["label"]))
    tmp = {"pipeline": pipe,
           "tfidf_mean": np.asarray(pipe.named_steps["tfidf"].transform(list(tr["claim"])).mean(axis=0)).ravel()}
    expl = LinearExplainer(tmp)
    claim = va["claim"].iloc[0]
    local = dict(expl.local(claim, "refuted", top_n=1000))
    assert local
    shap = pytest.importorskip("shap")
    lr = pipe.named_steps["logistic_regression"]
    Xtr = pipe.named_steps["tfidf"].transform(list(tr["claim"]))
    e = shap.LinearExplainer(lr, Xtr.toarray())
    vals = e.shap_values(pipe.named_steps["tfidf"].transform([claim]).toarray())
    vals = vals[:, :, config.LABELS.index("refuted")] if getattr(vals, "ndim", 0) == 3 else vals[config.LABELS.index("refuted")]
    feats = pipe.named_steps["tfidf"].get_feature_names_out()
    for term, v in local.items():
        i = list(feats).index(term)
        assert abs(vals.ravel()[i] - v) < 1e-4


def test_retrieval_dense_hybrid_and_recall(frames):
    df, *_ = frames
    corpus = data.build_evidence_corpus(df)
    r = Retriever.build(corpus, HashEmbedder(), progress=False)
    claim = df.iloc[5]
    hit = r.search(claim["claim"], k=3)
    assert hit and hit[0]["claim_id"] == claim["claim_id"]
    hyb = r.search(claim["claim"], k=3, hybrid=True)
    assert hyb[0]["claim_id"] == claim["claim_id"]
    excl = r.search(claim["claim"], k=5, exclude_claim_id=claim["claim_id"])
    assert all(e["claim_id"] != claim["claim_id"] for e in excl)
    rec = r.recall_at_ks(df.head(20), ks=(1, 5))
    assert rec["5"] >= rec["1"] > 0.5


def test_retriever_save_load_roundtrip(tmp_path, frames):
    df, *_ = frames
    r = Retriever.build(data.build_evidence_corpus(df), HashEmbedder(), progress=False)
    r.save(tmp_path)
    r2 = Retriever.load(tmp_path, embedder=HashEmbedder())
    q = df.iloc[3]["claim"]
    assert [e["evidence_id"] for e in r.search(q, 3)] == [e["evidence_id"] for e in r2.search(q, 3)]


def test_stacking_ensemble(frames):
    _, tr, va, te = frames
    zoo = classical.build_models()
    zoo.pop("random_forest")
    classical.fit_all(zoo, tr["claim"], tr["label"])
    pv = {n: classical.proba_matrix(p, va["claim"]) for n, p in zoo.items()}
    pt = {n: classical.proba_matrix(p, te["claim"]) for n, p in zoo.items()}
    st = StackingEnsemble(list(pv)).fit(pv, va["label"])
    p = st.predict_proba(pt)
    assert p.shape == (len(te), 3) and np.allclose(p.sum(1), 1, atol=1e-5)
    assert abs(sum(st.member_weights().values()) - 1) < 1e-6


def test_claim_image_index_search(frames):
    df, *_ = frames
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(len(df), 32)).astype("float32")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    idx = ClaimImageIndex.build(df, emb)
    hits = idx.search(emb[7], k=3)
    assert hits[0]["claim_id"] == df.iloc[7]["claim_id"]


def test_streaming_extract_keeps_only_needed_files(tmp_path, mocheg_root):
    """Simulate the Zenodo tar.gz: nested root, big files we must skip, images in and out of split folders."""
    import io
    import tarfile as tf
    from scripts.download_mocheg import stream_extract

    archive = tmp_path / "mocheg_v1.tar.gz"
    with tf.open(archive, "w:gz") as tar:
        for p in sorted(mocheg_root.rglob("*")):                       # real fake-release files, nested
            if p.is_file():
                tar.add(p, arcname=f"mocheg_v1/{p.relative_to(mocheg_root).as_posix()}")
        for name, payload in (("mocheg_v1/Corpus3.csv", b"x" * 5000), ("mocheg_v1/images/1-2-3-global.jpg", b"jpg"),
                              ("mocheg_v1/supplementary/img_corpus_emb.pkl", b"y" * 5000),
                              ("mocheg_v1/supplementary/Corpus3_sentence_level.csv", b"claim_id,corpus_id,paragraph\n")):
            info = tf.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    out = tmp_path / "out"
    with open(archive, "rb") as fh:
        kept = stream_extract(fh, out)
    names = {p.name for p in out.rglob("*") if p.is_file()}
    assert "Corpus2.csv" in names and kept["images"] > 0
    assert "Corpus3.csv" not in names and "img_corpus_emb.pkl" not in names and "1-2-3-global.jpg" not in names
    assert "Corpus3_sentence_level.csv" not in names
    with open(archive, "rb") as fh:                                    # opt-in sentences + idempotent re-run
        again = stream_extract(fh, out, with_sentences=True)
    assert again["skipped_existing"] > 0 and again["images"] == 0
    assert "Corpus3_sentence_level.csv" in {p.name for p in out.rglob("*")}
    train, val, test, _, stats = data.load_mocheg(out)                 # loader works on the extracted subset
    assert len(train) and sum(stats["claims_with_image"].values()) > 0


def test_streamed_images_are_shrunk_and_keep_claim_prefix(tmp_path):
    import io
    import tarfile as tf
    from PIL import Image
    from scripts.download_mocheg import stream_extract

    buf = io.BytesIO()
    Image.new("RGB", (2000, 1000), (10, 20, 30)).save(buf, "PNG")
    archive = tmp_path / "a.tar.gz"
    with tf.open(archive, "w:gz") as tar:
        info = tf.TarInfo("root/train/images/77-proof-1-big.png")
        info.size = len(buf.getvalue())
        tar.addfile(info, io.BytesIO(buf.getvalue()))
        bad = tf.TarInfo("root/train/images/78-proof-1-broken.png")
        bad.size = 4
        tar.addfile(bad, io.BytesIO(b"nope"))
    with open(archive, "rb") as fh:
        kept = stream_extract(fh, tmp_path / "o", max_side=512)
    out = list((tmp_path / "o").rglob("77-proof-1-big.jpg"))
    assert kept["images"] == 1 and kept["unreadable_images"] == 1 and len(out) == 1
    assert max(Image.open(out[0]).size) == 512


def test_subset_selection_compares_all_vs_with_evidence(frames):
    _, tr, va, _ = frames
    tr = tr.copy()
    tr["text_evidence"] = [[] if i % 3 == 0 else e for i, e in enumerate(tr["text_evidence"])]   # evidence-less claims
    small = lambda: {k: v for k, v in classical.build_models().items() if k in ("logistic_regression", "naive_bayes")}
    models, log, subsets = classical.select_by_validation(tr, va, make_models=small)
    assert set(models) == {"logistic_regression", "naive_bayes"} and set(subsets) == {"all", "with_evidence"}
    for n, row in log.items():
        assert row["chosen"] in ("all", "with_evidence") and row["all"] >= 0 and row["with_evidence"] >= 0
        assert row[row["chosen"]] == max(row["all"], row["with_evidence"])
    full = {k: v for k, v in zip(["a"], [tr])}
    _, log2, sub2 = classical.select_by_validation(frames[1].assign(text_evidence=frames[1]["text_evidence"]), va, make_models=small)
    assert set(sub2) == {"all"}                                                                      # nothing to compare


def test_calibrated_svm_does_not_collapse_on_imbalanced_data():
    """Regression test: SVM trained on a 65/22/13 split must still predict all classes on balanced test data."""
    rng = np.random.default_rng(1)
    vocab = {c: [f"{c}{i}" for i in range(60)] for c in config.LABELS}
    common = [f"w{i}" for i in range(300)]

    def mk(c):
        words = list(rng.choice(common, 17))
        if rng.random() < 0.55:
            words.append(rng.choice(vocab[c]))
        return " ".join(words)

    def make(counts):
        X, y = [], []
        for c, n in counts.items():
            X += [mk(c) for _ in range(n)]
            y += [c] * n
        return X, y
    Xtr, ytr = make({"refuted": 650, "supported": 220, "nei": 160})
    Xte, yte = make({"refuted": 300, "supported": 300, "nei": 300})
    models = classical.fit_all({"linear_svm": classical.build_models()["linear_svm"]}, Xtr, ytr)
    pred = np.array(config.LABELS)[classical.proba_matrix(models["linear_svm"], Xte).argmax(1)]
    counts = {c: int((pred == c).sum()) for c in config.LABELS}
    assert min(counts.values()) > 100, counts          # old behaviour: ~0 'nei' predictions


def test_feature_cache_roundtrip_aligned_by_claim_id(tmp_path, monkeypatch):
    from traceverify import features
    monkeypatch.setattr(config, "FEATURE_DIR", tmp_path)
    ids = [str(i) for i in range(50)]
    img = np.random.default_rng(0).random((50, config.CLIP_DIM)).astype("float32")
    txt = np.random.default_rng(1).random((50, config.CLIP_DIM)).astype("float32")
    has = np.arange(50) % 2 == 0
    features.save_features("t", ids, img, txt, has)
    df = pd.DataFrame({"claim_id": ["7", "3", "999", "10"]})             # different order + an unknown id
    i, t, h = features.load_features("t", df)
    assert np.allclose(i[0], img[7]) and np.allclose(t[1], txt[3]) and list(h) == [False, False, False, True]
    assert not i[2].any()
