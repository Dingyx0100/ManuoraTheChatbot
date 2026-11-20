"""Train the TF-IDF and MiniLM action classifiers (experiment R, arms C and D) on the synthetic train split.

    python scripts/router/train_classifier.py --data train_v0_1 --method tfidf
    python scripts/router/train_classifier.py --data train_v0_1 --method minilm

Fits on `train` only; the regularisation strength is chosen on `val`. Test splits are never read here.
Writes data/models/router_<method>_<datahash>.joblib (+ .json with the split counts and val accuracy).
"""
from __future__ import annotations

import argparse
import hashlib
import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from manuora.evaluation import routing_data
from manuora.evaluation.common import ROOT
from manuora.evaluation.run import documents_for
from manuora.evaluation.common import settings
from manuora.generation.router import ClassifierRouter, classifier_text, features

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def table(examples, documents):
    rows = []
    for e in examples:
        context = routing_data.final_context(e, documents)
        if context is not None:  # an early reply answered it; no router runs
            rows.append((features(context), e["label"]["action"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--method", choices=("tfidf", "minilm"), required=True)
    args = parser.parse_args()

    documents = documents_for(settings())
    train = routing_data.load(args.data, ("train",))
    val = routing_data.load(args.data, ("val",))
    assert train and all(e["split"] == "train" for e in train)
    train_rows, val_rows = table(train, documents), table(val, documents)
    y_train, y_val = [a for _, a in train_rows], [a for _, a in val_rows]

    if args.method == "tfidf":
        x_train = [classifier_text(f) for f, _ in train_rows]
        x_val = [classifier_text(f) for f, _ in val_rows]

        def build(c):
            return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
                                 LogisticRegression(C=c, max_iter=4000, class_weight="balanced"))
    else:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(ENCODER, device="cpu")
        bundle = {"method": "minilm", "encoder": ENCODER, "model": None}
        router = ClassifierRouter.__new__(ClassifierRouter)
        router.bundle, router.method, router._encoder = bundle, "minilm", encoder
        x_train = np.vstack([router.vector(f) for f, _ in train_rows])
        x_val = np.vstack([router.vector(f) for f, _ in val_rows])

        def build(c):
            return LogisticRegression(C=c, max_iter=4000, class_weight="balanced")

    scores = {}
    for c in (0.5, 2.0, 8.0, 32.0):
        model = build(c).fit(x_train, y_train)
        scores[c] = float(np.mean(model.predict(x_val) == np.array(y_val)))
    best = max(scores, key=lambda c: (scores[c], -c))
    model = build(best).fit(x_train, y_train)

    digest = hashlib.sha1((routing_data.DATA / args.data / "examples.jsonl").read_bytes()).hexdigest()[:10]
    out = ROOT / f"data/models/router_{args.method}_{digest}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"method": args.method, "encoder": ENCODER if args.method == "minilm" else None, "model": model}, out)
    meta = {"data": args.data, "data_sha1": digest, "method": args.method, "train_rows": len(train_rows),
            "val_rows": len(val_rows), "val_action_accuracy_by_C": scores, "C": best}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(out.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
