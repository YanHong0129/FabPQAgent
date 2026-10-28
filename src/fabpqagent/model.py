"""DAY 1 — YOU WRITE THIS. Predict which lots fail, and explain why.

Tests: tests/test_model.py   Run: uv run pytest tests/test_model.py

Things to be ready to explain in the interview:
- Why accuracy is useless here (only 6.6% of lots fail: "always pass" scores 93%).
- Why PR-AUC (average precision) instead of ROC-AUC for rare failures.
- How you chose the decision threshold (a fab cares more about missing a bad lot than a false alarm).
- Why you fit imputation/scaling inside the Pipeline (no leakage from the test set).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fabpqagent.data import ROOT, sensor_columns

MODEL_PATH = ROOT / "data" / "model.joblib"   # app.py loads the model from here
MAX_MISSING_FRACTION = 0.5
TEST_SIZE = 0.25
LR_MAX_ITER = 2000                # ~450 scaled features need more than the default 100 iterations
RF_TREES = 500                    # more trees = more stable probabilities; ~2s to fit here
PLACEHOLDER_THRESHOLD = 0.2       # replaced in Step 7 by a threshold chosen from the PR curve


@dataclass(frozen=True)
class TrainedModel:
    pipeline: Any                    # fitted sklearn Pipeline (impute -> [scale] -> classifier)
    features: list[str]              # sensor columns the pipeline expects, in order
    threshold: float                 # probability above which a lot is flagged as "will fail"
    metrics: dict[str, float]        # must include: pr_auc, baseline_pr_auc, precision, recall
    pass_stats: pd.DataFrame = field(repr=False, default=None)  # per-sensor mean/std over PASS lots (for top_features)
    importances: pd.Series = field(repr=False, default=None)    # sensor -> importance, sorted descending


def select_features(df: pd.DataFrame) -> list[str]:
    """Sensor columns worth using: drop ones missing in > MAX_MISSING_FRACTION of lots and
    ones that never change (zero variance). Keep the original column order.

    Hint: df[sensor_columns()].isna().mean(), .std()
    """
    sensors = df[sensor_columns()]
    mostly_present = sensors.isna().mean() <= MAX_MISSING_FRACTION
    varies = sensors.std() > 0  # NaN (all-missing column) > 0 is False, so those are dropped too
    keep = mostly_present & varies
    return keep.index[keep].tolist()


def train(df: pd.DataFrame, seed: int = 0) -> TrainedModel:
    """Train a fail classifier on `df` (columns lot_id, ts, failed, s000..s589).

    Steps:
      1. features = select_features(df); X = df[features]; y = df["failed"]
      2. Stratified train/test split (TEST_SIZE, random_state=seed) so both halves have fails.
      3. Pipeline: SimpleImputer(median) -> classifier. Start with LogisticRegression
         (needs StandardScaler, class_weight="balanced"), then try RandomForestClassifier
         (class_weight="balanced_subsample"). Keep whichever has the higher test PR-AUC.
      4. metrics on the TEST set: pr_auc = average_precision_score, baseline_pr_auc = fail rate,
         precision / recall at your chosen threshold.
      5. pass_stats = mean and std of each feature over PASS lots in the TRAIN set.
      6. importances: RF -> feature_importances_; LR -> abs(coef_). As a pd.Series, sorted desc.

    Reference: a quick random forest reaches PR-AUC ~0.19 vs baseline ~0.066. SECOM is hard;
    beating 2x baseline is a real result. Say so honestly in the README.
    """
    features = select_features(df)
    X, y = df[features], df["failed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=seed
    )

    threshold = PLACEHOLDER_THRESHOLD
    builders = {"logistic_regression": _logistic_pipeline, "random_forest": _forest_pipeline}
    fitted = {name: build(seed).fit(X_train, y_train) for name, build in builders.items()}
    pr_aucs = {name: _evaluate(p, X_test, y_test, threshold)["pr_auc"] for name, p in fitted.items()}
    best = max(pr_aucs, key=pr_aucs.get)

    pipeline = fitted[best]
    metrics = {
        **_evaluate(pipeline, X_test, y_test, threshold),
        **{f"pr_auc_{name}": score for name, score in pr_aucs.items()},  # keep both for the README
    }

    return TrainedModel(
        pipeline=pipeline,
        features=features,
        threshold=threshold,
        metrics=metrics,
        pass_stats=X_train[y_train == 0].agg(["mean", "std"]).T,
        importances=_importances(pipeline, features),
    )


def _logistic_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=LR_MAX_ITER, random_state=seed)),
    ])


def _forest_pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # no scaler: tree splits don't care about scale
        ("clf", RandomForestClassifier(
            n_estimators=RF_TREES, class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        )),
    ])


def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float) -> dict[str, float]:
    proba = pipeline.predict_proba(X_test)[:, 1]   # column 1 = P(failed == 1)
    flagged = proba > threshold
    return {
        "pr_auc": float(average_precision_score(y_test, proba)),
        "baseline_pr_auc": float(y_test.mean()),   # PR-AUC of a random scorer = fail rate
        "precision": float(precision_score(y_test, flagged, zero_division=0)),
        "recall": float(recall_score(y_test, flagged)),
    }


def _importances(pipeline: Pipeline, features: list[str]) -> pd.Series:
    clf = pipeline[-1]
    values = clf.feature_importances_ if hasattr(clf, "feature_importances_") else abs(clf.coef_[0])
    return pd.Series(values, index=features).sort_values(ascending=False)


def predict_fail(model: TrainedModel, lot: pd.Series) -> float:
    """Probability (0-1) that this lot fails. `lot` is one row of the lots table."""
    row = lot[model.features].astype(float).to_frame().T  # one-row DataFrame, training column order
    return float(model.pipeline.predict_proba(row)[0, 1])


def top_features(model: TrainedModel, lot: pd.Series, k: int = 5, candidates: int = 30) -> list[dict[str, float | str]]:
    """Explain a prediction: among the `candidates` most important sensors, return the k where
    this lot is most unusual vs. passing lots, as dicts:
        {"sensor": "s059", "value": 9.1, "pass_mean": 2.6, "z": 4.2}
    sorted by abs(z) descending. Skip sensors where the lot's value is missing.
    z = (value - pass_mean) / pass_std
    """
    stats = model.pass_stats
    scored = [
        {
            "sensor": sensor,
            "value": float(lot[sensor]),
            "pass_mean": float(stats.at[sensor, "mean"]),
            "z": float((lot[sensor] - stats.at[sensor, "mean"]) / stats.at[sensor, "std"]),
        }
        for sensor in model.importances.index[:candidates]
        if pd.notna(lot[sensor]) and stats.at[sensor, "std"] > 0  # skip missing values; zero spread -> z undefined
    ]
    return sorted(scored, key=lambda f: abs(f["z"]), reverse=True)[:k]


def save(model: TrainedModel, path) -> None:
    """Hint: joblib.dump (ships with scikit-learn)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path) -> TrainedModel:
    model = joblib.load(path)  # unpickling runs code: only load files you created yourself
    if not isinstance(model, TrainedModel):
        raise TypeError(f"{path} does not contain a TrainedModel (got {type(model).__name__})")
    return model


if __name__ == "__main__":
    # Import this module by its package name. Objects built by the `__main__` copy would be pickled
    # as `__main__.TrainedModel`, which app.py could not find when it unpickles the file.
    from fabpqagent import model as fab_model
    from fabpqagent.db import load_lots

    trained = fab_model.train(load_lots())
    print(f"model: {type(trained.pipeline[-1]).__name__}   threshold: {trained.threshold}")
    for name, value in trained.metrics.items():
        print(f"  {name:28s} {value:.3f}")
    fab_model.save(trained, fab_model.MODEL_PATH)
    print(f"saved -> {fab_model.MODEL_PATH}")
