"""Day 1: model.py"""
import math

import numpy as np
import pandas as pd

from fabpqagent.data import sensor_columns
from fabpqagent.model import load, predict_fail, save, select_features, top_features


def test_select_features_drops_mostly_missing_and_constant():
    cols = sensor_columns()
    df = pd.DataFrame({c: np.arange(10, dtype=float) for c in cols})
    df["s001"] = [np.nan] * 6 + [1.0] * 4      # 60% missing -> drop
    df["s002"] = 5.0                            # constant -> drop
    df["s003"] = [np.nan] * 4 + list(range(6))  # 40% missing -> keep
    features = select_features(df)
    assert "s001" not in features and "s002" not in features
    assert "s003" in features
    assert features == [c for c in cols if c in features]  # original order


def test_train_beats_baseline_by_2x(trained):
    m = trained.metrics
    assert {"pr_auc", "baseline_pr_auc", "precision", "recall"} <= m.keys()
    assert 0.05 < m["baseline_pr_auc"] < 0.09
    assert m["pr_auc"] >= 2 * m["baseline_pr_auc"], m
    assert 0 <= m["precision"] <= 1 and 0 <= m["recall"] <= 1
    assert 0 < trained.threshold < 1


def test_importances_and_pass_stats_cover_features(trained):
    assert set(trained.importances.index) == set(trained.features)
    assert trained.importances.is_monotonic_decreasing
    assert {"mean", "std"} <= set(trained.pass_stats.columns)


def test_predict_fail_is_probability(trained, lots):
    probs = [predict_fail(trained, lots.iloc[i]) for i in (0, 2, 100)]
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_failed_lots_score_higher_on_average(trained, lots):
    sample = lots.groupby("failed").head(40)
    probs = sample.apply(lambda row: predict_fail(trained, row), axis=1)
    assert probs[sample["failed"] == 1].mean() > probs[sample["failed"] == 0].mean()


def test_top_features_format_and_order(trained, lots):
    feats = top_features(trained, lots.iloc[2], k=5)
    assert len(feats) == 5
    assert all({"sensor", "value", "pass_mean", "z"} <= f.keys() for f in feats)
    zs = [abs(f["z"]) for f in feats]
    assert zs == sorted(zs, reverse=True)
    assert all(not math.isnan(f["value"]) for f in feats)
    assert set(f["sensor"] for f in feats) <= set(trained.importances.index[:30])


def test_save_load_round_trip(trained, lots, tmp_path):
    path = tmp_path / "model.joblib"
    save(trained, path)
    again = load(path)
    assert predict_fail(again, lots.iloc[5]) == predict_fail(trained, lots.iloc[5])
