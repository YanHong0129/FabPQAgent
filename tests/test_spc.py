"""Day 1: spc.py"""
import numpy as np
import pandas as pd
import pytest

from fabpqagent.spc import spc_check

BASELINE = pd.Series([9.0, 11.0] * 10)  # mean 10, sample std ~1.026


def test_limits_from_baseline():
    r = spc_check("s000", pd.Series([10.0]), BASELINE)
    std = BASELINE.std()
    assert r.center == pytest.approx(10.0)
    assert r.ucl == pytest.approx(10 + 3 * std)
    assert r.lcl == pytest.approx(10 - 3 * std)
    assert r.in_control


def test_rule1_reports_index_labels():
    values = pd.Series([10.0, 20.0, 10.0, 0.0], index=[100, 101, 102, 103])
    r = spc_check("s000", values, BASELINE)
    assert r.beyond_limits == [101, 103]
    assert not r.in_control


def test_rule_of_eight_reported_once_at_eighth_point():
    values = pd.Series([10.5] * 10 + [9.5] * 3, index=range(50, 63))
    r = spc_check("s000", values, BASELINE)
    assert r.shift_runs == [57]


def test_point_on_center_breaks_run():
    values = pd.Series([10.5] * 7 + [10.0] + [10.5] * 7)
    assert spc_check("s000", values, BASELINE).shift_runs == []


def test_two_separate_runs():
    values = pd.Series([10.5] * 8 + [9.5] * 8)
    assert spc_check("s000", values, BASELINE).shift_runs == [7, 15]


def test_nan_ignored():
    values = pd.Series([10.5, np.nan] * 8)
    r = spc_check("s000", values, pd.concat([BASELINE, pd.Series([np.nan])]))
    assert r.shift_runs == [14]  # 8 non-NaN points above centre; the 8th sits at label 14


def test_baseline_too_small():
    with pytest.raises(ValueError):
        spc_check("s000", pd.Series([1.0]), pd.Series([1.0, np.nan]))
