"""Statistical process control (SPC) check for one sensor.

Tests: tests/test_spc.py

Background:
- Control limits come from a stable BASELINE period: centre = mean, UCL/LCL = mean +/- 3 sigma.
- Western Electric rule 1: any single point beyond 3 sigma  -> out of control.
- Western Electric rule 4 (often called "rule of 8"): 8 consecutive points on the same side of
  the centre line -> the process has shifted, even if no point crosses a limit.
Reference: NIST/SEMATECH e-Handbook, section 6.3.2.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SIGMA_LIMIT = 3.0
RUN_LENGTH = 8


@dataclass(frozen=True)
class SPCResult:
    sensor: str
    center: float
    ucl: float
    lcl: float
    beyond_limits: list[int]   # index labels of points beyond UCL/LCL (rule 1)
    shift_runs: list[int]      # index label where each run of RUN_LENGTH same-side points ENDS (rule 4)

    @property
    def in_control(self) -> bool:
        return not self.beyond_limits and not self.shift_runs


def spc_check(sensor: str, values: pd.Series, baseline: pd.Series) -> SPCResult:
    """Check `values` (in time order) against limits computed from `baseline`.

    - Ignore NaN in both series (drop them; keep original index labels in the results).
    - Use the sample std (pandas default, ddof=1).
    - shift_runs: report a run once, at the label of its 8th point. A run of 10 same-side points
      is ONE run (reported at the 8th), not three. A point exactly on the centre breaks a run.
    - Raise ValueError if the baseline has fewer than 2 non-NaN points.
    """
    base = baseline.dropna()
    if len(base) < 2:
        raise ValueError(f"{sensor}: need at least 2 baseline points to estimate sigma, got {len(base)}")

    center = float(base.mean())
    sigma = float(base.std())  # sample std (ddof=1)
    ucl, lcl = center + SIGMA_LIMIT * sigma, center - SIGMA_LIMIT * sigma

    points = values.dropna()  # keeps the original index labels
    beyond = points[(points > ucl) | (points < lcl)].index.tolist()
    return SPCResult(sensor, center, ucl, lcl, beyond_limits=beyond, shift_runs=_shift_runs(points, center))


def _shift_runs(points: pd.Series, center: float) -> list[int]:
    """Label of the RUN_LENGTH-th point of each run of same-side points."""
    runs: list[int] = []
    side, length = 0, 0
    for label, value in points.items():
        new_side = (value > center) - (value < center)  # +1 above, -1 below, 0 exactly on the centre
        if new_side == 0:
            length = 0          # a point on the centre breaks the run
        elif new_side == side:
            length += 1         # same side as the previous point: the run grows
        else:
            length = 1          # switched sides: a new run starts here
        side = new_side
        if length == RUN_LENGTH:  # "==" not ">=": a run of 10 is reported once, at its 8th point
            runs.append(label)
    return runs
