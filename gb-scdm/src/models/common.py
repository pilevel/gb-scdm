from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class PolicyResult:
    probabilities: np.ndarray
    actions: np.ndarray
    explanations: List[Dict[str, Any]]


class BasePolicy:
    name = "base"

    def fit(self, x, y, loss, expert_actions, df, **kwargs):
        raise NotImplementedError

    def predict(self, x, df, **kwargs) -> PolicyResult:
        raise NotImplementedError


def risk_to_action(prob: np.ndarray, df: Optional[pd.DataFrame] = None, adjustment_limit: float = 0.35) -> np.ndarray:
    p = np.asarray(prob, dtype=float).reshape(-1)
    n = len(p)
    base = np.zeros((n, 5), dtype=float)
    if df is not None and len(df) == n:
        demand = df.get("demand_pressure", pd.Series(np.zeros(n))).to_numpy(dtype=float)
        inv = df.get("inventory_ratio", pd.Series(np.ones(n))).to_numpy(dtype=float)
        cong = df.get("port_congestion", pd.Series(np.zeros(n))).to_numpy(dtype=float)
        rel = df.get("supplier_reliability", pd.Series(np.ones(n))).to_numpy(dtype=float)
        util = df.get("capacity_utilization", pd.Series(np.zeros(n))).to_numpy(dtype=float)
    else:
        demand = inv = cong = rel = util = np.zeros(n)
    base[:, 0] = 0.15 + 0.55 * p + 0.20 * (1.0 - rel)
    base[:, 1] = 0.10 + 0.55 * p + 0.25 * cong
    base[:, 2] = 0.10 + 0.45 * p + 0.25 * util
    base[:, 3] = 0.15 + 0.50 * p + 0.25 * (1.0 - inv)
    base[:, 4] = 0.10 + 0.45 * p + 0.25 * demand
    base = np.clip(base, 0.0, 1.0)
    out = np.zeros_like(base)
    prev = np.zeros(5)
    for i in range(n):
        out[i] = np.clip(base[i], prev - adjustment_limit, prev + adjustment_limit)
        out[i] = np.clip(out[i], 0.0, 1.0)
        prev = out[i]
    return out
