from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.covariance import LedoitWolf


@dataclass
class FinancialPhysicsTransformer:

    stress_columns: Optional[List[str]] = None
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.stress_columns is None:
            self.stress_columns = [
                "demand_pressure",
                "inventory_ratio",
                "capacity_utilization",
                "supplier_reliability",
                "port_congestion",
                "transport_delay",
                "cost_shock",
                "network_propagation",
                "geopolitical_risk",
            ]
        self.feature_names_out_ = [
            "fp_distance_to_default_proxy",
            "fp_default_probability_proxy",
            "fp_liquidity_pressure",
            "fp_network_stress_energy",
            "fp_tail_loss_pressure",
        ]
        self.median_: Optional[pd.Series] = None
        self.mad_: Optional[pd.Series] = None
        self.precision_: Optional[np.ndarray] = None

    def _raw_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.stress_columns or [] if c in df.columns]
        if not cols:
            raise ValueError("FinancialPhysicsTransformer requires at least one stress column in df.")
        raw = df[cols].copy()
        for c in cols:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
        return raw

    def fit(self, df: pd.DataFrame) -> "FinancialPhysicsTransformer":
        raw = self._raw_matrix(df)
        med = raw.median(axis=0, skipna=True)
        mad = (raw - med).abs().median(axis=0, skipna=True).replace(0, self.eps)
        z = ((raw.fillna(med) - med) / (mad + self.eps)).to_numpy(dtype=float)
        self.median_ = med
        self.mad_ = mad
        try:
            lw = LedoitWolf().fit(z)
            self.precision_ = np.asarray(lw.precision_, dtype=float)
        except Exception:
            cov = np.cov(z, rowvar=False) + np.eye(z.shape[1]) * 1e-3
            self.precision_ = np.linalg.pinv(cov)
        return self

    @staticmethod
    def _col(df: pd.DataFrame, name: str, default: float) -> np.ndarray:
        if name not in df.columns:
            return np.full(len(df), default, dtype=float)
        return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(dtype=float)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.median_ is None or self.mad_ is None or self.precision_ is None:
            raise RuntimeError("FinancialPhysicsTransformer must be fitted before transform().")
        raw = self._raw_matrix(df)
        raw = raw.reindex(columns=list(self.median_.index))
        z = ((raw.fillna(self.median_) - self.median_) / (self.mad_ + self.eps)).to_numpy(dtype=float)
        energy = np.sqrt(np.maximum(np.sum((z @ self.precision_) * z, axis=1), 0.0))

        demand = np.clip(self._col(df, "demand_pressure", 0.5), 0.0, 3.0)
        inv = np.clip(self._col(df, "inventory_ratio", 0.5), 0.0, 3.0)
        rel = np.clip(self._col(df, "supplier_reliability", 0.8), 0.0, 1.0)
        port = np.clip(self._col(df, "port_congestion", 0.2), 0.0, 3.0)
        cap = np.clip(self._col(df, "capacity_utilization", 0.75), 0.0, 3.0)
        cost = np.clip(self._col(df, "cost_shock", 0.0), 0.0, 3.0)
        geo = np.clip(self._col(df, "geopolitical_risk", 0.0), 0.0, 3.0)
        prop = np.clip(self._col(df, "network_propagation", 0.2), 0.0, 3.0)

        asset_buffer = 0.15 + rel + 0.45 * inv + 0.20 * np.maximum(1.0 - port, 0.0)
        obligations = 0.15 + 0.55 * demand + 0.35 * cap + 0.35 * cost + 0.25 * geo + 0.20 * prop
        sigma = np.clip(0.12 + 0.22 * prop + 0.18 * port + 0.17 * cost + 0.12 * geo, 0.05, 2.0)
        distance_to_default = (
            np.log((asset_buffer + self.eps) / (obligations + self.eps))
            + (0.03 - 0.5 * sigma ** 2)
        ) / sigma
        default_probability = ndtr(-distance_to_default)
        liquidity_pressure = (cost + demand + cap) / (inv + rel + 0.20)
        tail_loss_pressure = default_probability * (1.0 + energy) * (0.5 + port + prop)

        out = np.vstack([
            distance_to_default,
            default_probability,
            liquidity_pressure,
            energy,
            tail_loss_pressure,
        ]).T
        out = np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
        return np.clip(out, -8.0, 8.0).astype(float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
