from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


ACTION_EFFECT = np.asarray([0.18, 0.14, 0.16, 0.12, 0.08], dtype=float)
ACTION_COST = np.asarray([20.0, 17.0, 23.0, 13.0, 9.0], dtype=float)
ACTION_CARBON = np.asarray([0.55, 0.78, 0.38, 0.22, 0.08], dtype=float)
ACTION_NAMES = [
    "alternate_supplier",
    "alternate_route",
    "capacity_expansion",
    "safety_stock",
    "demand_prioritisation",
]


def enforce_action_constraints(action, previous_action=None, adjustment_limit: float = 0.35) -> np.ndarray:

    a = np.asarray(action, dtype=float).reshape(5)
    previous = np.zeros(5, dtype=float) if previous_action is None else np.asarray(previous_action, dtype=float).reshape(5)
    limit = float(max(adjustment_limit, 1e-8))
    a = np.clip(a, previous - limit, previous + limit)
    return np.clip(a, 0.0, 1.0)


def project_action_sequence(actions, adjustment_limit: float = 0.35) -> np.ndarray:
    arr = np.asarray(actions, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"actions must have shape (n, 5), got {arr.shape}")
    out = np.zeros_like(arr, dtype=float)
    previous = np.zeros(5, dtype=float)
    for i in range(len(arr)):
        out[i] = enforce_action_constraints(arr[i], previous, adjustment_limit)
        previous = out[i]
    return out


def _col(df: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name not in df.columns:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default).to_numpy(dtype=float)


def evaluate_decision_outcomes(df: pd.DataFrame, actions, service_target: float = 0.95) -> Dict[str, np.ndarray]:

    a = np.asarray(actions, dtype=float)
    if a.shape != (len(df), 5):
        raise ValueError(f"actions must have shape ({len(df)}, 5), got {a.shape}")

    demand = np.maximum(_col(df, "demand_units", 100.0), 1.0)
    base_supply = np.maximum(_col(df, "base_supply", 92.0), 0.0)
    congestion = np.clip(_col(df, "port_congestion", 0.25), 0.0, 1.0)
    reliability = np.clip(_col(df, "supplier_reliability", 0.82), 0.0, 1.0)
    utilization = np.clip(_col(df, "capacity_utilization", 0.72), 0.0, 1.0)
    inventory = np.clip(_col(df, "inventory_ratio", 0.55), 0.0, 1.0)
    cost_shock = np.clip(_col(df, "cost_shock", 0.10), 0.0, 1.5)
    carbon = np.clip(_col(df, "carbon_intensity", 0.45), 0.0, 1.5)
    disruption = np.clip(_col(df, "disruption", 0.0), 0.0, 1.0)

    modifiers = np.column_stack([
        0.70 + 0.55 * (1.0 - reliability),
        0.68 + 0.62 * congestion,
        0.72 + 0.38 * utilization,
        0.70 + 0.50 * (1.0 - inventory),
        0.72 + 0.35 * np.clip(_col(df, "demand_pressure", 0.5), 0.0, 1.0),
    ])
    mitigation_supply = demand * np.sum(a * ACTION_EFFECT[None, :] * modifiers, axis=1)

    synergy = demand * (0.045 * a[:, 0] * a[:, 1] + 0.025 * a[:, 2] * a[:, 3])
    available_supply = np.maximum(0.0, base_supply + mitigation_supply + synergy)

    shortage = np.maximum(demand - available_supply, 0.0)
    fulfilled = demand - shortage
    service = np.clip(fulfilled / demand, 0.0, 1.0)
    shortage_ratio = shortage / demand

    previous = np.vstack([np.zeros((1, 5)), a[:-1]])
    adjustment = np.sum(np.abs(a - previous), axis=1)
    action_cost = np.sum(a * ACTION_COST[None, :] * (1.0 + 0.45 * cost_shock[:, None]), axis=1)
    carbon_cost = np.sum(a * ACTION_CARBON[None, :], axis=1) * (4.5 + 3.0 * carbon)
    base_cost = 48.0 + 0.13 * demand + 15.0 * cost_shock + 8.0 * congestion + 6.0 * disruption
    shortage_penalty = shortage * (3.0 + 2.0 * disruption)
    adjustment_cost = 8.0 * adjustment + 3.0 * adjustment**2
    operating_cost = base_cost + action_cost + carbon_cost + shortage_penalty + adjustment_cost

    target_violation = np.maximum(float(service_target) - service, 0.0)
    return {
        "demand": demand,
        "base_supply": base_supply,
        "available_supply": available_supply,
        "fulfilled": fulfilled,
        "shortage": shortage,
        "shortage_ratio": shortage_ratio,
        "service": service,
        "operating_cost": operating_cost,
        "action_cost": action_cost,
        "carbon_cost": carbon_cost,
        "adjustment_cost": adjustment_cost,
        "target_violation": target_violation,
    }


def expert_action_from_state(df: pd.DataFrame, adjustment_limit: float = 0.35) -> np.ndarray:
    
    n = len(df)
    demand = np.clip(_col(df, "demand_pressure", 0.5), 0, 1)
    inventory = np.clip(_col(df, "inventory_ratio", 0.5), 0, 1)
    utilization = np.clip(_col(df, "capacity_utilization", 0.7), 0, 1)
    reliability = np.clip(_col(df, "supplier_reliability", 0.8), 0, 1)
    congestion = np.clip(_col(df, "port_congestion", 0.2), 0, 1)
    propagation = np.clip(_col(df, "network_propagation", 0.2), 0, 1)
    geo = np.clip(_col(df, "geopolitical_risk", 0.1), 0, 1)
    risk = np.clip(
        0.18 * demand + 0.16 * (1 - inventory) + 0.14 * utilization
        + 0.18 * (1 - reliability) + 0.14 * congestion
        + 0.12 * propagation + 0.08 * geo,
        0, 1,
    )
    raw = np.column_stack([
        0.06 + 0.58 * risk + 0.25 * (1 - reliability) + 0.10 * geo,
        0.05 + 0.50 * risk + 0.30 * congestion + 0.10 * propagation,
        0.05 + 0.42 * risk + 0.28 * utilization + 0.12 * demand,
        0.07 + 0.44 * risk + 0.32 * (1 - inventory),
        0.04 + 0.38 * risk + 0.32 * demand,
    ])
    return project_action_sequence(np.clip(raw, 0, 1), adjustment_limit)
