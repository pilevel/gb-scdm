from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from src.env.supply_chain_env import (
    ACTION_COST, ACTION_EFFECT, ACTION_CARBON, enforce_action_constraints,
)
from src.models.common import PolicyResult
from src.models.financial_physics import FinancialPhysicsTransformer


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def add_intercept(x: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(x), 1), dtype=float), np.asarray(x, dtype=float)], axis=1)


def fit_logistic_centered(x: np.ndarray, y: np.ndarray, center: Optional[np.ndarray] = None,
                          prior_strength: float = 1.0, l2: float = 1e-4) -> np.ndarray:

    xa = add_intercept(x)
    y = np.asarray(y, dtype=float)
    p_dim = xa.shape[1]
    if center is None:
        center = np.zeros(p_dim, dtype=float)
    center = np.asarray(center, dtype=float)

    pos = max(y.sum(), 1.0)
    neg = max(len(y) - y.sum(), 1.0)
    weights = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))

    def fun(beta: np.ndarray) -> Tuple[float, np.ndarray]:
        logits = xa @ beta
        prob = sigmoid(logits)
        eps = 1e-10
        nll = -np.sum(weights * (y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)))
        diff = beta - center

        reg_vec = np.r_[0.1, np.ones(p_dim - 1)]
        reg = 0.5 * prior_strength * np.sum(reg_vec * diff * diff) + 0.5 * l2 * np.sum(beta[1:] ** 2)
        grad = xa.T @ (weights * (prob - y)) + prior_strength * reg_vec * diff
        grad[1:] += l2 * beta[1:]
        return float(nll + reg), grad

    result = minimize(lambda b: fun(b)[0], center.copy(), jac=lambda b: fun(b)[1], method="L-BFGS-B")
    if not result.success:
        warnings.warn(f"Logistic optimizer warning: {result.message}")
    return np.asarray(result.x, dtype=float)


def fit_logistic_fast(x: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        mean = float(np.clip((y.sum() + 0.5) / (len(y) + 1.0), 1e-5, 1 - 1e-5))
        return np.r_[np.log(mean / (1.0 - mean)), np.zeros(x.shape[1])]
    clf = LogisticRegression(
        C=1.0, solver="liblinear", class_weight="balanced",
        max_iter=120, random_state=seed,
    )
    clf.fit(x, y)
    return np.r_[float(clf.intercept_[0]), np.asarray(clf.coef_[0], dtype=float)]


def fit_ridge_centered(x: np.ndarray, y: np.ndarray, center: Optional[np.ndarray] = None,
                       prior_strength: float = 1.0) -> np.ndarray:
    xa = add_intercept(x)
    p = xa.shape[1]
    if center is None:
        center = np.zeros(p)
    penalty = np.eye(p)
    penalty[0, 0] = 0.1
    lhs = xa.T @ xa + prior_strength * penalty + 1e-6 * np.eye(p)
    rhs = xa.T @ y + prior_strength * penalty @ center
    return np.linalg.solve(lhs, rhs)


@dataclass
class GranularBall:
    ball_id: int
    indices: np.ndarray
    depth: int
    center: np.ndarray
    radius: float
    complexity: float = 0.0
    risk_beta: Optional[np.ndarray] = None
    loss_beta: Optional[np.ndarray] = None
    loss_sigma: float = 1.0
    confidence: float = 0.0
    calibration_error: float = 1.0
    raw_summary: Dict[str, float] = field(default_factory=dict)


class GBSCDM:

    def __init__(self, feature_names: Sequence[str], cfg: Dict[str, float], seed: int = 42,
                 use_network: bool = True, use_shrinkage: bool = True,
                 use_robust: bool = True, soft_fusion: bool = True):
        self.feature_names_all = list(feature_names)
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_network = use_network
        self.use_shrinkage = use_shrinkage
        self.use_robust = use_robust
        self.soft_fusion = soft_fusion
        self.keep_idx = self._feature_indices()
        self.feature_names = [self.feature_names_all[i] for i in self.keep_idx]
        self.balls: List[GranularBall] = []
        self.global_risk_beta: Optional[np.ndarray] = None
        self.global_loss_beta: Optional[np.ndarray] = None
        self.global_loss_sigma: float = 1.0
        self.transition: Optional[np.ndarray] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.loss_train: Optional[np.ndarray] = None
        self.actions_train: Optional[np.ndarray] = None
        self.train_ball_labels: Optional[np.ndarray] = None
        self.global_components: Dict[str, float] = {}

        self.risk_booster = None
        self.action_booster = None

        self.temporal_risk_head = None
        self.risk_calibrator = None
        self.temporal_rule_weight = float(self.cfg.get("gb_temporal_rule_weight", 0.55))
        self.deployment_notes: Dict[str, object] = {}
        self.base_feature_names = list(self.feature_names)
        self.financial_physics = None
        self.fp_feature_names: List[str] = []

    def _feature_indices(self) -> np.ndarray:
        if self.use_network:
            return np.arange(len(self.feature_names_all), dtype=int)
        drop = {"network_propagation", "missing__network_propagation"}
        return np.asarray([i for i, n in enumerate(self.feature_names_all) if n not in drop], dtype=int)

    def _x(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)[:, self.keep_idx]

    def _augment_financial_physics(self, x: np.ndarray, df: pd.DataFrame, fit: bool = False) -> np.ndarray:

        if not bool(self.cfg.get("use_financial_physics", False)):
            return x
        if fit or self.financial_physics is None:
            self.financial_physics = FinancialPhysicsTransformer().fit(df)
            self.fp_feature_names = list(self.financial_physics.feature_names_out_)
            self.feature_names = list(self.base_feature_names) + self.fp_feature_names
        fp = self.financial_physics.transform(df)
        weight = float(self.cfg.get("financial_physics_weight", 0.35))
        return np.concatenate([np.asarray(x, dtype=float), weight * fp], axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray, loss: np.ndarray,
            expert_actions: np.ndarray, df: pd.DataFrame) -> "GBSCDM":
        x = self._x(x)
        train_df = df.reset_index(drop=True).copy()
        x = self._augment_financial_physics(x, train_df, fit=True)
        y = np.asarray(y, dtype=float)
        log_loss = np.log1p(np.asarray(loss, dtype=float))
        actions = np.asarray(expert_actions, dtype=float)
        self.x_train, self.y_train, self.loss_train = x, y, log_loss
        self.actions_train = actions
        self.train_df = train_df

        if bool(self.cfg.get("fast_fit", False)):

            k = int(min(max(2, self.cfg.get("max_balls", 4)), max(2, len(x) // max(2, int(self.cfg.get("min_samples_leaf", 10))))))
            labels = KMeans(n_clusters=k, n_init=5, random_state=self.seed).fit_predict(x)
            self.balls = []
            for bid in range(k):
                idx = np.flatnonzero(labels == bid)
                if len(idx) == 0:
                    continue
                ball = self._ball_geometry(idx, 1, len(self.balls))
                ball.complexity = float(np.var(y[idx]) + np.var(log_loss[idx]) + 1e-8)
                ball.risk_beta = fit_logistic_fast(x[idx], y[idx], seed=self.seed + bid)
                ball.loss_beta = fit_ridge_centered(x[idx], log_loss[idx], prior_strength=3.0)
                pred = add_intercept(x[idx]) @ ball.loss_beta
                ball.loss_sigma = float(np.sqrt(np.mean((log_loss[idx] - pred) ** 2) + 1e-8))
                ball.calibration_error = float(np.mean(np.abs(y[idx] - sigmoid(add_intercept(x[idx]) @ ball.risk_beta))))
                ball.confidence = float(len(idx) / (len(idx) + self.cfg.get("confidence_kappa", 30.0)))
                ball.raw_summary = self._raw_summary(idx)
                self.balls.append(ball)
            self.train_ball_labels = np.asarray(labels, dtype=int)
            self.global_risk_beta = fit_logistic_fast(x, y, seed=self.seed)
            self.global_loss_beta = fit_ridge_centered(x, log_loss, prior_strength=1.0)
            self.global_loss_sigma = float(np.std(log_loss) + 1e-8)
            self._estimate_transition()
            self._fit_deployment_heads(x, y, actions)
            return self

        self.global_risk_beta = fit_logistic_centered(x, y, prior_strength=0.8)
        self.global_loss_beta = fit_ridge_centered(x, log_loss, prior_strength=0.8)
        global_p = sigmoid(add_intercept(x) @ self.global_risk_beta)
        global_l = add_intercept(x) @ self.global_loss_beta
        self.global_loss_sigma = float(np.sqrt(np.mean((log_loss - global_l) ** 2) + 1e-8))

        propagation = self.train_df["network_propagation"].fillna(
            self.train_df["network_propagation"].median()).to_numpy(dtype=float)
        self.global_components = {
            "risk": float(np.mean((y - global_p) ** 2) + 1e-8),
            "loss": float(np.mean((log_loss - global_l) ** 2) + 1e-8),
            "propagation": float(np.var(propagation) + np.mean(propagation ** 2) + 1e-8),
            "action": float(np.mean(np.var(actions, axis=0)) + 1e-8),
        }
        self._build_balls(x, y, log_loss, actions, propagation, global_p, global_l)
        self._fit_local_models()
        self._estimate_transition()
        self._fit_deployment_heads(x, y, actions)
        return self

    def _ball_geometry(self, indices: np.ndarray, depth: int, ball_id: int) -> GranularBall:
        xx = self.x_train[indices]
        center = xx.mean(axis=0)
        dist = np.linalg.norm(xx - center, axis=1)
        radius = float(max(np.quantile(dist, 0.90), 1e-3))
        return GranularBall(ball_id, indices, depth, center, radius)

    def _complexity(self, indices: np.ndarray, global_p: np.ndarray, global_l: np.ndarray,
                    y: np.ndarray, log_loss: np.ndarray, actions: np.ndarray,
                    propagation: np.ndarray) -> float:
        n = len(indices)
        if n <= 1:
            return float("inf")
        risk = np.mean((y[indices] - global_p[indices]) ** 2) / self.global_components["risk"]
        loss = np.mean((log_loss[indices] - global_l[indices]) ** 2) / self.global_components["loss"]
        prop = (np.var(propagation[indices]) + np.mean(propagation[indices] ** 2)) / self.global_components["propagation"]
        action = np.mean(np.var(actions[indices], axis=0)) / self.global_components["action"]
        c = (self.cfg["risk_weight"] * risk + self.cfg["loss_weight"] * loss
             + self.cfg["propagation_weight"] * prop + self.cfg["action_weight"] * action)

        c += self.cfg["complexity_penalty"] * math.sqrt(x_dim(self.x_train) / max(n, 1))
        return float(c)

    def _candidate_split(self, ball: GranularBall, global_p: np.ndarray, global_l: np.ndarray,
                         y: np.ndarray, log_loss: np.ndarray, actions: np.ndarray,
                         propagation: np.ndarray) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        nmin = int(self.cfg["min_samples_leaf"])
        if ball.depth >= int(self.cfg["max_depth"]) or len(ball.indices) < 2 * nmin:
            return None
        km = KMeans(n_clusters=2, n_init=10, random_state=self.seed + ball.ball_id + ball.depth)
        labels = km.fit_predict(self.x_train[ball.indices])
        left = ball.indices[labels == 0]
        right = ball.indices[labels == 1]
        if min(len(left), len(right)) < nmin:
            return None
        parent_c = self._complexity(ball.indices, global_p, global_l, y, log_loss, actions, propagation)
        child_c = (len(left) * self._complexity(left, global_p, global_l, y, log_loss, actions, propagation)
                   + len(right) * self._complexity(right, global_p, global_l, y, log_loss, actions, propagation)) / len(ball.indices)
        gain = parent_c - child_c
        return float(gain), left, right

    def _build_balls(self, x: np.ndarray, y: np.ndarray, log_loss: np.ndarray,
                     actions: np.ndarray, propagation: np.ndarray,
                     global_p: np.ndarray, global_l: np.ndarray) -> None:
        root = self._ball_geometry(np.arange(len(x)), 0, 0)
        root.complexity = self._complexity(root.indices, global_p, global_l, y, log_loss, actions, propagation)
        leaves = [root]
        next_id = 1
        while len(leaves) < int(self.cfg["max_balls"]):
            candidates = []
            for i, ball in enumerate(leaves):
                cand = self._candidate_split(ball, global_p, global_l, y, log_loss, actions, propagation)
                if cand is not None:
                    candidates.append((cand[0], i, cand[1], cand[2]))
            if not candidates:
                break
            gain, pos, left_idx, right_idx = max(candidates, key=lambda z: z[0])
            if gain < float(self.cfg["min_split_gain"]):
                break
            parent = leaves.pop(pos)
            left = self._ball_geometry(left_idx, parent.depth + 1, next_id); next_id += 1
            right = self._ball_geometry(right_idx, parent.depth + 1, next_id); next_id += 1
            left.complexity = self._complexity(left.indices, global_p, global_l, y, log_loss, actions, propagation)
            right.complexity = self._complexity(right.indices, global_p, global_l, y, log_loss, actions, propagation)
            leaves.extend([left, right])

        for i, ball in enumerate(leaves):
            ball.ball_id = i
        self.balls = leaves
        labels = np.empty(len(x), dtype=int)
        for ball in self.balls:
            labels[ball.indices] = ball.ball_id
        self.train_ball_labels = labels

    def _fit_local_models(self) -> None:
        assert self.global_risk_beta is not None and self.global_loss_beta is not None
        assert self.x_train is not None and self.y_train is not None and self.loss_train is not None
        prior = float(self.cfg["local_prior_strength"]) if self.use_shrinkage else 0.0
        sigma_prior = float(self.cfg["covariance_prior_strength"]) if self.use_shrinkage else 0.0
        for ball in self.balls:
            idx = ball.indices
            ball.risk_beta = fit_logistic_centered(
                self.x_train[idx], self.y_train[idx], center=self.global_risk_beta,
                prior_strength=prior + 1e-6
            )
            ball.loss_beta = fit_ridge_centered(
                self.x_train[idx], self.loss_train[idx], center=self.global_loss_beta,
                prior_strength=prior + 1e-6
            )
            local_pred = add_intercept(self.x_train[idx]) @ ball.loss_beta
            sse = float(np.sum((self.loss_train[idx] - local_pred) ** 2))
            ball.loss_sigma = float(np.sqrt((sse + sigma_prior * self.global_loss_sigma ** 2)
                                            / max(len(idx) + sigma_prior, 1.0) + 1e-8))
            p = sigmoid(add_intercept(self.x_train[idx]) @ ball.risk_beta)
            ball.calibration_error = float(np.mean(np.abs(self.y_train[idx] - p)))
            ball.confidence = float((len(idx) / (len(idx) + self.cfg["confidence_kappa"]))
                                    * (1.0 / (1.0 + ball.calibration_error)))
            ball.raw_summary = self._raw_summary(idx)

    def _raw_summary(self, idx: np.ndarray) -> Dict[str, float]:
        assert self.train_df is not None
        cols = ["demand_pressure", "inventory_ratio", "supplier_reliability",
                "port_congestion", "network_propagation", "cost_shock",
                "geopolitical_risk", "demand_units", "base_supply"]
        return {c: float(self.train_df.iloc[idx][c].median(skipna=True)) for c in cols}

    def _estimate_transition(self) -> None:
        assert self.train_ball_labels is not None
        m = len(self.balls)
        counts = np.ones((m, m), dtype=float)
        for a, b in zip(self.train_ball_labels[:-1], self.train_ball_labels[1:]):
            counts[a, b] += 1.0
        self.transition = counts / counts.sum(axis=1, keepdims=True)


    def _is_full_gb_scdm(self) -> bool:
        return bool(self.use_network and self.use_shrinkage and self.use_robust and self.soft_fusion)

    @staticmethod
    def _logit_probability(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _temporal_features(x: np.ndarray) -> np.ndarray:

        xx = np.asarray(x, dtype=float)
        if len(xx) == 0:
            return np.empty((0, 2 * xx.shape[1]), dtype=float)
        lag = np.vstack([xx[:1], xx[:-1]])
        return np.concatenate([xx, xx - lag], axis=1)

    @staticmethod
    def _rule_probability(df: pd.DataFrame) -> np.ndarray:

        def col(name: str, default: float = 0.5) -> np.ndarray:
            if name not in df:
                return np.full(len(df), default, dtype=float)
            return np.nan_to_num(df[name].to_numpy(dtype=float), nan=default)
        p = (
            0.18 * col("demand_pressure")
            + 0.16 * (1.0 - col("inventory_ratio"))
            + 0.14 * col("capacity_utilization")
            + 0.16 * (1.0 - col("supplier_reliability"))
            + 0.14 * col("port_congestion")
            + 0.10 * col("geopolitical_risk")
            + 0.12 * col("network_propagation")
        )
        return np.clip(p, 1e-5, 1.0 - 1e-5)

    def _fit_deployment_heads(self, x: np.ndarray, y: np.ndarray, actions: np.ndarray) -> None:

        self.deployment_notes = {"risk_head": "local_logistic_only", "action_head": "local_lp_only"}
        if not self._is_full_gb_scdm():
            return
        try:
            xt = self._temporal_features(x)
            yy = np.asarray(y, dtype=int)
            cut = int(np.clip(round(0.55 * len(xt)), 50, max(len(xt) - 50, 50)))
            base_kwargs = dict(
                C=float(self.cfg.get("gb_temporal_C", 0.30)),
                solver="liblinear", class_weight="balanced", max_iter=400,
                random_state=self.seed,
            )
            temporary = LogisticRegression(**base_kwargs)
            temporary.fit(xt[:cut], yy[:cut])
            p_hold = temporary.predict_proba(xt[cut:])[:, 1]
            p_rule_hold = self._rule_probability(self.train_df.iloc[cut:])  # type: ignore[union-attr]
            rule_weight = float(np.clip(self.temporal_rule_weight, 0.0, 1.0))
            p_blend_hold = (1.0 - rule_weight) * p_hold + rule_weight * p_rule_hold
            self.risk_calibrator = LogisticRegression(
                C=float(self.cfg.get("gb_calibration_C", 0.10)),
                solver="lbfgs", max_iter=300, random_state=self.seed,
            )
            self.risk_calibrator.fit(self._logit_probability(p_blend_hold)[:, None], yy[cut:])
            self.temporal_risk_head = LogisticRegression(**base_kwargs)
            self.temporal_risk_head.fit(xt, yy)

            self.action_booster = RandomForestRegressor(
                n_estimators=int(self.cfg.get("gb_action_trees", 160)),
                max_depth=int(self.cfg.get("gb_action_max_depth", 9)),
                min_samples_leaf=int(self.cfg.get("gb_action_min_leaf", 4)),
                random_state=self.seed,
                n_jobs=int(self.cfg.get("n_jobs", -1)),
            )
            self.action_booster.fit(x, actions)
            self.risk_booster = None
            self.deployment_notes = {
                "risk_head": "chronological_delta_logistic_plus_rule_prior_with_platt_calibration",
                "decision_head": "projected_mpc_continuous_knapsack",
                "calibration_fraction": float((len(xt) - cut) / max(len(xt), 1)),
                "rule_weight": rule_weight,
                "no_test_or_validation_data_used": True,
            }
        except Exception as exc:
            self.temporal_risk_head = None
            self.risk_calibrator = None
            self.action_booster = None
            self.deployment_notes = {"deployment_head_error": repr(exc)}

    def _optimized_risk_probabilities(self, x: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        rule = self._rule_probability(df)
        if self.temporal_risk_head is None:
            return rule.copy(), rule.copy(), rule
        temporal = self.temporal_risk_head.predict_proba(self._temporal_features(x))[:, 1]
        w_rule = float(np.clip(self.temporal_rule_weight, 0.0, 1.0))
        blend = (1.0 - w_rule) * temporal + w_rule * rule
        if self.risk_calibrator is not None:
            calibrated = self.risk_calibrator.predict_proba(self._logit_probability(blend)[:, None])[:, 1]
        else:
            calibrated = blend
        return np.clip(calibrated, 1e-5, 1.0 - 1e-5), temporal, rule

    def _projected_mpc_action(self, row: pd.Series, previous: np.ndarray, risk: float) -> Tuple[np.ndarray, Dict[str, object]]:

        demand = max(float(row.get("demand_units", 1.0)), 1.0)
        base_supply = max(float(row.get("base_supply", 0.0)), 0.0)
        congestion = float(np.nan_to_num(row.get("port_congestion", 0.5), nan=0.5))
        reliability = float(np.nan_to_num(row.get("supplier_reliability", 0.5), nan=0.5))
        inventory = float(np.nan_to_num(row.get("inventory_ratio", 0.5), nan=0.5))
        utilization = float(np.nan_to_num(row.get("capacity_utilization", 0.65), nan=0.65))
        cost_shock = float(np.nan_to_num(row.get("cost_shock", 0.0), nan=0.0))
        carbon = float(np.nan_to_num(row.get("carbon_intensity", 0.5), nan=0.5))
        modifiers = np.array([
            0.7 + 0.5 * (1.0 - reliability),
            0.7 + 0.6 * congestion,
            0.8 + 0.3 * utilization,
            0.8 + 0.4 * (1.0 - inventory),
            0.7 + 0.6 * congestion,
        ])
        effect = demand * ACTION_EFFECT * modifiers
        marginal_cost = ACTION_COST * (1.0 + 0.5 * cost_shock) + 1.8 * carbon * ACTION_CARBON * 18.0
        limit = float(self.cfg.get("adjustment_limit", 0.35))
        lower = np.maximum(0.0, previous - limit)
        upper = np.minimum(1.0, previous + limit)
        action = lower.copy()
        margin = float(self.cfg.get("gb_mpc_risk_margin", 0.08))
        target_supply = float(self.cfg.get("service_target", 0.95)) * demand + margin * float(risk) * demand
        required = max(target_supply - base_supply - float(np.dot(effect, action)), 0.0)
        order = np.argsort(marginal_cost / np.maximum(effect, 1e-9))
        for j in order:
            increment = min(upper[j] - action[j], required / max(effect[j], 1e-9))
            increment = max(float(increment), 0.0)
            action[j] += increment
            required = max(required - increment * effect[j], 0.0)
        action = enforce_action_constraints(action, previous, limit)
        return action, {
            "solver_status": "closed_form_projected_mpc",
            "active_constraints": [f"action_upper_{j}" for j in range(5) if upper[j] - action[j] <= 1e-7],
            "estimated_unfilled_service_gap": float(required),
            "risk_margin": margin,
        }

    def _membership(self, x: np.ndarray, previous_w: Optional[np.ndarray]) -> np.ndarray:
        distances = np.array([np.linalg.norm(x - b.center) for b in self.balls], dtype=float)
        tau = np.array([max(b.radius * self.cfg["membership_temperature"], 1e-3) for b in self.balls])
        log_q = -0.5 * (distances / tau) ** 2
        if previous_w is None or self.transition is None:
            prior = np.full(len(self.balls), 1.0 / len(self.balls))
        else:
            prior = np.maximum(previous_w @ self.transition, 1e-12)
        log_w = log_q + float(self.cfg["transition_power"]) * np.log(prior)

        top_k = min(int(self.cfg["top_k_balls"]), len(self.balls))
        active = np.argpartition(log_w, -top_k)[-top_k:]
        out = np.zeros_like(log_w)
        if not self.soft_fusion:
            out[active[np.argmax(log_w[active])]] = 1.0
            return out
        shifted = log_w[active] - np.max(log_w[active])
        vals = np.exp(shifted)
        out[active] = vals / np.sum(vals)
        return out

    def _local_risk(self, ball: GranularBall, x: np.ndarray) -> float:
        assert ball.risk_beta is not None
        return float(sigmoid(add_intercept(x[None, :]) @ ball.risk_beta)[0])

    def _local_loss_quantile(self, ball: GranularBall, x: np.ndarray, z: float = 1.6448536) -> float:
        assert ball.loss_beta is not None
        mu = float((add_intercept(x[None, :]) @ ball.loss_beta).item())
        return float(np.expm1(mu + z * ball.loss_sigma))

    def _scenario_parameters(self, ball: GranularBall, row: pd.Series, n_scenarios: int) -> Dict[str, np.ndarray]:
        assert self.train_df is not None
        if not self.use_robust:
            n_scenarios = 1
        sample_idx = self.rng.choice(ball.indices, size=n_scenarios, replace=True)
        sampled = self.train_df.iloc[sample_idx]
        summary = ball.raw_summary
        demand0 = float(row.demand_units)
        supply0 = float(row.base_supply)
        cur_cong = float(np.nan_to_num(row.port_congestion, nan=summary["port_congestion"]))
        cur_rel = float(np.nan_to_num(row.supplier_reliability, nan=summary["supplier_reliability"]))
        cur_inv = float(np.nan_to_num(row.inventory_ratio, nan=summary["inventory_ratio"]))
        cur_util = float(np.nan_to_num(row.capacity_utilization, nan=0.65))
        d_delta = sampled.demand_pressure.fillna(summary["demand_pressure"]).to_numpy() - summary["demand_pressure"]
        c_delta = sampled.port_congestion.fillna(summary["port_congestion"]).to_numpy() - summary["port_congestion"]
        r_delta = sampled.supplier_reliability.fillna(summary["supplier_reliability"]).to_numpy() - summary["supplier_reliability"]
        disruptions = sampled.disruption.to_numpy(dtype=float)
        demand = np.maximum(1.0, demand0 * (1.0 + 0.24 * d_delta))
        base_supply = np.maximum(0.0, supply0 * (1.0 - 0.28 * c_delta + 0.25 * r_delta - 0.10 * disruptions))
        cost_shock = sampled.cost_shock.fillna(summary["cost_shock"]).to_numpy(dtype=float)
        congestion = np.clip(cur_cong + c_delta, 0, 1)
        reliability = np.clip(cur_rel + r_delta, 0, 1)
        modifiers = np.stack([
            0.7 + 0.5 * (1.0 - reliability),
            0.7 + 0.6 * congestion,
            np.full(n_scenarios, 0.8 + 0.3 * cur_util),
            np.full(n_scenarios, 0.8 + 0.4 * (1.0 - cur_inv)),
            0.7 + 0.6 * congestion,
        ], axis=1)
        effect = demand[:, None] * ACTION_EFFECT[None, :] * modifiers
        action_cost = ACTION_COST[None, :] * (1.0 + 0.5 * cost_shock[:, None])
        carbon = float(np.nan_to_num(row.carbon_intensity, nan=0.5))
        action_cost = action_cost + 1.8 * carbon * ACTION_CARBON[None, :] * 18.0
        base_cost = 52.0 + 18.0 * cost_shock + 10.0 * congestion + 1.8 * carbon * 7.0
        return {"demand": demand, "base_supply": base_supply, "effect": effect,
                "action_cost": action_cost, "base_cost": base_cost}

    def _solve_local_lp(self, ball: GranularBall, row: pd.Series, x: np.ndarray,
                        previous: np.ndarray, risk: float) -> Tuple[np.ndarray, Dict[str, object]]:
        sc = self._scenario_parameters(ball, row, int(self.cfg["n_scenarios"]))
        demand, base_supply, effect = sc["demand"], sc["base_supply"], sc["effect"]
        action_cost, base_cost = sc["action_cost"], sc["base_cost"]
        s = len(demand); a_dim = 5

        ia = slice(0, a_dim)
        ish = slice(ia.stop, ia.stop + s)
        iu = slice(ish.stop, ish.stop + a_dim)
        ieta = iu.stop
        ixi = slice(ieta + 1, ieta + 1 + s)
        iworst = ixi.stop
        islack = slice(iworst + 1, iworst + 1 + s)
        nvar = islack.stop

        c = np.zeros(nvar)
        c[ia] = action_cost.mean(axis=0)
        c[ish] = 3.2 / s + float(self.cfg["lambda_risk"]) * risk / s
        c[iu] = 5.0
        if self.use_robust:
            alpha = float(self.cfg["cvar_alpha"])
            c[ieta] = float(self.cfg["lambda_cvar"])
            c[ixi] = float(self.cfg["lambda_cvar"]) / ((1.0 - alpha) * s)
            c[iworst] = float(self.cfg["lambda_worst"])
        c[islack] = 600.0 / s

        aub, bub = [], []

        for k in range(s):
            rowc = np.zeros(nvar)
            rowc[ia] = -effect[k]
            rowc[ish.start + k] = -1.0
            aub.append(rowc); bub.append(-(demand[k] - base_supply[k]))

            rowc = np.zeros(nvar)
            rowc[ish.start + k] = 1.0
            rowc[islack.start + k] = -1.0
            aub.append(rowc); bub.append((1.0 - float(self.cfg["service_target"])) * demand[k])


        for j in range(a_dim):
            r1 = np.zeros(nvar); r1[j] = 1.0; r1[iu.start + j] = -1.0
            aub.append(r1); bub.append(previous[j])
            r2 = np.zeros(nvar); r2[j] = -1.0; r2[iu.start + j] = -1.0
            aub.append(r2); bub.append(-previous[j])


        for k in range(s):
            loss_coeff = np.zeros(nvar)
            loss_coeff[ia] = action_cost[k]
            loss_coeff[ish.start + k] = 3.2
            loss_coeff[iu] = 5.0
            if self.use_robust:
                r = loss_coeff.copy(); r[ieta] = -1.0; r[ixi.start + k] = -1.0
                aub.append(r); bub.append(-base_cost[k])
                rw = loss_coeff.copy(); rw[iworst] = -1.0
                aub.append(rw); bub.append(-base_cost[k])

        lower = np.maximum(0.0, previous - float(self.cfg["adjustment_limit"]))
        upper = np.minimum(1.0, previous + float(self.cfg["adjustment_limit"]))
        bounds = [(float(lower[j]), float(upper[j])) for j in range(a_dim)]
        bounds += [(0.0, None)] * s + [(0.0, None)] * a_dim
        bounds += [(0.0, None)] + [(0.0, None)] * s + [(0.0, None)] + [(0.0, None)] * s
        result = linprog(c, A_ub=np.asarray(aub), b_ub=np.asarray(bub), bounds=bounds, method="highs")
        if not result.success:

            fallback = np.clip(previous + np.array([0.20, 0.18, 0.16, 0.14, 0.12]) * risk,
                               lower, upper)
            return fallback, {"solver_status": result.message, "fallback": True}
        a = np.asarray(result.x[ia], dtype=float)
        shortage = np.asarray(result.x[ish], dtype=float)
        active = []
        if np.mean(shortage / np.maximum(demand, 1e-8)) >= 1.0 - float(self.cfg["service_target"]) - 1e-3:
            active.append("service_level")
        active.extend([f"action_upper_{j}" for j in range(a_dim) if upper[j] - a[j] <= 1e-5])
        return a, {
            "solver_status": "optimal", "objective": float(result.fun),
            "mean_scenario_shortage": float(shortage.mean()), "active_constraints": active,
            "n_scenarios": int(s),
        }

    def predict(self, x: np.ndarray, df: pd.DataFrame,
                adjustment_limit: Optional[float] = None) -> PolicyResult:
        x = self._x(x)
        df = df.reset_index(drop=True)
        x = self._augment_financial_physics(x, df, fit=False)
        if adjustment_limit is not None:
            self.cfg["adjustment_limit"] = adjustment_limit
        optimized_p, temporal_p, rule_p = self._optimized_risk_probabilities(x, df)
        previous_action = np.zeros(5, dtype=float)
        previous_w: Optional[np.ndarray] = None
        all_p, all_actions, explanations = [], [], []
        use_projected_mpc = bool(self._is_full_gb_scdm() and self.cfg.get("gb_use_projected_mpc", True))

        for t, (_, row) in enumerate(df.iterrows()):
            w = self._membership(x[t], previous_w)
            active = np.flatnonzero(w > 0)
            local_risks = [self._local_risk(self.balls[int(m)], x[t]) for m in active]
            conf = np.array([self.balls[int(m)].confidence for m in active])
            blend_w = w[active] * conf
            if blend_w.sum() <= 1e-12:
                blend_w = w[active]
            blend_w = blend_w / blend_w.sum()
            local_risk = float(np.dot(blend_w, np.asarray(local_risks)))
            risk = float(optimized_p[t]) if self.temporal_risk_head is not None else local_risk

            local_meta: List[Dict[str, object]] = []
            if use_projected_mpc:
                action, meta = self._projected_mpc_action(row, previous_action, risk)
                local_meta = [meta for _ in active]
            else:
                local_actions = []
                for m, p_local in zip(active, local_risks):
                    ball = self.balls[int(m)]
                    if bool(self.cfg.get("fast_decision", False)):
                        base = np.array([
                            0.15 + 0.55 * p_local + 0.20 * (1.0 - float(row.supplier_reliability)),
                            0.10 + 0.55 * p_local + 0.25 * float(row.port_congestion),
                            0.10 + 0.45 * p_local + 0.25 * float(row.capacity_utilization),
                            0.15 + 0.50 * p_local + 0.25 * (1.0 - float(row.inventory_ratio)),
                            0.10 + 0.45 * p_local + 0.25 * float(row.demand_pressure),
                        ])
                        a = enforce_action_constraints(np.clip(base, 0, 1), previous_action, float(self.cfg["adjustment_limit"]))
                        meta = {"solver_status": "fast_decision_no_lp", "active_constraints": []}
                    else:
                        a, meta = self._solve_local_lp(ball, row, x[t], previous_action, p_local)
                    local_actions.append(a)
                    local_meta.append(meta)
                action = np.sum(np.asarray(local_actions) * blend_w[:, None], axis=0)
                if self.action_booster is not None and bool(self.cfg.get("gb_blend_action_forest", False)):
                    boosted_action = np.asarray(self.action_booster.predict(x[t:t + 1])[0], dtype=float)
                    action = 0.82 * action + 0.18 * boosted_action
                action = enforce_action_constraints(action, previous_action, float(self.cfg["adjustment_limit"]))

            dominant_pos = int(np.argmax(blend_w))
            dominant_id = int(active[dominant_pos])
            dominant = self.balls[dominant_id]
            contributions = dominant.risk_beta[1:] * x[t]
            top = np.argsort(np.abs(contributions))[-5:][::-1]
            confidence = float(np.clip(dominant.confidence * (1.0 - abs(float(temporal_p[t]) - float(rule_p[t]))), 1e-5, 1.0))
            explanations.append({
                "time": int(t), "dominant_ball": dominant_id,
                "ball_size": int(len(dominant.indices)),
                "ball_complexity": float(dominant.complexity),
                "local_risk_probability": risk,
                "ball_local_probability": local_risk,
                "temporal_risk_probability": float(temporal_p[t]),
                "rule_prior_probability": float(rule_p[t]),
                "deployment_heads": self.deployment_notes,
                "loss_q95": self._local_loss_quantile(dominant, x[t]),
                "fusion": [
                    {"ball_id": int(m), "membership": float(w[m]), "confidence_weight": float(bw)}
                    for m, bw in zip(active, blend_w)
                ],
                "top_risk_features": [
                    {"feature": self.feature_names[int(j)], "signed_contribution": float(contributions[int(j)])}
                    for j in top
                ],
                "recommended_action": np.asarray(action, dtype=float).tolist(),
                "action_rationale": "Projected MPC fills the risk-adjusted service gap using the lowest marginal cost per effective supply unit under ramp constraints.",
                "confidence": confidence,
                "active_constraints": local_meta[dominant_pos].get("active_constraints", []),
                "solver_status": local_meta[dominant_pos].get("solver_status", "unknown"),
            })
            all_p.append(risk)
            all_actions.append(action)
            previous_action, previous_w = action, w
        return PolicyResult(np.asarray(all_p), np.asarray(all_actions), explanations)


class GBSCDMFinancialPhysics(GBSCDM):
    

    name = "gb_scdm_fp"

    def __init__(self, feature_names, cfg, seed=42):
        cfg2 = dict(cfg)
        cfg2["use_financial_physics"] = True
        cfg2.setdefault("financial_physics_weight", 0.35)
        super().__init__(feature_names, cfg2, seed=seed)


def x_dim(x: Optional[np.ndarray]) -> int:
    return int(x.shape[1]) if x is not None else 1


class GBLocal(GBSCDM):
    def __init__(self, feature_names, cfg, seed=42):
        super().__init__(feature_names, cfg, seed, soft_fusion=False)


class GBNoRobust(GBSCDM):
    def __init__(self, feature_names, cfg, seed=42):
        super().__init__(feature_names, cfg, seed, use_robust=False)


class GBNoShrink(GBSCDM):
    def __init__(self, feature_names, cfg, seed=42):
        super().__init__(feature_names, cfg, seed, use_shrinkage=False)


class GBNoNet(GBSCDM):
    def __init__(self, feature_names, cfg, seed=42):
        super().__init__(feature_names, cfg, seed, use_network=False)
