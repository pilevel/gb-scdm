# GB-SCDM

> Intelligent Decision-Making for Global Supply Chain Disruption Risks Driven by Granular Ball.

---

<a id="english"></a>

## English

### Overview

GB-SCDM is a research-oriented Python prototype for supply chain disruption risk assessment and mitigation decision support. It combines granular ball partitioning, global and local predictive models, temporal transitions, robust scenario optimization, continuous action constraints, and optional financial-physics features.

The model is designed to:

- estimate supply chain risk probabilities over time;
- predict high-quantile losses under local operating regimes;
- recommend mitigation actions under service, cost, carbon, and ramp constraints;
- expose local model membership, feature contributions, active constraints, and solver status;
- augment operational stress data with financial-physics-inspired features.

The current repository is a **core algorithm snapshot**. It does not include a CLI, dataset, test suite, packaging metadata, or license file.

### Main Components

1. **Granular ball partitioning** using recursive K-Means splits controlled by risk, loss, propagation, action heterogeneity, and complexity penalties.
2. **Local shrinkage models** that estimate local logistic-risk and ridge-loss parameters around global coefficients.
3. **Temporal soft fusion** using ball distance, a learned transition matrix, and local confidence.
4. **Risk calibration** that can blend a temporal logistic head with a rule prior and apply Platt-style calibration.
5. **Mitigation optimization** through projected MPC or scenario-based linear programming with CVaR and worst-case penalties.
6. **Explainable outputs** containing the dominant ball, probabilities, confidence, top feature contributions, actions, and active constraints.
7. **Financial-physics augmentation** with distance-to-default, default-probability, liquidity-pressure, network-energy, and tail-loss proxies.

### Installation

Python 3.9 or newer is recommended. Run examples from the repository root so that imports under `src` resolve correctly.

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install numpy pandas scipy scikit-learn
```

The repository currently has no `requirements.txt` or `pyproject.toml`. Pin verified dependency versions before publishing reproducible experiments.

### API Summary

#### Training

```python
model.fit(x, y, loss, expert_actions, df)
```

- `x`: numeric matrix with shape `(n_samples, n_features)`;
- `y`: binary risk or disruption labels;
- `loss`: non-negative loss targets, transformed internally with `log1p`;
- `expert_actions`: expert action matrix with shape `(n_samples, 5)`;
- `df`: row-aligned raw supply-chain state data.

#### Prediction

```python
result = model.predict(x, df)
```

The returned `PolicyResult` contains:

- `probabilities`: risk probabilities;
- `actions`: continuous mitigation actions with shape `(n_samples, 5)`;
- `explanations`: per-step model, feature, confidence, and optimization metadata.

### Action Order

The five action columns always follow this order:

```python
[
    "alternate_supplier",
    "alternate_route",
    "capacity_expansion",
    "safety_stock",
    "demand_prioritisation",
]
```

Each action is constrained to `[0, 1]`. The `adjustment_limit` configuration controls the maximum change between consecutive time steps.

### Recommended Data Columns

For the complete training, explanation, and optimization path, provide:

```text
demand_pressure
inventory_ratio
capacity_utilization
supplier_reliability
port_congestion
transport_delay
cost_shock
network_propagation
geopolitical_risk
demand_units
base_supply
carbon_intensity
disruption
```

Some helper functions define defaults for missing columns, but several main-model paths access fields directly. Validate the input schema, missing values, ranges, and units before training.

### Model Variants

| Class | Description |
| --- | --- |
| `GBSCDM` | Full model. |
| `GBSCDMFinancialPhysics` | Enables financial-physics feature augmentation. |
| `GBLocal` | Disables soft fusion and selects one local ball. |
| `GBNoRobust` | Disables multi-scenario robust optimization. |
| `GBNoShrink` | Disables local-to-global parameter shrinkage. |
| `GBNoNet` | Removes `network_propagation` and its missingness indicator from `x`. |

### Evaluation Utilities

`src/env/supply_chain_env.py` provides:

- `enforce_action_constraints(...)` for one-step action projection;
- `project_action_sequence(...)` for sequential projection;
- `expert_action_from_state(...)` for heuristic expert actions;
- `evaluate_decision_outcomes(...)` for service, shortage, cost, carbon, and adjustment metrics.

### Operational Notes

- Preserve chronological ordering because the model learns ball transitions and temporal-difference features.
- Scale features consistently; Euclidean ball membership is sensitive to feature units.
- Prediction is sequential, and each action depends on the previous action.
- Passing `adjustment_limit` to `predict` mutates the model configuration; avoid sharing one mutable instance across concurrent requests.
- Treat this repository as research code. Add validation, monitoring, human approval, tests, persistence, logging, and benchmark studies before operational deployment.

### License

No `LICENSE` file was included in the supplied repository. The copyright holder should select and add an appropriate license before public distribution, modification, or reuse.

---

## Contributing
  
Bug fixes, tests, documentation, reproducibility improvements, and production hardening are welcome. Please preserve API compatibility where possible and add tests for new behavior.
