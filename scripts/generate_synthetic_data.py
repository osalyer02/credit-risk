"""Generate a synthetic credit-risk training dataset for local demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic credit-risk training data")
    parser.add_argument("--rows", type=int, default=2500, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/credit_train.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def build_dataset(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    annual_income = rng.normal(85000, 24000, rows).clip(20000, 240000)
    loan_amount = rng.normal(14000, 8000, rows).clip(1000, 70000)
    dti = rng.normal(17.5, 8.0, rows).clip(0, 45)
    fico_range_low = rng.integers(580, 790, size=rows)
    fico_range_high = np.minimum(fico_range_low + rng.integers(0, 20, size=rows), 850)
    revolving_utilization = rng.normal(42, 24, rows).clip(0, 140)
    open_accounts = rng.integers(1, 22, size=rows)
    delinquencies_2y = rng.choice([0, 1, 2, 3, 4], p=[0.69, 0.19, 0.08, 0.03, 0.01], size=rows)

    loan_to_income = loan_amount / annual_income
    credit_score_mid = (fico_range_low + fico_range_high) / 2
    open_account_pressure = np.maximum(0, 12 - open_accounts)

    # Keep terms bounded to avoid numerical instability while creating realistic signal.
    logits = (
        -4.0
        + 2.6 * loan_to_income
        + 0.048 * dti
        + 0.016 * revolving_utilization
        + 0.48 * delinquencies_2y
        + 0.07 * open_account_pressure
        + 0.95 * ((700 - credit_score_mid) / 90)
    )

    default_probability = _sigmoid(logits).clip(1e-5, 1 - 1e-5)
    default_flag = (rng.random(rows) < default_probability).astype(int)

    return pd.DataFrame(
        {
            "application_id": [f"app-{index:06d}" for index in range(rows)],
            "annual_income": annual_income.round(2),
            "loan_amount": loan_amount.round(2),
            "dti": dti.round(3),
            "fico_range_low": fico_range_low,
            "fico_range_high": fico_range_high,
            "revolving_utilization": revolving_utilization.round(3),
            "open_accounts": open_accounts,
            "delinquencies_2y": delinquencies_2y,
            "default_flag": default_flag,
        }
    )


def main() -> None:
    args = parse_args()
    if args.rows <= 50:
        raise ValueError("--rows must be greater than 50 to support train/val/test splits")

    dataset = build_dataset(rows=args.rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    default_rate = float(dataset["default_flag"].mean())
    print(
        f"Wrote {len(dataset)} rows to {args.output} "
        f"(default_rate={default_rate:.3f}, seed={args.seed})"
    )


if __name__ == "__main__":
    main()
