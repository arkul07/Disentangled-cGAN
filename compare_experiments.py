"""Summarize MLflow experiments into a report-ready CSV and terminal table.

This script reads the local MLflow SQLite tracking database and extracts the
latest metrics from the baseline and disentangled experiments so you can compare
training runs in your report.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from mlflow.tracking import MlflowClient

from config import TRACKING_URI, EXPERIMENT_NAME_BASELINE, EXPERIMENT_NAME_DISENTANGLED, OUTPUT_DIR


@dataclass
class RunSummary:
    experiment: str
    run_name: str
    run_id: str
    status: str
    start_time: str
    end_time: str
    metrics: Dict[str, float]
    params: Dict[str, str]


METRIC_KEYS = ["d_loss", "g_loss", "g_adv_loss", "info_loss", "g_total_loss"]
PARAM_KEYS = ["latent_dim", "latent_dim_c", "latent_dim_n", "lambda_info", "batch_size", "epochs", "lr"]


def _format_time(ts: Optional[int]) -> str:
    if ts is None:
        return ""
    return pd.to_datetime(ts, unit="ms").strftime("%Y-%m-%d %H:%M:%S")


def _extract_latest_run(client: MlflowClient, experiment_name: str) -> Optional[RunSummary]:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None

    run = runs[0]
    metrics = {k: run.data.metrics[k] for k in METRIC_KEYS if k in run.data.metrics}
    params = {k: run.data.params[k] for k in PARAM_KEYS if k in run.data.params}

    return RunSummary(
        experiment=experiment_name,
        run_name=run.data.tags.get("mlflow.runName", ""),
        run_id=run.info.run_id,
        status=run.info.status,
        start_time=_format_time(run.info.start_time),
        end_time=_format_time(run.info.end_time),
        metrics=metrics,
        params=params,
    )


def _metric_value(summary: Optional[RunSummary], key: str) -> str:
    if summary is None:
        return ""
    value = summary.metrics.get(key)
    return "" if value is None else f"{value:.4f}"


def _param_value(summary: Optional[RunSummary], key: str) -> str:
    if summary is None:
        return ""
    return summary.params.get(key, "")


def build_rows() -> List[Dict[str, str]]:
    client = MlflowClient(tracking_uri=TRACKING_URI)
    baseline = _extract_latest_run(client, EXPERIMENT_NAME_BASELINE)
    disentangled = _extract_latest_run(client, EXPERIMENT_NAME_DISENTANGLED)

    rows = []
    for name, summary in [("baseline", baseline), ("disentangled", disentangled)]:
        rows.append(
            {
                "model": name,
                "experiment": summary.experiment if summary else "",
                "run_name": summary.run_name if summary else "",
                "run_id": summary.run_id if summary else "",
                "status": summary.status if summary else "",
                "start_time": summary.start_time if summary else "",
                "end_time": summary.end_time if summary else "",
                "d_loss": _metric_value(summary, "d_loss"),
                "g_loss": _metric_value(summary, "g_loss"),
                "g_adv_loss": _metric_value(summary, "g_adv_loss"),
                "info_loss": _metric_value(summary, "info_loss"),
                "g_total_loss": _metric_value(summary, "g_total_loss"),
                "latent_dim": _param_value(summary, "latent_dim"),
                "latent_dim_c": _param_value(summary, "latent_dim_c"),
                "latent_dim_n": _param_value(summary, "latent_dim_n"),
                "lambda_info": _param_value(summary, "lambda_info"),
                "batch_size": _param_value(summary, "batch_size"),
                "epochs": _param_value(summary, "epochs"),
                "lr": _param_value(summary, "lr"),
            }
        )
    return rows


def print_table(rows: List[Dict[str, str]]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        print("No runs found.")
        return
    print(df.to_string(index=False))


def save_csv(rows: List[Dict[str, str]]) -> Path:
    out_dir = Path(OUTPUT_DIR) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "experiment_comparison.csv"
    if rows:
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return out_path


def main() -> None:
    rows = build_rows()
    print_table(rows)
    out_path = save_csv(rows)
    print(f"\nSaved comparison CSV to: {out_path}")


if __name__ == "__main__":
    main()
