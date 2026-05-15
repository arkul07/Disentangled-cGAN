"""Lightweight MLflow experiment tracking helpers.

The helper degrades gracefully to no-op mode when tracking is disabled.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency fallback
    mlflow = None


@dataclass
class ExperimentTracker:
    """Small wrapper around MLflow for logging metrics and artifacts."""

    enabled: bool
    experiment_name: str
    tracking_uri: str
    run_name: Optional[str] = None
    active: bool = field(default=False, init=False)

    def start(self) -> None:
        """Start a tracking run if tracking is enabled."""
        if not self.enabled or mlflow is None:
            self.active = False
            return
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name)
        self.active = True

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a parameter dictionary."""
        if self.active and mlflow is not None:
            mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics for a given step."""
        if self.active and mlflow is not None:
            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        """Log a file artifact if it exists."""
        if self.active and mlflow is not None and Path(path).exists():
            mlflow.log_artifact(path)

    def end(self) -> None:
        """End the current run."""
        if self.active and mlflow is not None:
            mlflow.end_run()
        self.active = False


def create_tracker(enabled: bool, experiment_name: str, tracking_uri: str, run_name: Optional[str] = None) -> ExperimentTracker:
    """Factory for an ExperimentTracker."""
    return ExperimentTracker(
        enabled=enabled,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
    )
