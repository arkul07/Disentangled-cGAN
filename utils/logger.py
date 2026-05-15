"""Simple logger utilities for concise training output."""
from typing import Any
import time


def log_epoch(epoch: int, total_epochs: int, d_loss: float, g_loss: float) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] Epoch {epoch}/{total_epochs} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f}")
