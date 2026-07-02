"""
src/utils/io.py — I/O helpers (parquet cache read/write)
"""
from pathlib import Path
from typing import Any
from configs.settings import get_logger

logger = get_logger(__name__)


def save_parquet(df: Any, path: Path) -> None:
    """Persist a pandas DataFrame to parquet with snappy compression."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Saved parquet: {path} ({path.stat().st_size/1e6:.1f} MB)")


def load_parquet(path: Path) -> Any:
    """Load a parquet file into a pandas DataFrame."""
    import pandas as pd
    logger.info(f"Loading parquet: {path}")
    return pd.read_parquet(path, engine="pyarrow")
