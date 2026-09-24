"""Repository and results locations."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = Path(os.environ.get("RAW_DIR", DATA_DIR / "raw"))
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"


def results_root() -> Path:
    """Where result CSVs go. On Colab the setup cell points this at Google Drive
    so results survive a disconnect; locally it defaults to the repository."""
    return Path(os.environ.get("RESULTS_ROOT", REPO_ROOT))


def results_path(dataset_folder: str, experiment_folder: str, smoke: bool) -> Path:
    suffix = "__SMOKE" if smoke else ""
    name = f"{dataset_folder}__{experiment_folder}{suffix}.csv"
    return results_root() / dataset_folder / experiment_folder / "results" / name
