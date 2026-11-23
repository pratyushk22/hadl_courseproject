"""Make the embedding files needed by the covariance diagnostic available.

* Files that exist in the team Google Drive are DOWNLOADED from it (never regenerated):
      embedding_generation/odia_embeddings.pt                                (~665 MB)
      low_rank_method/odia_low_rank_run/final_weighted_train_draws_embeddings.pt (~114 MB)
* The real-train embeddings (embedding_generation/odia_real_train_embeddings.pt, 886 x 768) are
  not in the Drive, so they are created with Vanshika's own function
  covariance_matching.load_or_embed_real() - same model and recipe as covariance matching, and
  it aborts unless re-embedding stored pool rows reproduces them (cosine >= 0.99).

All three paths are git-ignored (*.pt). Needs `gdown` for the downloads and, for the real
embeddings, transformers + torch + internet access (intfloat/multilingual-e5-base, ~1.1 GB).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from . import config

CM_SCRIPT = config.PROJECT_ROOT / "covariance_matching" / "covariance_matching.py"


def fetch_from_drive(path: str | Path) -> Path:
    """Download a file listed in config.DRIVE_FILE_IDS to its repository path (if missing)."""
    path = Path(path)
    if path.is_file() and path.stat().st_size > 0:
        return path
    file_id = {Path(k).resolve(): v for k, v in config.DRIVE_FILE_IDS.items()}.get(path.resolve())
    if file_id is None:
        raise FileNotFoundError(f"{path} is missing and has no known Drive file id")
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("`pip install gdown` is needed to fetch embeddings from the team Drive") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    print(f"[drive] downloading {path.relative_to(config.PROJECT_ROOT)} from the team Drive ...", flush=True)
    result = gdown.download(id=file_id, output=str(tmp), quiet=True)
    if not result or not tmp.is_file() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download of {path.name} from the team Drive ({config.TEAM_DRIVE_FOLDER}) failed. "
                           "Drive sometimes refuses automated downloads of large files for a while; "
                           "retry later or download it manually to " + str(path))
    tmp.rename(path)
    print(f"[drive] saved {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)
    return path


def _load_cm_module():
    """Import covariance_matching.py by path without writing __pycache__ into that folder."""
    spec = importlib.util.spec_from_file_location("hadl_covariance_matching", CM_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


def ensure_real_train_embeddings() -> Path:
    """Create embedding_generation/odia_real_train_embeddings.pt with Vanshika's function."""
    target = Path(config.REAL_TRAIN_EMBEDDINGS_PT)
    if target.is_file() and target.stat().st_size > 0:
        return target
    import pandas as pd
    import torch

    cm = _load_cm_module()
    if Path(cm.REAL_EMB_PATH).resolve() != target.resolve():
        raise RuntimeError(f"covariance_matching.py writes {cm.REAL_EMB_PATH}, expected {target}")
    fetch_from_drive(cm.POOL_EMB_PATH)                 # needed by her recipe parity check
    pool_df = pd.read_parquet(cm.POOL_DF_PATH).reset_index(drop=True)
    pool_emb = torch.load(cm.POOL_EMB_PATH, map_location="cpu", weights_only=True)
    real_df = pd.read_json(cm.REAL_TRAIN_PATH, lines=True)
    print("[real embeddings] creating them with covariance_matching.load_or_embed_real()", flush=True)
    cm.load_or_embed_real(real_df, pool_df, pool_emb, recompute=False)
    return target


def ensure_all() -> tuple[Path, Path]:
    """(synthetic embeddings path, real-train embeddings path), fetching/creating as needed."""
    synth = fetch_from_drive(config.SYNTH_EMBEDDINGS_PT)
    real = ensure_real_train_embeddings()
    return synth, real
