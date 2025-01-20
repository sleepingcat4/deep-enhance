import logging
import torch
from pathlib import Path

RUN_NAME = "enhancer_stage2"

logger = logging.getLogger(__name__)

def get_source_url(relpath):
    return f"https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/{RUN_NAME}/{relpath}?download=true"

def get_target_path(relpath: str | Path, run_dir: str | Path):
    return Path(run_dir) / relpath

def download(run_dir: str | Path = None):
    run_dir = Path(run_dir) if run_dir else Path.cwd() / "model"
    run_dir.mkdir(parents=True, exist_ok=True)

    relpaths = ["hparams.yaml", "ds/G/latest", "ds/G/default/mp_rank_00_model_states.pt"]

    for relpath in relpaths:
        path = get_target_path(relpath, run_dir)
        if path.exists():
            continue
        url = get_source_url(relpath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path))

    return run_dir
