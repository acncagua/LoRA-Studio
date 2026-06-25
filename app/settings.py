import os
from pathlib import Path

APP_NAME = "LoRA-Studio"
SD_SCRIPTS_RELEASE_TAG = "v0.10.5"
SD_SCRIPTS_RELEASE_COMMIT = "a1b48df"
SD_SCRIPTS_REPO_URL = "https://github.com/kohya-ss/sd-scripts.git"

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DB_PATH = Path(os.environ.get("LORA_STUDIO_DB", DATA_DIR / "app.db"))
DATASETS_DIR = ROOT_DIR / "datasets"
EXTERNAL_DIR = ROOT_DIR / "external"
SD_SCRIPTS_DIR = EXTERNAL_DIR / "sd-scripts"
RUNS_DIR = ROOT_DIR / "runs"
LOGS_DIR = ROOT_DIR / "logs"
EXPORTS_DIR = ROOT_DIR / "exports"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8768
DEMO_MODE = os.environ.get("LORA_STUDIO_DEMO_MODE", "").strip().lower() in {"1", "true", "yes", "on"}

for directory in (DATA_DIR, DB_PATH.parent, DATASETS_DIR, EXTERNAL_DIR, RUNS_DIR, LOGS_DIR, EXPORTS_DIR, EMBEDDINGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
