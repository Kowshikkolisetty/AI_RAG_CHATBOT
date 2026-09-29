import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"

for env_path in [SRC_DIR / ".env", BASE_DIR / ".env"]:
    if env_path.exists():
        load_dotenv(env_path, override=True)

# Keep compatibility with direct local execution from the repo root or src folder.
load_dotenv(".env", override=True)


def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is not None:
        return value.strip()
    return default


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
