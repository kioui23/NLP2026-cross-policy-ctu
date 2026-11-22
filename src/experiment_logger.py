
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# 1. Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "hw2_results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Experiment Configuration
# ============================================================

EXPERIMENT_CONFIG = {
    "retrieval_top_k": 2,
    "max_retrieval_rounds": 4,
    "temperature": 0.0,
    "seed": 42,
}


# ============================================================
# 3. Run ID
# ============================================================

def create_run_id():
    """Create a unique ID for an experiment run."""
    return str(uuid.uuid4())


# Default run ID for modules that use a shared experiment context.
RUN_ID = create_run_id()


# ============================================================
# 4. Timestamp Utility
# ============================================================

def utc_now():
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# 5. JSONL Logging
# ============================================================

_JSONL_LOCKS = {}
_JSONL_LOCKS_GUARD = threading.Lock()


def _get_file_lock(path):
    """
    Get a thread lock associated with a specific file path.

    Prevents concurrent threads from writing to the same JSONL file
    at the same time.
    """
    path_key = str(path.resolve())

    with _JSONL_LOCKS_GUARD:
        if path_key not in _JSONL_LOCKS:
            _JSONL_LOCKS[path_key] = threading.Lock()

        return _JSONL_LOCKS[path_key]


def append_jsonl(record, filename, directory=None):
    """
    Append a record to a JSONL file.

    Args:
        record: Dictionary to save.
        filename: Target JSONL filename.
        directory: Optional directory override.

    Returns:
        Path to the saved file.
    """
    if directory is None:
        directory = RESULTS_DIR

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / filename
    lock = _get_file_lock(path)

    serialized_record = json.dumps(
        record,
        ensure_ascii=False,
        default=str,
    )

    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(serialized_record + "\n")
            f.flush()
            os.fsync(f.fileno())

    return path


def load_jsonl_records(path):
    """
    Load records from a JSONL file.

    Corrupted lines are skipped with a warning.
    """
    path = Path(path)

    if not path.exists():
        return []

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                print(
                    f"[Warning] Corrupted JSONL record: "
                    f"{path.name}:{line_number}"
                )
                print(error)

    return records


# ============================================================
# 6. Manifest Utility
# ============================================================

def save_manifest(manifest, path):
    """
    Save experiment metadata as a JSON manifest.

    Args:
        manifest: Metadata dictionary.
        path: Destination JSON file path.

    Returns:
        Path to the saved manifest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    return path


def load_manifest(path):
    """Load a JSON manifest."""
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 7. Checkpoint Logging
# ============================================================

def append_jsonl_checkpoint(path, record):
    """
    Append a checkpoint record with flush and fsync.

    This is useful for long-running experiments where preserving
    completed records is important.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lock = _get_file_lock(path)

    serialized_record = json.dumps(
        record,
        ensure_ascii=False,
        default=str,
    )

    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(serialized_record + "\n")
            f.flush()
            os.fsync(f.fileno())


# ============================================================
# 8. Logging Initialization
# ============================================================

print("Experiment logger initialized.")
print("Project root:", PROJECT_ROOT)
print("Results directory:", RESULTS_DIR)
print("RUN_ID:", RUN_ID)
