"""Atomic evidence writes and streamed file identity checks."""
import hashlib,json
from pathlib import Path

def write_json(path, value):
    """Atomically replace a task-owned JSON record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def digest(path):
    """Compute a file digest without materializing a model in memory."""
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 2**20), b""):
            result.update(block)
    return result.hexdigest()
