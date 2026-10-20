"""Reproducible runtime identity and crash-safe experiment checkpoints."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def model_artifact_identity(location):
    """Hash local model/tokenizer artifacts; remote IDs must be revision-pinned."""
    directory = Path(location)
    if not directory.is_dir():
        raise ValueError('Reproducible evaluation requires a local model snapshot')
    files = sorted(p for p in directory.rglob('*') if p.is_file()
                   and '__pycache__' not in p.parts
                   and p.suffix in ('.json', '.model', '.txt', '.jinja', '.py', '.so', '.yaml', '.tiktoken', '.spm', '.bpe'))
    index = directory / 'model.safetensors.index.json'
    if index.exists():
        files += [directory / name for name in sorted(set(json.loads(index.read_text())['weight_map'].values()))]
    else:
        files += sorted(directory.glob('*.safetensors'))
        if not any(path.suffix == '.safetensors' for path in files):
            files += sorted(directory.glob('pytorch_model*.bin'))
    hashes = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        hashes[str(path.relative_to(directory))] = digest.hexdigest()
    return {'snapshot_path': str(directory.resolve()), 'artifact_sha256': hashes}


def read_jsonl(path):
    """Read physical JSONL records; Unicode separators inside strings are data."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def runtime_identity():
    files = sorted(p for folder in ("src", "experiments", "sota")
                   for p in (ROOT / folder).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".cpp", ".cu", ".h", ".hpp", ".sh", ".json"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    packages = {}
    for name in ("torch", "transformers", "datasets", "numpy", "accelerate",
                 "huggingface_hub", "rouge", "ninja"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {"git_revision": revision, "source_sha256": digest.hexdigest(),
            "packages": packages}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         delete=False) as stream:
            temporary = stream.name
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def resume_results(path, config):
    if not Path(path).exists():
        return {}
    data = json.loads(Path(path).read_text())
    if data.get("config") != config:
        raise ValueError(f"Configuration/runtime mismatch in {path}; use a new output path")
    return data["results"]
