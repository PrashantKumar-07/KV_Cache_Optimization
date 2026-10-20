"""Materialize the six exact LongBench JSONL files used by recorded evaluations."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/longbench"))
    parser.add_argument("--archive", type=Path, help="Use an existing official data.zip offline")
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("longbench_manifest.json").read_text())
    archive = args.archive or Path(hf_hub_download(
        manifest["repo"], manifest["archive"], repo_type="dataset", revision=manifest["revision"]))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("LongBench archive differs from the recorded source")
    with zipfile.ZipFile(archive) as bundle:
        contents = {name: bundle.read("data/" + name) for name in manifest["files"]}
    for name, content in contents.items():
        if hashlib.sha256(content).hexdigest() != manifest["files"][name]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
        target = args.output / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"Preserve conflicting dataset before retrying: {target}")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in contents.items():
        (args.output / name).write_bytes(content)
    print(f"Verified {len(contents)} source files: {args.output.resolve()}")


if __name__ == "__main__":
    main()
