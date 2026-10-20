"""Download the pinned Mistral snapshot without server-specific paths."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--revision", default="63a8b081895390a26e140280378bc85ec8bce07a",
                        help="Pinned snapshot used by the recorded experiments")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "models/mistral-7b-instruct")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    path = snapshot_download(repo_id=args.repo, revision=args.revision,
                             local_dir=str(args.output),
                             cache_dir=str(args.cache_dir) if args.cache_dir else None)
    print(path)


if __name__ == "__main__":
    main()
