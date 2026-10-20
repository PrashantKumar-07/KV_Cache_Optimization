"""Fetch the exact upstream selectors required by the CPU tests and adapters."""
import argparse
from pathlib import Path
import subprocess

REFERENCES = (
    ("sota/snapkv", "https://github.com/FasterDecoding/SnapKV.git",
     "e216ddc84c5bd210378cbdbbba12ba02102aa640"),
    ("results/improvement_design_20261016/KVCache-Factory",
     "https://github.com/Zefan-Cai/KVCache-Factory.git",
     "68cd9551a63fedf362ddb0edb008badd04c24996"),
)
OPTIONAL_REFERENCES = (
    ("results/proper_baselines_20261016/IceCache-bf16",
     "https://github.com/yuzhenmao/IceCache.git",
     "f2afaffcd0959444b778daccbb327fead89e24ca"),
    ("results/proper_baselines_20261016/M-DCI",
     "https://github.com/yuzhenmao/M-DCI.git",
     "2141a1c842e3420b69843d0c1cd76ccc6ac72dfc"),
    ("results/proper_baselines_20261016/RocketKV",
     "https://github.com/NVlabs/RocketKV.git",
     "63637c8eb4b06b5eb83cd687723b0eeb19f48f65"),
    ("results/proper_baselines_20261016/rapids-cmake",
     "https://github.com/rapidsai/rapids-cmake.git",
     "69f5222465ec3c8c54f107fcf8750f040034e156"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--all", action="store_true",
                        help="Also fetch optional research baseline sources; does not build them")
    args = parser.parse_args()
    for relative, url, revision in REFERENCES + (OPTIONAL_REFERENCES if args.all else ()):
        target = args.root / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                            url, str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision],
                           check=True)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"],
                                         text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(target), "status", "--porcelain"],
                                        text=True)
        if actual != revision or dirty:
            raise RuntimeError(f"{target}: expected clean {revision}; preserve local changes first")
        print(f"{relative}: {actual}")


if __name__ == "__main__":
    main()
