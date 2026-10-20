"""Reuse the policy-only profiler for the ordinary verified native-reuse path."""
import hashlib
from pathlib import Path
import profile_cpu_batched as driver
from compare_reuse_visible import RecordedNativePolicy
from native_reuse_visible import kernel


class ProfileNative(driver.ProfileCalls, RecordedNativePolicy):
    pass


def main():
    original_policy, original_write = driver.ProfileDirectPolicy, driver.atomic_json
    files = list(Path('src').glob('*.py')) + list(Path('experiments').glob('*.py')) + list(Path('experiments').glob('*.cu')) + list(Path('experiments').glob('*.cpp'))
    identity = dict(actual_profile_policy='ordinary_native_reuse',
        reuse_binary_sha256=hashlib.sha256(Path(kernel().__file__).read_bytes()).hexdigest(),
        native_profile_source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

    def write(path, report):
        report.update(identity)
        original_write(path, report)

    driver.ProfileDirectPolicy, driver.atomic_json = ProfileNative, write
    try:
        driver.main()
    finally:
        driver.ProfileDirectPolicy, driver.atomic_json = original_policy, original_write


if __name__ == '__main__':
    main()
