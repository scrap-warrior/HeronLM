"""
Shared test runner.

Two roles, and the reason for the second one is state rather than convenience. Each suite can still be
run on its own, which is what each module docstring says. When several suites run together, each one runs
in its own process: torch's settings that matter to these tests are process-wide - the default dtype, the
TF32 flags the Phase 0 training tests set, and the RNG state - so a suite that runs after another in the
same process is measured against settings it did not set. A runner that only aggregates cannot change a
result, and that is the whole point of it.

Run: python tests/run_tests.py
     python tests/run_tests.py test_phase0.py       one suite
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Every suite is measured at TINY scale, so this needs no GPU. `preflight.py` is deliberately not a
# suite: it measures params, memory and throughput at gate scale, which is a number a CI job cannot
# reproduce on a runner and should not try to.
SUITES = ["test_reference.py", "test_converter.py", "test_loader.py", "test_phase0.py", "test_phase1.py"]


def run(namespace: dict) -> int:
    """
    The loop each module used to carry.

    Name order is the sorted order, so a suite reports the same sequence it reported before the loop was
    shared. Exceptions are reported rather than allowed to end the suite, because a suite that stops at
    the first error hides the rest of itself: the exit code only means something when it counted every
    test it was given.
    """
    failures = 0
    for name, fn in sorted(namespace.items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
            except Exception as exc:
                failures += 1
                print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    return failures


def run_suite(suite: str) -> tuple[int, int]:
    """Run one suite in a fresh process and return (tests counted, failures)."""
    path = HERE / suite
    # The child is writing to a pipe, and a pipe uses the locale encoding rather than UTF-8, which is why
    # a test that prints a merge string raises on Windows. UTF-8 is fixed here so the runner does not
    # depend on the machine's locale, and the bytes are written straight through rather than decoded and
    # re-encoded.
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), capture_output=True, env=env)
    # The text layer buffers, and a pipe is buffered until it is flushed, so the header a suite is running
    # under would reach the log after that suite's own output. Flushing before writing the child's bytes is
    # what keeps a failure readable as a failure of a named suite.
    sys.stdout.flush()
    sys.stdout.buffer.write(result.stdout)
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
    lines = result.stdout.split(b"\n")
    counted = sum(1 for line in lines if line[:6] in (b"ok    ", b"FAIL  ", b"ERROR "))
    failed = sum(1 for line in lines if line[:6] in (b"FAIL  ", b"ERROR "))
    if result.returncode and not failed:
        # A non-zero exit with no FAIL or ERROR line means the suite died before it could report, which is
        # a failure of the run rather than a failure of a test, and it must not read as a passing suite.
        print(f"ERROR {suite}: exited {result.returncode} without reporting a test")
        failed = 1
    return counted, failed


def main(argv: list[str]) -> int:
    suites = argv or SUITES
    total = 0
    failed = 0
    for suite in suites:
        print(f"=== {suite} ===")
        counted, failures = run_suite(suite)
        total += counted
        failed += failures
        print(f"      {counted} tests, {failures} failed")
    print(f"\nTOTAL {total} tests, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
