"""Sandboxed execution of generated programs against stdin/stdout test pairs.

Pilot-grade isolation: fresh temp directory, minimal environment, wall-clock
timeout per test, and CPU/address-space rlimits where the platform supports
them. This is not a container; production runs need container isolation with
filesystem and network restrictions (see wiki/Coding-Domain.md).
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class TestOutcome:
    input: str
    expected: str
    actual: str
    passed: bool
    error: Optional[str] = None


@dataclass
class ExecutionReport:
    outcomes: List[TestOutcome] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def passed(self) -> int:
        return sum(o.passed for o in self.outcomes)

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.outcomes else 0.0

    @property
    def error(self) -> Optional[str]:
        errors = [o.error for o in self.outcomes if o.error]
        return errors[0] if errors else None

    def feedback_text(self, max_cases: int = 3) -> str:
        """Compact failure summary usable as GEPA/controller feedback."""
        lines = [f"passed {self.passed}/{self.total} tests"]
        for o in self.outcomes:
            if not o.passed and len(lines) <= max_cases:
                lines.append(
                    f"input={o.input[:80]!r} expected={o.expected[:60]!r} got={o.actual[:60]!r}"
                    + (f" error={o.error[:120]!r}" if o.error else "")
                )
        return "\n".join(lines)


def _limits() -> None:
    """Preexec hook: CPU seconds and address-space caps (best effort per platform)."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ValueError, OSError):
        pass


def run_program_against_tests(
    code: str, tests: List[Dict[str, str]], timeout: float = 5.0
) -> ExecutionReport:
    """Run `code` in a fresh subprocess per test; compare stripped stdout."""
    report = ExecutionReport()
    with tempfile.TemporaryDirectory() as tmp:
        program = Path(tmp) / "program.py"
        program.write_text(code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": tmp, "TMPDIR": tmp}
        for test in tests:
            start = time.time()
            actual, error = "", None
            try:
                result = subprocess.run(
                    [sys.executable, str(program)],
                    input=test["input"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tmp,
                    env=env,
                    preexec_fn=_limits,
                )
                actual = (result.stdout or "").strip()
                if result.returncode != 0:
                    error = (result.stderr or "").strip()[-400:] or f"exit code {result.returncode}"
            except subprocess.TimeoutExpired:
                error = f"timeout after {timeout}s"
            report.seconds += time.time() - start
            report.outcomes.append(
                TestOutcome(
                    input=test["input"],
                    expected=test["output"],
                    actual=actual,
                    passed=(error is None and actual == test["output"].strip()),
                    error=error,
                )
            )
    return report
