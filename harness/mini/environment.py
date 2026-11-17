# SPDX-License-Identifier: Apache-2.0
# Vendored from mini-swe-agent's LocalEnvironment (MIT), plus a command guard and a
# per-session budget of commands, since the agents here run on the experimenter's machine.

from __future__ import annotations

import os
import platform
import re
import signal
import subprocess
from typing import Any

from pydantic import BaseModel

from .exceptions import Submitted

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

# Commands the sandbox refuses outright. The workspace is a copy and the harness, the hidden
# materials and the experimenter's home are outside it; nothing legitimate needs these.
BLOCKED_PATTERNS = (
    r"\bsudo\b", r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*\s+/(\s|$)", r"\bcurl\b", r"\bwget\b", r"\bssh\b", r"\bscp\b",
    r"\bnc\b", r"\bpip\s+install\b", r"\buv\s+pip\b", r"\bgit\s+push\b", r"~/\.", r"/Users/[^/\s]+/\.",
    r"\bchmod\s+[0-7]*7[0-7]*\s+/", r"\bkill\s+-9\s+-1\b", r"\bshutdown\b", r"\breboot\b", r"\bopen\s+http",
    r"\bosascript\b", r"\blaunchctl\b", r"\bsecurity\s+(?:find|dump|export|unlock|show)", r"\bopen\s+-[a-zA-Z]",
)  # fmt: skip

def report_in_comments(command: str) -> str:
    """The report of a submit command that carries it as comment lines under the echo
    (``echo <marker>`` / ``# FEATURE: ...`` / ``# REPORT: ...``), without the comment marks;
    empty when the lines under the echo are anything else."""
    lines = command.strip().splitlines()
    if len(lines) < 2 or lines[0].strip() != f"echo {SUBMIT_MARKER}":
        return ""
    rest = [line for line in lines[1:] if line.strip()]
    if not rest or not all(line.lstrip().startswith("#") for line in rest):
        return ""
    return "\n".join(re.sub(r"^\s*#\s?", "", line) for line in rest) + "\n"


# Variables of the harness process that no role's shell may inherit.
SECRET_NAME = re.compile(r"API_KEY|AUTH_TOKEN|TOKEN|SECRET|PASSWORD|CREDENTIAL|^SSH_", re.I)


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 120
    command_budget: int = 0  # 0 = unlimited
    blocked_patterns: tuple[str, ...] = BLOCKED_PATTERNS
    # A macOS sandbox profile (`sandbox-exec -f`): when set, every command runs under it and
    # with `env` alone as its environment; when empty, the command runs unconfined with the
    # harness's environment minus its secrets.
    sandbox_profile: str = ""


class LocalEnvironment:
    """Executes bash commands on the local machine, each in a fresh subshell in ``cwd``."""

    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.commands_run = 0
        self._blocked = [re.compile(p) for p in self.config.blocked_patterns]

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        if (hit := next((p.pattern for p in self._blocked if p.search(command)), None)) is not None:
            output = {
                "output": f"This command is not permitted in this workspace (matched {hit!r}).",
                "returncode": 126,
                "exception_info": "",
                "extra": {"blocked_by": hit},
            }
            self._check_finished(output)
            return output
        # The submit command itself is not held to the budget: a session that has used its
        # commands must still be able to hand in.
        submitting = command.lstrip().startswith(f"echo {SUBMIT_MARKER}")
        if 0 < self.config.command_budget <= self.commands_run and not submitting:
            return {
                "output": "The command budget for this session is used up. Submit your work now with "
                f"`echo {SUBMIT_MARKER}`.",
                "returncode": 125,
                "exception_info": "",
                "extra": {"budget_exhausted": True},
            }
        self.commands_run += 1
        try:
            if self.config.sandbox_profile:
                result = _run(
                    ["/usr/bin/sandbox-exec", "-f", self.config.sandbox_profile, "/bin/sh", "-c", command],
                    cwd, dict(self.config.env), timeout or self.config.timeout,
                )
            else:
                inherited = {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}
                result = _run(command, cwd, inherited | self.config.env, timeout or self.config.timeout)
            output = {
                "output": result.stdout,
                "returncode": result.returncode,
                "exception_info": "",
            }
        except Exception as e:
            raw = getattr(e, "output", None)
            raw = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
            output = {
                "output": raw,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        self._check_finished(output, command)
        return output

    def _check_finished(self, output: dict, command: str = "") -> None:
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_MARKER and output["returncode"] == 0:
            # The report is what the command printed after the marker; a report written under
            # the echo as comment lines prints nothing and is read from the command itself.
            submission = "".join(lines[1:])
            if not submission.strip():
                submission = report_in_comments(command)
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return {**self.config.model_dump(), **platform.uname()._asdict(), **kwargs}

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }


def _run(
    command: str | list[str], cwd: str, env: dict[str, str], timeout: int
) -> subprocess.CompletedProcess[str]:
    """Like subprocess.run, but kills the whole process group on timeout so no children are orphaned.
    A string is a shell command; a list is run as given (the sandboxed shell). The command has
    no input: one that reads its standard input (``cat`` with no file) meets the end of it at
    once and never whatever the harness itself was started with."""
    process = subprocess.Popen(
        command,
        shell=isinstance(command, str),
        text=True,
        cwd=cwd,
        env=env,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
        stdout, _ = process.communicate()
        raise subprocess.TimeoutExpired(command, timeout, output=stdout) from None
    return subprocess.CompletedProcess(command, process.returncode, stdout=stdout)
