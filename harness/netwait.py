# SPDX-License-Identifier: Apache-2.0

"""Waiting out a network outage. The roles' HTTP calls and the gate's evaluations stall
until the API hosts are reachable again, instead of scoring the outage as the subject's
failures or crashing a run that has cost hours."""

from __future__ import annotations

import os
import re
import socket
import sys
import time
from urllib.parse import urlsplit

HOSTS = ("api.anthropic.com", "api.openai.com")
NETWORK_EXIT = 4  # run_eval: the API was unreachable, nothing was scored
# A provider that answers, and refuses the account (its credit ran out, its quota is used
# up). Like an outage this says nothing about the subject or a role: it is waited out, and
# what it interrupted is not scored.
ACCOUNT_ERROR = re.compile(r"credit balance is too low|insufficient_quota|exceeded your current quota|positive balance to do inference|add balance manually", re.I)
ACCOUNT_PAUSE_SECONDS = 120.0
ACCOUNT_WAIT_SECONDS = 7200.0


def account_error(text: object) -> bool:
    return bool(ACCOUNT_ERROR.search(str(text)))


def _proxy() -> tuple[str, int] | None:
    for key in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(key)
        if value:
            parts = urlsplit(value if "://" in value else f"http://{value}")
            return parts.hostname or value, parts.port or 8080
    return None


def online(hosts: tuple[str, ...] = HOSTS, timeout: float = 10.0) -> bool:
    """Whether the API hosts are reachable: through the configured proxy when there is one
    (direct names may not resolve on such a network), else by name and a TCP connection."""
    try:
        proxy = _proxy()
        if proxy is not None:
            with socket.create_connection(proxy, timeout=timeout):
                pass
            return True
        for host in hosts:
            socket.getaddrinfo(host, 443)
        with socket.create_connection((hosts[0], 443), timeout=timeout):
            pass
        return True
    except OSError:
        return False


def wait_for_network(
    max_wait: float = 7200.0, poll: float = 30.0, hosts: tuple[str, ...] = HOSTS
) -> bool:
    """Blocks until ``online`` again, at most ``max_wait`` seconds; False at the deadline."""
    deadline = time.monotonic() + max_wait
    announced = False
    while not online(hosts):
        if time.monotonic() >= deadline:
            return False
        if not announced:
            print(
                f"netwait: {hosts} unreachable; waiting up to {max_wait / 60:.0f} min",
                file=sys.stderr,
            )
            announced = True
        time.sleep(poll)
    if announced:
        print("netwait: the network is back", file=sys.stderr)
    return True
