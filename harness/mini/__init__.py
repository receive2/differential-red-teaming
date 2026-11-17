# SPDX-License-Identifier: Apache-2.0

"""A vendored, litellm-free core of mini-swe-agent (SWE-agent/mini-swe-agent, MIT): the
step loop (``agent.py``), the local bash environment (``environment.py``) and the text-based
action protocol, with model adapters written on the OpenAI and Anthropic HTTP APIs
(``models.py``). The package itself is not installed: it depends on litellm, and the adapters
here need httpx alone. Behavior follows upstream v2.4.6; the trajectory format is kept."""
