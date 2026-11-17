# SPDX-License-Identifier: Apache-2.0
# Vendored from mini-swe-agent's DefaultAgent (MIT); the loop, limits and trajectory format
# are upstream's. The templates default to the text-based protocol (one ```mswea_bash_command
# block per response), which needs no tool-calling support from the model API.

from __future__ import annotations

import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from .exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, TimeExceeded
from .templates import INSTANCE_TEMPLATE, SYSTEM_TEMPLATE


def recursive_merge(*dictionaries: dict | None) -> dict:
    result: dict[str, Any] = {}
    for d in dictionaries:
        if d is None:
            continue
        for key, value in d.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = recursive_merge(result[key], value)
            elif isinstance(value, dict):
                result[key] = recursive_merge(value)
            else:
                result[key] = value
    return result


class AgentConfig(BaseModel):
    system_template: str = SYSTEM_TEMPLATE
    instance_template: str = INSTANCE_TEMPLATE
    step_limit: int = 0
    cost_limit: float = 3.0
    wall_time_limit_seconds: int = 0
    max_consecutive_format_errors: int = 3
    output_path: Path | None = None


class DefaultAgent:
    def __init__(self, model: Any, env: Any, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars: dict[str, Any] = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.messages.extend(messages)
        return list(messages)

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until the agent is finished. Returns the exit message's ``extra``."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system", content=self._render_template(self.config.system_template)
            ),
            self.model.format_message(
                role="user", content=self._render_template(self.config.instance_template)
            ),
        )
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as e:
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if (
                    0
                    < self.config.max_consecutive_format_errors
                    <= self.n_consecutive_format_errors
                ):
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        return self.execute_actions(self.query())

    def query(self) -> dict:
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        # The wall clock is the session's own time: what its model spent waiting for a network
        # or an account that did not answer is left out.
        waited = float(getattr(self.model, "waited_seconds", 0.0) or 0.0)
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time - waited):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = [
            self.env.execute(action) for action in message.get("extra", {}).get("actions", [])
        ]
        return self.add_messages(
            *self.model.format_observation_messages(message, outputs, self.get_template_vars())
        )

    def serialize(self, *extra_dicts) -> dict:
        last_extra = (self.messages[-1] if self.messages else {}).get("extra", {})
        data = {
            "info": {
                "model_stats": {"instance_cost": self.cost, "api_calls": self.n_calls},
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": "2.4.6-vendored",
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=1, default=str))
        return data
