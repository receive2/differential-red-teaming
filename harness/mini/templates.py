# SPDX-License-Identifier: Apache-2.0
# Adapted from mini-swe-agent's mini_textbased.yaml (MIT): the one-block action protocol,
# the observation rendering and the format-error reply. The task text itself is the role's.

SYSTEM_TEMPLATE = """You are a capable engineer working in a shell inside a code repository.

Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
Your reasoning and analysis here. Explain why you want to perform the action.

```mswea_bash_command
your_command_here
```
</format_example>

Failure to follow these rules will cause your response to be rejected.
"""

INSTANCE_TEMPLATE = """{{task}}

## Working in the shell

1. Every response must contain exactly one action, enclosed in triple backticks as shown below.
2. The system executes it in a fresh subshell in the workspace root and shows you the result.
   Directory or environment variable changes do not persist between actions; prefix a command
   with `cd dir && ...` when you need another directory.
3. You have a budget of {{command_budget}} commands for this session; the result of each command
   says nothing about the budget until it is used up. Plan your work.
4. Finish by issuing exactly `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` on its own, followed on
   the next lines by your report (what you did, what is left, what you observed). Nothing else may
   be in that command. After it you cannot continue.
{%- if system == "Darwin" %}
5. This machine is macOS: `sed -i` needs an empty suffix, `sed -i '' 's/a/b/' file`.
{%- endif %}

<example_response>
THOUGHT: I need to see what the repository holds before changing anything.

```mswea_bash_command
ls -la
```
</example_response>

### Creating or overwriting a file

```mswea_bash_command
cat <<'END_OF_FILE' > path/to/file.py
print("hello")
END_OF_FILE
```

### Viewing part of a file

```mswea_bash_command
nl -ba path/to/file.py | sed -n '10,40p'
```
"""

OBSERVATION_TEMPLATE = """{% if output.exception_info -%}
<exception>{{output.exception_info}}</exception>
{% endif -%}
<returncode>{{output.returncode}}</returncode>
{% if output.output | length < 10000 -%}
<output>
{{ output.output -}}
</output>
{%- else -%}
<warning>
The output of your last command was too long. Use head, tail, sed -n or a narrower grep to see less.
</warning>
<output_head>
{{ output.output[:5000] }}
</output_head>
<elided_chars>
{{ output.output | length - 10000 }} characters elided
</elided_chars>
<output_tail>
{{ output.output[-5000:] }}
</output_tail>
{%- endif -%}
"""

FORMAT_ERROR_TEMPLATE = """{% if finish_reason is defined and finish_reason in ["length", "max_tokens"] -%}
Your previous response reached the output token limit before it held a complete action, so it was cut off. Respond more concisely and provide exactly one action in the required format.
{%- else -%}
Format error:

<error>
{{error}}
</error>

Please always provide EXACTLY ONE action in triple backticks, found {{actions|length}} actions, formatted as:

<response_example>
Your reasoning here.

```mswea_bash_command
<action>
```
</response_example>

If you have finished, submit with `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` followed by your report on the next lines.
{%- endif %}"""
