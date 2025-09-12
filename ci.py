from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def _client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai")
    return OpenAI()

def create_container(name: str = "bt-ci") -> Dict[str, Any]:
    client = _client()
    c = client.containers.create(name=name)
    return getattr(c, "model_dump", lambda **_: c)(exclude_none=True)

def responses_code_interpreter_auto(
    instructions: str,
    input_text: str,
    model: Optional[str] = None,
    file_ids: Optional[List[str]] = None,
    tool_choice: Optional[str] = None,
) -> Dict[str, Any]:
    client = _client()
    m = model or os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4.1")
    tools = [{
        "type": "code_interpreter",
        "container": {"type": "auto", **({"file_ids": file_ids} if file_ids else {})},
    }]
    kwargs = {"model": m, "tools": tools, "instructions": instructions, "input": input_text}
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    resp = client.responses.create(**kwargs)
    return {
        "output_text": getattr(resp, "output_text", None),
        "raw": getattr(resp, "model_dump", lambda **_: resp)(exclude_none=True),
    }

def responses_code_interpreter_explicit(
    container_id: str,
    input_text: str,
    model: Optional[str] = None,
    tool_choice: str = "required",
    instructions: Optional[str] = None,
) -> Dict[str, Any]:
    client = _client()
    m = model or os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4.1")
    tools = [{"type": "code_interpreter", "container": container_id}]
    resp = client.responses.create(
        model=m,
        tools=tools,
        input=input_text,
        tool_choice=tool_choice,
        **({"instructions": instructions} if instructions else {}),
    )
    return {
        "output_text": getattr(resp, "output_text", None),
        "raw": getattr(resp, "model_dump", lambda **_: resp)(exclude_none=True),
    }
