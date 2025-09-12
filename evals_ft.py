from __future__ import annotations
from typing import Any, Dict

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def _client():
    if OpenAI is None:
        raise RuntimeError("openai package not installed. pip install openai")
    return OpenAI()

def upload_evals_file(path: str) -> Dict[str, Any]:
    client = _client()
    with open(path, "rb") as f:
        file = client.files.create(file=f, purpose="evals")
    return getattr(file, "model_dump", lambda **_: file)(exclude_none=True)

def create_eval(name: str, data_source_config: Dict[str, Any], testing_criteria: list[Dict[str, Any]]) -> Dict[str, Any]:
    client = _client()
    if not hasattr(client, "evals"):
        return {"error": "This openai SDK does not expose evals yet. Upgrade package."}
    ev = client.evals.create(name=name, data_source_config=data_source_config, testing_criteria=testing_criteria)
    return getattr(ev, "model_dump", lambda **_: ev)(exclude_none=True)

def start_eval_run(eval_id: str, name: str, model: str, file_id: str, template_messages: list[dict]) -> Dict[str, Any]:
    client = _client()
    if not hasattr(client, "evals") or not hasattr(client.evals, "runs"):
        return {"error": "This openai SDK does not expose evals.runs yet. Upgrade package."}
    run = client.evals.runs.create(
        eval_id,
        name=name,
        data_source={
            "type": "responses",
            "model": model,
            "input_messages": {"type": "template", "template": template_messages},
            "source": {"type": "file_id", "id": file_id},
        },
    )
    return getattr(run, "model_dump", lambda **_: run)(exclude_none=True)

def get_eval_run(eval_id: str, run_id: str) -> Dict[str, Any]:
    client = _client()
    if not hasattr(client, "evals") or not hasattr(client.evals, "runs"):
        return {"error": "This openai SDK does not expose evals.runs yet. Upgrade package."}
    run = client.evals.runs.retrieve(eval_id, run_id)
    return getattr(run, "model_dump", lambda **_: run)(exclude_none=True)

def start_sft_job(training_file_id: str, model: str = "gpt-4.1-mini-2025-04-14", suffix: str | None = None) -> Dict[str, Any]:
    client = _client()
    if not hasattr(client, "fine_tuning"):
        return {"error": "This openai SDK does not expose fine_tuning yet. Upgrade package."}
    job = client.fine_tuning.jobs.create(training_file=training_file_id, model=model, **({"suffix": suffix} if suffix else {}))
    return getattr(job, "model_dump", lambda **_: job)(exclude_none=True)
