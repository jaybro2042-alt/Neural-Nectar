"""
Agent helpers: ensure master_config is available at runtime and provide a fine_tune_strategy helper
that prepares training data, uploads it, and creates a fine-tuning job using OpenAI's API.

This file is appended by the orchestrator on user request. It does not run automatically; import and call
functions as needed from scripts or the REPL.
"""
import os
import json
import time
from typing import Optional


def load_master_config():
    """Try common locations for master_config and return the parsed JSON dict or None."""
    candidates = [
        'master_config.json',
        os.path.join('strategies', 'master_config.json'),
        os.path.join('strategies', 'master_config.backup.json'),
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception:
            continue
    return None


def ensure_master_config_available(write_back: bool = True):
    """If master_config.json is missing at repo root but present in strategies/, copy it to root so tools
    that expect it at root can load it. Returns path to the config used.
    """
    root = 'master_config.json'
    strat = os.path.join('strategies', 'master_config.json')
    if os.path.exists(root):
        return root
    if os.path.exists(strat):
        if write_back:
            try:
                with open(strat, 'r', encoding='utf-8') as sf:
                    data = sf.read()
                with open(root, 'w', encoding='utf-8') as rf:
                    rf.write(data)
                return root
            except Exception:
                return strat
        return strat
    return None


# --- Fine-tune helper ---
# Uses openai Python library when available. Falls back to HTTP POST if not.

def fine_tune_strategy(training_jsonl_path: str, model: str = 'gpt-4o-mini', n_epochs: int = 4, validation_jsonl_path: Optional[str] = None, force_upload: bool = False):
    """Upload training file, create a fine-tune job, and return the job dict.

    Requirements:
    - OPENAI_API_KEY must be present in environment or master_config.master.openai_api_key
    - training_jsonl_path must point to a newline-delimited JSONL file where each line is an object
      appropriate for the target base model (see OpenAI docs on fine-tuning formats).

    This helper will:
      1) Try to import openai SDK and use openai.File.create + openai.FineTuningJob.create (or openai.FineTunes.create)
      2) If SDK is not available, attempt direct HTTP calls to the API endpoints.

    Note: This function does not wait for fine-tuning completion by default, it returns the created job object.
    """
    # load API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        mc = load_master_config()
        if mc:
            api_key = mc.get('openai_api_key') or mc.get('api_key')
    if not api_key:
        raise RuntimeError('OpenAI API key not found in environment (OPENAI_API_KEY) or master_config')

    # check file exists
    if not os.path.exists(training_jsonl_path):
        raise FileNotFoundError(training_jsonl_path)

    # attempt to use openai SDK
    try:
        import openai
        openai.api_key = api_key
        # Upload training file
        with open(training_jsonl_path, 'rb') as f:
            upload_resp = openai.File.create(file=f, purpose='fine-tune')
        training_file_id = upload_resp.id if hasattr(upload_resp, 'id') else upload_resp.get('id')
        kwargs = {'training_file': training_file_id, 'model': model}
        if validation_jsonl_path and os.path.exists(validation_jsonl_path):
            with open(validation_jsonl_path, 'rb') as vf:
                val_resp = openai.File.create(file=vf, purpose='fine-tune')
            validation_file_id = val_resp.id if hasattr(val_resp, 'id') else val_resp.get('id')
            kwargs['validation_file'] = validation_file_id
        # hyperparameters
        kwargs['n_epochs'] = n_epochs
        # Create fine-tune job
        # The exact SDK call name can differ by openai library version. Try multiple known variants.
        try:
            job = openai.FineTune.create(**kwargs)
        except Exception:
            try:
                job = openai.FineTuningJob.create(**kwargs)
            except Exception:
                job = openai.FineTunes.create(**kwargs)
        return {'status': 'created', 'job': job, 'training_file': training_file_id}
    except Exception as e_sdk:
        # Fall back to HTTP POST
        try:
            import requests
        except Exception:
            raise RuntimeError('Neither openai SDK nor requests available; cannot create fine-tune job: ' + str(e_sdk))
        headers = {'Authorization': f'Bearer {api_key}'}
        # upload file
        files = {'file': open(training_jsonl_path, 'rb')}
        resp = requests.post('https://api.openai.com/v1/files', headers=headers, files=files)
        if not resp.ok:
            raise RuntimeError('File upload failed: ' + resp.text)
        training_file_id = resp.json().get('id')
        payload = {'training_file': training_file_id, 'model': model, 'n_epochs': n_epochs}
        if validation_jsonl_path and os.path.exists(validation_jsonl_path):
            files = {'file': open(validation_jsonl_path, 'rb')}
            r2 = requests.post('https://api.openai.com/v1/files', headers=headers, files=files)
            if not r2.ok:
                raise RuntimeError('Validation file upload failed: ' + r2.text)
            payload['validation_file'] = r2.json().get('id')
        rjob = requests.post('https://api.openai.com/v1/fine_tuning/jobs', headers=headers, json=payload)
        if not rjob.ok:
            # older endpoint name fallback
            rjob = requests.post('https://api.openai.com/v1/fine-tunes', headers=headers, json=payload)
        if not rjob.ok:
            raise RuntimeError('Fine-tune create failed: ' + rjob.text)
        return {'status': 'created', 'job': rjob.json(), 'training_file': training_file_id}


# Utility to prepare simple training JSONL from backtest trades
def build_training_from_backtest(trades: list, out_jsonl_path: str, prompt_template: Optional[str] = None):
    """
    Convert a list of trades (dicts) into a simple supervised JSONL format for fine-tuning a behavior model.
    Each line will be {"prompt": <state>, "completion": <action>}.
    This is a very naive transformer — you should review and curate training data for quality.
    """
    if prompt_template is None:
        prompt_template = 'Given the recent market features: {features}\nDecide action:'
    with open(out_jsonl_path, 'w', encoding='utf-8') as f:
        for t in trades:
            # build a simple features string
            features = []
            for k in ['entry_price', 'exit_price', 'pnl', 'units']:
                if k in t:
                    features.append(f"{k}={t.get(k)}")
            prompt = prompt_template.format(features='; '.join(features))
            action = 'BUY' if t.get('pnl', 0) > 0 else 'HOLD' if t.get('pnl',0)==0 else 'SELL'
            line = json.dumps({'prompt': prompt, 'completion': ' ' + action})
            f.write(line + '\n')
    return out_jsonl_path

