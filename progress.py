"""
Simple progress reporting helper. Writes progress entries to data/progress.json and optionally prints to console.
Each task entry: {task_id, task_name, status, percent, details, timestamp}
"""
import json
import os
import time
from typing import Optional

PATH = os.path.join('data', 'progress.json')


def _ensure_dir():
    d = os.path.dirname(PATH)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load():
    _ensure_dir()
    if not os.path.exists(PATH):
        return []
    try:
        with open(PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def _save(items):
    _ensure_dir()
    try:
        with open(PATH + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2)
        os.replace(PATH + '.tmp', PATH)
        return True
    except Exception:
        return False


def report(task_id: str, task_name: str, status: str, percent: float = 0.0, details: Optional[str] = None):
    items = _load()
    entry = {'task_id': task_id, 'task_name': task_name, 'status': status, 'percent': float(percent), 'details': details or '', 'timestamp': int(time.time())}
    # replace existing with same task_id
    for i, it in enumerate(items):
        if it.get('task_id') == task_id:
            items[i] = entry
            _save(items)
            return entry
    items.append(entry)
    _save(items)
    return entry


def list_progress():
    return _load()

