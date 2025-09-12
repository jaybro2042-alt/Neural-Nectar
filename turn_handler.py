"""
turn_handler.py
Helpers to detect the 'Max turns' error and optionally auto-retry/continue.
This file is created by the orchestrator to provide a central place for handling external turn-limit errors.
"""
import json
import time

MASTER_CONFIG_PATH = 'master_config.json'


def load_master_config():
    try:
        with open(MASTER_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def is_max_turns_message(msg: str) -> bool:
    if not msg:
        return False
    return 'max turn' in msg.lower() or 'max turns' in msg.lower() or 'max_turns' in msg.lower() or 'Max turns' in msg


def should_auto_continue() -> bool:
    cfg = load_master_config()
    return cfg.get('console', {}).get('auto_continue_on_turn_limit', False)


def max_restarts_allowed() -> int:
    cfg = load_master_config()
    return int(cfg.get('console', {}).get('auto_continue_max_restarts', 0))


def handle_max_turns_error(exc_msg: str, current_restart_count: int) -> (bool, int):
    """Return (should_retry, new_restart_count).

    If auto-continue is enabled and restart_count < max, returns True and increments counter.
    Otherwise returns False.
    """
    if not is_max_turns_message(exc_msg):
        return False, current_restart_count

    if not should_auto_continue():
        return False, current_restart_count

    maxr = max_restarts_allowed()
    if current_restart_count >= maxr:
        return False, current_restart_count

    # simple backoff
    time.sleep(1 + current_restart_count * 2)
    return True, current_restart_count + 1

