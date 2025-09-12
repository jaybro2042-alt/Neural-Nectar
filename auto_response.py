import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG_PATH = ROOT / "master_config.json"

def load_config():
    if not CFG_PATH.exists():
        return {"auto_response": {"enabled": False, "mode": "best_guess"}}
    return json.loads(CFG_PATH.read_text(encoding='utf-8'))

def save_config(cfg):
    CFG_PATH.write_text(json.dumps(cfg, indent=2), encoding='utf-8')

def set_auto_response(enabled: bool, mode: str = None):
    cfg = load_config()
    cfg.setdefault("auto_response", {})
    cfg["auto_response"]["enabled"] = bool(enabled)
    if mode:
        cfg["auto_response"]["mode"] = mode
    save_config(cfg)
    return cfg["auto_response"]

def get_auto_response():
    cfg = load_config()
    return cfg.get("auto_response", {"enabled": False, "mode": "best_guess"})

from .plain_english import plain_english_from_summary

def maybe_auto_reply(summary: dict):
    cfg = get_auto_response()
    if not cfg.get("enabled"):
        return None
    mode = cfg.get("mode", "best_guess")
    text = plain_english_from_summary(summary)
    if mode == "concise":
        return text
    if mode == "actionable":
        return text + " Actionable: run parameter sweep (EMA 8-20 step2, EMA slow 20-60 step10; RSI thresholds 30-70), try 1h timeframe."
    # best_guess / detailed
    return text
