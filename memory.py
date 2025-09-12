"""
Simple persistent memory tool.
- Stores entries in data/memory.json by default.
- Each entry: {value, created_at, expires_at (or None), last_access}
- Supports TTL, max_entries with LRU eviction, atomic saves.
- API: set(key, value, ttl=None), get(key, default=None), delete(key), list(prefix=None), clear(), info()
"""
import json
import os
import time
import threading
from typing import Any, Dict, Optional

DEFAULT_PATH = os.path.join('data', 'memory.json')
_lock = threading.RLock()

# in-memory cache mirror
_store: Dict[str, Dict] = {}
_loaded = False


def _ensure_data_dir():
    d = os.path.dirname(DEFAULT_PATH)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _load(path: str = DEFAULT_PATH):
    global _loaded, _store
    with _lock:
        if _loaded:
            return
        _ensure_data_dir()
        if not os.path.exists(path):
            _store = {}
            _loaded = True
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # basic validation
                if isinstance(data, dict):
                    _store = data
                else:
                    _store = {}
        except Exception:
            _store = {}
        _loaded = True


def _save(path: str = DEFAULT_PATH):
    tmp = path + '.tmp'
    with _lock:
        _ensure_data_dir()
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(_store, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False


def _purge_expired():
    now = time.time()
    removed = []
    keys = list(_store.keys())
    for k in keys:
        meta = _store.get(k, {})
        exp = meta.get('expires_at')
        if exp is not None and exp <= now:
            removed.append(k)
            _store.pop(k, None)
    return removed


def _evict_lru(max_entries: int):
    # evict oldest last_access if over limit
    if max_entries is None or max_entries <= 0:
        return []
    removed = []
    if len(_store) <= max_entries:
        return removed
    # sort by last_access (oldest first)
    items = [(v.get('last_access', v.get('created_at', 0)), k) for k, v in _store.items()]
    items.sort()  # oldest first
    to_remove = len(_store) - max_entries
    for i in range(to_remove):
        _, k = items[i]
        _store.pop(k, None)
        removed.append(k)
    return removed


def set(key: str, value: Any, ttl: Optional[int] = None, path: str = DEFAULT_PATH, max_entries: Optional[int] = 1000, autosave: bool = True):
    """Set a key. ttl in seconds."""
    _load(path)
    with _lock:
        now = time.time()
        expires_at = (now + ttl) if (ttl is not None and ttl > 0) else None
        _store[key] = {'value': value, 'created_at': now, 'expires_at': expires_at, 'last_access': now}
        _purge_expired()
        _evict_lru(max_entries)
        if autosave:
            _save(path)
    return True


def get(key: str, default: Any = None, path: str = DEFAULT_PATH, autosave: bool = True):
    _load(path)
    with _lock:
        _purge_expired()
        meta = _store.get(key)
        if meta is None:
            return default
        meta['last_access'] = time.time()
        if autosave:
            _save(path)
        return meta.get('value', default)


def delete(key: str, path: str = DEFAULT_PATH, autosave: bool = True):
    _load(path)
    with _lock:
        existed = key in _store
        _store.pop(key, None)
        if autosave:
            _save(path)
    return existed


def list_keys(prefix: Optional[str] = None, path: str = DEFAULT_PATH):
    _load(path)
    with _lock:
        _purge_expired()
        if prefix is None:
            return list(_store.keys())
        return [k for k in _store.keys() if k.startswith(prefix)]


def clear(path: str = DEFAULT_PATH):
    _load(path)
    with _lock:
        _store.clear()
        _save(path)
    return True


def info(path: str = DEFAULT_PATH):
    _load(path)
    with _lock:
        _purge_expired()
        return {'entries': len(_store), 'keys_sample': list(_store.keys())[:20]}


# convenience alias
put = set
remove = delete


# --- Encryption-aware load/save override ---
import base64
import json as _json

try:
    from . import memory_aes as _memory_aes
    _aes_available = True
except Exception:
    try:
        import tools.memory_aes as _memory_aes
        _aes_available = True
    except Exception:
        _memory_aes = None
        _aes_available = False


def _get_encryption_config():
    try:
        with open('master_config.json', 'r', encoding='utf-8') as f:
            mc = _json.load(f)
            return mc.get('memory', {}).get('encryption', {})
    except Exception:
        return {}


def _key_bytes_from_config(cfg: dict):
    # key can be provided as hex or base64 or plain string
    key = cfg.get('key')
    if key is None:
        return None
    if isinstance(key, str):
        # try hex
        try:
            return bytes.fromhex(key)
        except Exception:
            pass
        # try base64
        try:
            return base64.b64decode(key)
        except Exception:
            pass
        # fallback to utf-8 bytes (not recommended)
        return key.encode('utf-8')
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    return None


def _load(path: str = DEFAULT_PATH):
    # override original _load: supports encrypted file when configured
    global _loaded, _store
    with _lock:
        if _loaded:
            return
        _ensure_data_dir()
        cfg = _get_encryption_config()
        enabled = bool(cfg.get('enabled', False))
        key_bytes = _key_bytes_from_config(cfg)
        if enabled and key_bytes is None:
            raise RuntimeError('Encryption enabled but no key found in master_config.memory.encryption.key')
        if not os.path.exists(path):
            _store = {}
            _loaded = True
            return
        try:
            if enabled:
                if not _aes_available or _memory_aes is None:
                    raise RuntimeError('Encryption requested but memory_aes helper or crypto library not available')
                # read encrypted content
                with open(path, 'rb') as f:
                    raw = f.read()
                try:
                    b64 = raw.decode('utf-8')
                except Exception:
                    b64 = raw
                plaintext = _memory_aes.decrypt_bytes(key_bytes, b64)
                data = _json.loads(plaintext.decode('utf-8'))
                if isinstance(data, dict):
                    _store = data
                else:
                    _store = {}
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = _json.load(f)
                    if isinstance(data, dict):
                        _store = data
                    else:
                        _store = {}
        except Exception:
            _store = {}
        _loaded = True


def _save(path: str = DEFAULT_PATH):
    # override original _save: supports encrypted output when configured
    tmp = path + '.tmp'
    with _lock:
        _ensure_data_dir()
        cfg = _get_encryption_config()
        enabled = bool(cfg.get('enabled', False))
        key_bytes = _key_bytes_from_config(cfg)
        try:
            if enabled:
                if key_bytes is None:
                    raise RuntimeError('Encryption enabled but no key found in master_config.memory.encryption.key')
                if not _aes_available or _memory_aes is None:
                    raise RuntimeError('Encryption requested but memory_aes helper or crypto library not available')
                plaintext = _json.dumps(_store, ensure_ascii=False, indent=2).encode('utf-8')
                b64 = _memory_aes.encrypt_bytes(key_bytes, plaintext)
                with open(tmp, 'wb') as f:
                    f.write(b64.encode('utf-8'))
            else:
                with open(tmp, 'w', encoding='utf-8') as f:
                    _json.dump(_store, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
            return True
        except Exception:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except Exception:
                pass
            return False

# expose helper functions to configure encryption at runtime

def enable_encryption(key: str, method: str = 'AES-GCM'):
    """Enable encryption by storing key into master_config.json under memory.encryption.key.
    Key may be hex, base64, or plain string. This function writes master_config.json directly.
    """
    try:
        with open('master_config.json', 'r', encoding='utf-8') as f:
            mc = _json.load(f)
    except Exception:
        mc = {}
    mc.setdefault('memory', {})['encryption'] = {'enabled': True, 'method': method, 'key': key}
    with open('master_config.json', 'w', encoding='utf-8') as f:
        _json.dump(mc, f, ensure_ascii=False, indent=2)
    return True


def disable_encryption():
    try:
        with open('master_config.json', 'r', encoding='utf-8') as f:
            mc = _json.load(f)
    except Exception:
        mc = {}
    if 'memory' in mc and 'encryption' in mc['memory']:
        mc['memory']['encryption']['enabled'] = False
    with open('master_config.json', 'w', encoding='utf-8') as f:
        _json.dump(mc, f, ensure_ascii=False, indent=2)
    return True

