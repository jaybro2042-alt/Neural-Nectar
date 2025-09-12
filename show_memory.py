"""Utility to print memory contents (for quick inspection)."""

from tools import memory
import json

if __name__ == '__main__':
    info = memory.info()
    print(json.dumps(info, indent=2))
    keys = memory.list_keys()
    for k in keys:
        v = memory.get(k)
        print(k, '->', v)

