from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, List, Any

try:
    from orchestrator import core as ORC
except Exception as e:
    raise RuntimeError("orchestrator.core not found; ensure your project has orchestrator/core.py") from e

def _sample_int(v: int, lo: int, hi: int) -> int:
    lo = int(max(2, lo)); hi = int(max(lo+1, hi))
    return int(random.randint(lo, hi))

def _sample_float(v: float, lo: float, hi: float) -> float:
    lo = float(lo); hi = float(max(lo + 1e-9, hi))
    return float(random.uniform(lo, hi))

def _infer_space(params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    space: Dict[str, Dict[str, float]] = {}
    for k, v in params.items():
        if isinstance(v, int):
            delta = max(2, int(round(abs(v) * 0.5)))
            space[k] = {"type": "int", "min": max(2, v - delta), "max": max(3, v + delta)}
        elif isinstance(v, float):
            delta = max(0.5, abs(v) * 0.25)
            space[k] = {"type": "float", "min": max(1e-6, v - delta), "max": v + delta}
    return space

def intensive_optimize(
    spec_path: str,
    primary: str = "sharpe",
    budget: int = 60,
    constraints: Dict[str, float] | None = None,
    search_space: Dict[str, Dict[str, float]] | None = None,
    seed: int | None = 42,
) -> Dict[str, Any]:
    """
    Random + local search over Backtrader parameters defined in spec_path.
    Returns: {'best': {...}, 'trials': N, 'candidates': [...]}
    """
    if seed is not None:
        random.seed(seed)

    base = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    params = dict(base.get("params", {}))
    if not params:
        return {"reason": "no params in spec", "spec_path": spec_path}

    space = search_space or _infer_space(params)

    tried: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None

    coarse = max(1, int(budget * 0.7))
    fine = max(0, budget - coarse)

    def _trial(updated_params: Dict[str, Any]) -> Dict[str, Any]:
        trial = json.loads(json.dumps(base))
        trial.setdefault("params", {}).update(updated_params)
        tmp = Path(getattr(ORC, "STRATS", Path("strategies"))) / f"__int_{base.get('name','spec')}_{abs(hash(str(updated_params)))%10**8}.json"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(trial, indent=2), encoding="utf-8")
        res = ORC.run_backtest_on_spec(str(tmp))
        try:
            tmp.unlink()
        except Exception:
            pass
        return res

    for _ in range(coarse):
        cand = {}
        for k, cfg in space.items():
            if cfg.get("type") == "int":
                v0 = int(params.get(k, 10))
                cand[k] = _sample_int(v0, cfg.get("min", 2), cfg.get("max", v0 * 2))
            else:
                v0 = float(params.get(k, 1.0))
                cand[k] = _sample_float(v0, cfg.get("min", 0.1), cfg.get("max", max(0.2, v0 * 2)))
        out = _trial(cand)
        tried.append(out)
        if (not best) or (out.get("metrics", {}).get(primary, -1e9) > best.get("metrics", {}).get(primary, -1e9)):
            best = out

    if best and fine > 0:
        center = dict(params)
        for _ in range(fine):
            cand = {}
            for k, cfg in space.items():
                if cfg.get("type") == "int":
                    v0 = int(center.get(k, 10))
                    lo = max(2, v0 - max(1, (cfg.get("max", v0) - cfg.get("min", 2)) // 6))
                    hi = v0 + max(2, (cfg.get("max", v0) - cfg.get("min", 2)) // 6)
                    cand[k] = _sample_int(v0, lo, hi)
                else:
                    v0 = float(center.get(k, 1.0))
                    width = max(0.1, (cfg.get("max", v0) - cfg.get("min", 0.1)) / 6)
                    cand[k] = _sample_float(v0, max(1e-6, v0 - width), v0 + width)
            out = _trial(cand)
            tried.append(out)
            if out.get("metrics", {}).get(primary, -1e9) > best.get("metrics", {}).get(primary, -1e9):
                best = out

    if not best:
        return {"reason": "no successful trials", "trials": len(tried)}

    # Use orchestrator's selection util if available
    if hasattr(ORC, "select_best"):
        pick = ORC.select_best(tried, primary=primary, **(constraints or {}))
        if isinstance(pick, dict) and "winner" in pick:
            best = pick["winner"]

    return {"best": best, "trials": len(tried), "candidates": tried}
