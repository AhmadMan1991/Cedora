"""Outlook accuracy tracker.

Records one daily snapshot of every symbol's directional bias (outlook.py) and
scores, going forward, whether that bias was right: did price actually move in
the predicted direction by the NEXT recorded day?

Fully deterministic. No history is reconstructed — accuracy accrues from the
first run, so the "last 7 days" number fills in over the first week of runs.

  record_outlook_snapshot(outlooks)  -> append/replace today's snapshot
  score_accuracy(window_days=7)      -> rolling scorecard dict for dashboard/Telegram
"""

import json
from pathlib import Path

import pandas as pd

HIST_FILE = Path(__file__).parent / "data" / "outlook_history.jsonl"
NEUTRAL_BAND = 0.15   # |norm| below this = no directional claim; excluded from scoring
KEEP_DAYS = 45


def _today() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")


def _load_history() -> list[dict]:
    rows: list[dict] = []
    if HIST_FILE.exists():
        for line in HIST_FILE.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def record_outlook_snapshot(outlooks: list) -> None:
    """Append today's snapshot (per-symbol price + bias), replacing any earlier
    snapshot from the same UTC day so multiple runs/day keep only the latest."""
    if not outlooks:
        return
    today = _today()
    snap = {"date": today, "ts": str(pd.Timestamp.now(tz="UTC")), "symbols": {}}
    for o in outlooks:
        try:
            price = o.get("price")
            snap["symbols"][o["symbol"]] = {
                "price": float(price) if price is not None else None,
                "norm": float(o["bias"]["norm"]),
                "label": o["bias"]["label"],
            }
        except Exception:
            continue
    if not snap["symbols"]:
        return
    rows = [r for r in _load_history() if r.get("date") != today]
    rows.append(snap)
    rows = rows[-KEEP_DAYS:]
    HIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    HIST_FILE.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def score_accuracy(window_days: int = 7) -> dict:
    """Score each recorded day's bias against the realized move to the NEXT
    recorded day. A directional bias is 'correct' when its sign matches the sign
    of the price change. Neutral reads (|norm| < NEUTRAL_BAND) are excluded.
    Aggregates the most recent `window_days`."""
    rows = sorted(_load_history(), key=lambda r: r.get("date", ""))
    base = {
        "window_days": window_days,
        "days": len(rows),
        "total": 0,
        "correct": 0,
        "neutral": 0,
        "accuracy_pct": None,
        "per_symbol": {},
        "as_of": rows[-1]["date"] if rows else _today(),
        "building": True,
    }
    if len(rows) < 2:
        return base

    latest = rows[-1]["date"]
    cutoff = (pd.Timestamp(latest) - pd.Timedelta(days=window_days)).strftime("%Y-%m-%d")
    total = correct = neutral = 0
    per: dict[str, list[int]] = {}
    for i in range(len(rows) - 1):
        cur, nxt = rows[i], rows[i + 1]
        if cur.get("date", "") < cutoff:
            continue
        for sym, cs in (cur.get("symbols") or {}).items():
            ns = (nxt.get("symbols") or {}).get(sym)
            if not ns:
                continue
            p0, p1, norm = cs.get("price"), ns.get("price"), cs.get("norm")
            if p0 is None or p1 is None or norm is None:
                continue
            if abs(norm) < NEUTRAL_BAND:
                neutral += 1
                continue
            pred = 1 if norm > 0 else -1
            actual = 1 if p1 > p0 else -1 if p1 < p0 else 0
            if actual == 0:
                continue
            slot = per.setdefault(sym, [0, 0])
            slot[1] += 1
            total += 1
            if pred == actual:
                slot[0] += 1
                correct += 1

    per_out = {
        s: {"correct": c, "total": t, "pct": round(c / t * 100, 1) if t else None}
        for s, (c, t) in sorted(per.items())
    }
    base.update(
        total=total,
        correct=correct,
        neutral=neutral,
        accuracy_pct=round(correct / total * 100, 1) if total else None,
        per_symbol=per_out,
        building=total == 0,
    )
    return base
