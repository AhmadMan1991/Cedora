"""
Signal engine — OUTLOOK-DRIVEN.

The live trade signals are generated directly from the deterministic outlook
(outlook.py), replacing the old intraday trend-pullback confluence model
(gold_engine.evaluate_setup, kept only for backtest/reference). One unified
method across every symbol:

  • DIRECTION  — the outlook's weighted 4H/Daily/Weekly bias vote. A signal
    fires only when |bias.norm| >= SIGNAL_MIN_BIAS (the outlook's own
    BULLISH/BEARISH threshold); MILDLY/NEUTRAL stands aside. This weighted
    multi-timeframe vote is the regime filter (it replaces the old ADX gate).
  • ENTRY      — the scenario TRIGGER level in the bias direction (the nearest
    real level to break: bull trigger above / bear trigger below). If no such
    level exists, entry falls back to current price.
  • TARGETS    — the scenario's next levels (T1/T2/T3), ADR-stepped to fill any
    that the level stack doesn't supply.
  • STOP       — the nearest structural reference level beyond the entry on the
    invalidation side, buffered by a fraction of ADR, with the risk distance
    CLAMPED to [SIGNAL_RISK_MIN_PCT, SIGNAL_RISK_MAX_PCT] of price so it is a
    sane absolute size for any instrument (no micro- or runaway stops).
  • CONFIDENCE — scaled from |bias.norm| (stronger vote → higher conviction).

No LLM anywhere — every number comes from price via the outlook.
"""

import json
from pathlib import Path

import pandas as pd

from config import (
    MARKETS, SIGNAL_ENABLED, SIGNAL_SYMBOLS, SIGNAL_MIN_BIAS, SIGNAL_STOP_ADR,
    SIGNAL_RISK_MIN_PCT, SIGNAL_RISK_MAX_PCT, SIGNAL_TGT_STEP_ADR,
)
from outlook import run_outlook

DATA_ROOT = Path(__file__).parent / "data"
STATE_FILE = DATA_ROOT / "signal_state.json"

# Reusable signal taxonomy (same louder-SNIPER treatment as the rest of the bot)
SIGNAL_TAXONOMY = {
    "SNIPER":           ("🎯", "SNIPER SETUP",     "Highest-conviction multi-timeframe alignment"),
    "HIGH_PROBABILITY": ("💠", "HIGH PROBABILITY", "Strong weighted 4H/Daily/Weekly agreement"),
    "LONG":             ("📈", "LONG",             "Outlook bias up — break-and-extend"),
    "SHORT":            ("📉", "SHORT",            "Outlook bias down — break-and-extend"),
}


# ── per-symbol dedup state (last-fired direction + entry level; fires on change) ──
def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    DATA_ROOT.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _round(v, dec):
    try:
        return round(float(v), dec)
    except Exception:
        return v


def outlook_signal(o: dict) -> dict | None:
    """Turn one outlook dict (outlook.compute_outlook) into a trade signal, or
    None if conviction is below SIGNAL_MIN_BIAS or the data is unusable."""
    norm = o.get("bias", {}).get("norm")
    price = o.get("price")
    if norm is None or price is None or price <= 0:
        return None
    if abs(norm) < SIGNAL_MIN_BIAS:
        return None  # MILDLY / NEUTRAL — not enough conviction to trade

    bull = norm > 0
    direction = "LONG" if bull else "SHORT"
    dec = o.get("decimals", 2)

    lv = o.get("levels", {}) or {}
    all_levels = [v for k, v in lv.items()
                  if k != "pos_in_week_pct" and isinstance(v, (int, float))]

    vol = o.get("volatility", {}) or {}
    adr = vol.get("adr") or (price * 0.01)   # fallback: 1% of price if ADR missing
    if adr <= 0:
        adr = price * 0.01

    sc = (o.get("scenarios", {}) or {}).get("bull" if bull else "bear", {}) or {}

    # ── ENTRY: scenario breakout trigger, else current price ──────────────
    entry = sc.get("trigger")
    if entry is None:
        entry = price

    # ── STOP: nearest structural level beyond entry on the invalidation side,
    #          buffered by a slice of ADR, risk clamped to a sane band ──────
    if bull:
        belows = [x for x in all_levels if x < entry]
        struct = max(belows) if belows else None
        raw_stop = (struct - adr * 0.25) if struct is not None else (entry - adr * SIGNAL_STOP_ADR)
        risk = entry - raw_stop
    else:
        aboves = [x for x in all_levels if x > entry]
        struct = min(aboves) if aboves else None
        raw_stop = (struct + adr * 0.25) if struct is not None else (entry + adr * SIGNAL_STOP_ADR)
        risk = raw_stop - entry

    risk = max(risk, entry * SIGNAL_RISK_MIN_PCT)
    risk = min(risk, entry * SIGNAL_RISK_MAX_PCT)
    if risk <= 0:
        return None
    stop = entry - risk if bull else entry + risk

    # ── TARGETS: structural levels, but RR-GUARDED so TP1/2/3 clear a risk
    #    multiple (>=1R / 2R / 3R). Each slot snaps OUT to the nearest real
    #    reference level at or beyond its R-floor; if none sits beyond it, the
    #    floor (or an ADR step past the previous target) is used. This stops a
    #    level that is only a few points past the breakout from becoming a
    #    nonsensically tight TP1 with sub-1R reward. ──────────────────────────
    pool = sorted(set(x for x in all_levels if (x > entry if bull else x < entry)),
                  reverse=not bull)   # nearest-first in the trade direction
    tps = []
    for i in range(3):
        floor = entry + risk * (i + 1) if bull else entry - risk * (i + 1)
        if bull:
            cand = next((L for L in sorted(pool) if L >= floor and (not tps or L > tps[-1])), None)
        else:
            cand = next((L for L in sorted(pool, reverse=True) if L <= floor and (not tps or L < tps[-1])), None)
        if cand is None:
            base = tps[-1] if tps else entry
            step = adr * SIGNAL_TGT_STEP_ADR
            cand = max(floor, base + step) if bull else min(floor, base - step)
        tps.append(cand)
    tp1, tp2, tp3 = tps[0], tps[1], tps[2]

    rr1 = round(abs(tp1 - entry) / risk, 1) if risk > 0 else 0.0

    # ── CONFIDENCE from the strength of the weighted vote ─────────────────
    conf = round(min(0.97, 0.55 + (abs(norm) - SIGNAL_MIN_BIAS) / (1.0 - SIGNAL_MIN_BIAS) * 0.42), 2)
    label = "SNIPER" if conf >= 0.80 else "HIGH_PROBABILITY" if conf >= 0.68 else direction

    st = o.get("structure", {}) or {}
    bias_label = o.get("bias", {}).get("label", "")
    factors = [
        f"Outlook bias {bias_label} (score {norm:+.2f}) — "
        f"Weekly {st.get('weekly','?')} / Daily {st.get('daily','?')} / 4H {st.get('h4','?')}",
        f"Entry on the {'bull' if bull else 'bear'} scenario trigger "
        f"({_round(entry, dec)}); invalidation at {_round(stop, dec)}",
    ]
    note = vol.get("note")
    if note and note != "—":
        factors.append(note)

    return {
        "asset": o["symbol"],
        "symbol": o["symbol"],
        "name": o.get("name", o["symbol"]),
        "emoji": o.get("emoji", ""),
        "decimals": dec,
        "direction": direction,
        "signal_label": label,
        "confidence": conf,
        "bias_norm": norm,
        "entry": _round(entry, dec),
        "stop_loss": _round(stop, dec),
        "target_1": _round(tp1, dec),
        "target_2": _round(tp2, dec),
        "target_3": _round(tp3, dec),
        "risk_reward": rr1,
        "risk_usd": _round(risk, dec),
        "timeframe": "4H/D/W",
        "layer": "outlook_signal",
        "factors": factors,
        "reasoning": f"{direction} — " + "; ".join(factors),
        "timestamp": str(pd.Timestamp.now(tz="UTC")),
    }


def run_signals(outlooks: list | None = None) -> list[dict]:
    """Generate signals from the outlook for every SIGNAL_SYMBOLS entry.

    Dedup: a symbol fires when a NEW setup appears — i.e. the direction flips OR
    the entry trigger moves to a different structural level vs the last signal
    sent. A standing, unchanged setup is not re-sent on every run; but as soon
    as price breaks to a new level (new entry) or the bias turns, it alerts.
    Pass `outlooks` to reuse an already-computed outlook list (the pipeline
    computes it once and feeds it to both the report and this engine)."""
    if not SIGNAL_ENABLED:
        return []
    if outlooks is None:
        outlooks = run_outlook(SIGNAL_SYMBOLS)

    state = _load_state()
    today = str(pd.Timestamp.now(tz="UTC").date())
    fired = []
    for o in outlooks:
        sym = o.get("symbol")
        if sym not in SIGNAL_SYMBOLS:
            continue
        try:
            sig = outlook_signal(o)
        except Exception as e:
            print(f"  ⚠ signal {sym} failed: {e}")
            continue
        emoji = MARKETS.get(sym, {}).get("emoji", "")
        if not sig:
            print(f"  {emoji} {sym}: no signal (bias below {SIGNAL_MIN_BIAS:+.2f})")
            continue
        prev = state.get(sym) or {}
        dec = o.get("decimals", MARKETS.get(sym, {}).get("decimals", 2))
        entry_lvl = None if sig.get("entry") is None else round(float(sig["entry"]), dec)
        # A "new setup" = the direction changed OR the entry trigger moved to a
        # different structural level. An unchanged, still-standing setup is not
        # re-sent, so the channel only pings on a genuinely new opportunity.
        if prev.get("direction") == sig["direction"] and prev.get("entry") == entry_lvl:
            print(f"  {emoji} {sym}: {sig['direction']} @ {entry_lvl} unchanged — skipped")
            continue
        state[sym] = {"date": today, "direction": sig["direction"], "entry": entry_lvl,
                      "ts": str(pd.Timestamp.now(tz="UTC"))}
        print(f"  {emoji} {sym}: {sig['signal_label']} {sig['direction']} @ {entry_lvl} "
              f"conf={sig['confidence']:.0%}")
        fired.append(sig)
    _save_state(state)
    return fired
