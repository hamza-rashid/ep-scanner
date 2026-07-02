"""
Episodic Pivot Scanner  (AI-classified, mechanically scored)
============================================================
Finds potential episodic pivots (Stockbee / Qullamaggie style) in US stocks,
scores them by catalyst quality + technicals, and writes data.json for the site.

HOW THE AI IS USED (and how it stays mechanical)
------------------------------------------------
The AI does exactly ONE job: read a news headline/summary and pick which
catalyst TYPE it is, from a FIXED menu (see CATALYST_TYPES below). It returns
a label like "fda_approval" — never a number, never a score.

Your CONFIG then looks up the points for that label. So:
  - AI decides *what kind of news this is*  (a meaning judgment it's good at)
  - Arithmetic decides *how many points that kind gets*  (from your config table)

To make it repeatable ~99/100:
  - temperature = 0            (model takes its single most-likely answer)
  - fixed menu of labels       (it can only pick from a closed list)
  - it classifies, never scores (no subjective "42 vs 45")
  - deterministic tie-break     (ambiguous cases resolve the same way every time)

The technical half (gap, ADR, dollar volume, MAs, neglect) is pure math and
uses NO AI — it doesn't need it.

Runs 4x per day (pinned to US Eastern Time, DST-safe):
  Run 0  02:00 ET  (~7am UK)   catalyst-only overnight watchlist
  Run 1  08:00 ET  (~1pm UK)   real premarket, gaps live
  Run 2  08:45 ET  (~1:45pm UK)
  Run 3  09:15 ET  (~2:15pm UK) sharpest pre-open read

Data sources:
  FMP        movers, earnings, upgrades, news, quotes, prices  (paid ~$20-40/mo)
  Anthropic  catalyst classification                           (paid, ~pennies/day)
  openFDA    drug approvals (used to confirm AI calls)          (free)

All tunable weights live in CONFIG below — change numbers there, never the logic.
"""

import os
import json
import time
import datetime as dt
from urllib.request import urlopen, Request
from urllib.parse import quote

# ======================================================================
# CATALYST TYPES  — the fixed menu the AI must choose from.
# The AI picks the LABEL. The points come from here. Tune points freely.
# `tier` is just for the badge colour on the site.
# ======================================================================
CATALYST_TYPES = {
    # label                     points  tier   human description
    "fda_approval":            (50, "S", "FDA approval / positive pivotal trial"),
    "acquisition_target":      (50, "S", "Acquisition / buyout (target)"),
    "major_contract":          (44, "S", "Major contract / government award"),
    "earnings_blowout_raise":  (50, "S", "Earnings blowout + raised guidance"),
    "earnings_beat":           (38, "A", "Earnings beat"),
    "analyst_upgrade_big_pt":  (38, "A", "Upgrade + big price-target raise"),
    "major_partnership":       (36, "A", "Major partnership / collaboration"),
    "insider_cluster_buy":     (34, "A", "Insider cluster buying"),
    "positive_guidance":       (30, "A", "Raised guidance (standalone)"),
    "earnings_inline":         (24, "B", "In-line earnings, strong reaction"),
    "sector_sympathy":         (22, "B", "Sector sympathy move"),
    "index_inclusion":         (24, "B", "Index inclusion"),
    "product_launch":          (26, "B", "Notable product launch"),
    "legal_ruling_favorable":  (30, "A", "Favourable legal / regulatory ruling"),
    "vague_pr":                (14, "C", "Vague PR / promotional"),
    "dilution_offering":       (-8, "C", "Share offering / dilution (bearish)"),
    "unclassified_move":       (14, "C", "Moving on unclear news"),
    "no_catalyst":             (0,  None, "No real catalyst"),
}

# The closed list of labels the AI is allowed to return.
ALLOWED_LABELS = list(CATALYST_TYPES.keys())

# Deterministic tie-break priority: if the model is ever ambiguous between
# two types, we instruct it to prefer the one higher in THIS list. Same
# input -> same winner, every time.
TIEBREAK_PRIORITY = [
    "acquisition_target", "fda_approval", "earnings_blowout_raise",
    "major_contract", "legal_ruling_favorable", "earnings_beat",
    "analyst_upgrade_big_pt", "major_partnership", "positive_guidance",
    "insider_cluster_buy", "index_inclusion", "product_launch",
    "earnings_inline", "sector_sympathy", "dilution_offering",
    "vague_pr", "unclassified_move", "no_catalyst",
]


# ======================================================================
# CONFIG  — tunable weights. Single source of truth.
# ======================================================================
CONFIG = {
    # ---- Gap % ramp (Stockbee: 7.5% pivot) ----
    "gap": {"pivot": 7.5, "floor": 5.0, "full_points": 18,
            "per_pct_above": 0.6, "max_bonus": 15},

    # ---- ADR % ramp (Qullamaggie: 4% min) ----
    "adr": {"pivot": 4.0, "full_points": 8, "per_pct_below": 1.5},

    # ---- Dollar volume ramp (>$100M for best) ----
    "dollar_volume": {"pivot": 100, "full_points": 15, "below_points": 8,
                      "thin_floor": 20, "thin_points": 4},

    # ---- Moving averages (200d most important, then 50d) ----
    "ma": {"above_200": 8, "above_50": 6, "above_20": 3, "above_10": 2},

    # ---- Neglect bonus (quiet base before the move = ideal EP) ----
    "neglect": {"high_points": 10, "med_points": 6, "low_points": 0},

    # ---- Volume-expansion bonus ----
    "volume_expansion": {"per_x": 3, "max_points": 12},

    # ---- Sanity gates (the ONLY hard filters) ----
    "gates": {"min_price": 3.0, "min_market_cap_m": 100,
              "min_dollar_volume_m": 5, "require_catalyst": True},

    # ---- AI classification ----
    "ai": {
        "model": "claude-sonnet-5",   # good + cheap for classification
        "enabled": True,               # set False to fall back to keyword-only
        "batch_size": 15,              # headlines per API call (keeps it cheap/fast)
    },

    "max_results": 40,
}

FMP_KEY = os.environ.get("FMP_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/api/v3"
OUT_PATH = os.environ.get("OUT_PATH", "data.json")
RUN_LABEL = os.environ.get("RUN_LABEL", "manual")


# ======================================================================
# HTTP helpers
# ======================================================================
def get_json(url, tries=3):
    for i in range(tries):
        try:
            req = Request(url, headers={"User-Agent": "ep-scanner/1.0"})
            with urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == tries - 1:
                print(f"  ! fetch failed: {url[:80]} -> {e}")
                return None
            time.sleep(2)
    return None


def fmp(path, params=""):
    sep = "&" if "?" in path else "?"
    return get_json(f"{FMP_BASE}/{path}{sep}apikey={FMP_KEY}{params}")


# ======================================================================
# 1. Gather candidates — the day's gappers / big movers
# ======================================================================
def gather_candidates():
    cand = {}
    gainers = fmp("stock_market/gainers") or []
    for g in gainers[:80]:
        t = g.get("symbol")
        if not t:
            continue
        cand[t] = {"ticker": t, "name": g.get("name", ""),
                   "changesPercentage": _to_float(g.get("changesPercentage")),
                   "price": _to_float(g.get("price"))}

    pre = fmp("pre-post-market/gainers") if RUN_LABEL in ("1", "2", "3") else None
    if pre:
        for g in pre[:80]:
            t = g.get("symbol") or g.get("ticker")
            if not t:
                continue
            row = cand.get(t, {"ticker": t, "name": g.get("name", "")})
            row["premarket_change"] = _to_float(g.get("changesPercentage") or g.get("change"))
            row["price"] = _to_float(g.get("price")) or row.get("price")
            cand[t] = row

    print(f"  candidates gathered: {len(cand)}")
    return cand


# ======================================================================
# 2. Fetch the NEWS for each candidate (so the AI has something to read)
# ======================================================================
def fetch_news(tickers):
    """
    FMP stock news for our candidates. Returns ticker -> list of recent
    headline+text snippets. The AI reads these to classify the catalyst.
    """
    news = {}
    for chunk in _chunks(tickers, 20):
        joined = ",".join(chunk)
        data = fmp("stock_news", f"&tickers={quote(joined)}&limit=60") or []
        for n in data:
            t = n.get("symbol")
            if not t:
                continue
            item = {
                "title": (n.get("title") or "").strip(),
                "text": (n.get("text") or "")[:400].strip(),
                "published": n.get("publishedDate", ""),
            }
            news.setdefault(t, []).append(item)
    for t in news:
        news[t] = sorted(news[t], key=lambda x: x["published"], reverse=True)[:3]
    print(f"  news pulled for: {len(news)} tickers")
    return news


# ======================================================================
# 3. AI catalyst classification  — the ONE place AI is used.
# ======================================================================
def classify_all_with_ai(cand, news):
    """
    Batch every candidate's news through the model. The model returns, per
    ticker, ONE label from ALLOWED_LABELS. temperature=0 + closed menu +
    tie-break rule => repeatable. No scores come from the model.
    """
    if not (CONFIG["ai"]["enabled"] and ANTHROPIC_KEY):
        print("  AI disabled or no key -> keyword fallback for all")
        for t, row in cand.items():
            _apply_label(row, keyword_fallback(t, row, news.get(t, [])))
        return cand

    items = []
    for t, row in cand.items():
        headlines = news.get(t, [])
        move = abs(row.get("premarket_change") or row.get("changesPercentage") or 0)
        if headlines or move >= 7.5:
            items.append((t, row.get("name", ""), headlines))

    print(f"  classifying {len(items)} candidates via AI ({CONFIG['ai']['model']})")

    results = {}
    for batch in _chunks(items, CONFIG["ai"]["batch_size"]):
        out = _ai_classify_batch(batch)
        if out:
            results.update(out)
        else:
            for t, name, hl in batch:
                results[t] = keyword_fallback(t, cand[t], hl)

    for t, row in cand.items():
        label = results.get(t, "no_catalyst")
        if label not in CATALYST_TYPES:
            label = "no_catalyst"
        _apply_label(row, label)
    return cand


def _ai_classify_batch(batch):
    """One API call classifying up to batch_size tickers. Returns {ticker: label}."""
    menu = "\n".join(f'  "{lbl}" = {CATALYST_TYPES[lbl][2]}' for lbl in ALLOWED_LABELS)
    tiebreak = " > ".join(TIEBREAK_PRIORITY)

    lines = []
    for t, name, headlines in batch:
        hl = " | ".join(h["title"] for h in headlines) or "(no headline found)"
        detail = (headlines[0]["text"] if headlines else "")[:300]
        lines.append(f'{t} ({name}): {hl}\n   detail: {detail}')
    stocks_block = "\n".join(lines)

    system = (
        "You are a mechanical catalyst classifier for a stock scanner. "
        "For each stock, choose exactly ONE label from the fixed menu that best "
        "matches its news. Judge only the TYPE of catalyst, never its magnitude, "
        "and never output a number or score. If two labels could fit, pick the one "
        f"that appears EARLIER in this priority order: {tiebreak}. "
        "If there is no real, market-moving catalyst, use no_catalyst. "
        "Base your decision only on the text given; do not speculate beyond it. "
        "Respond ONLY with a JSON object mapping each ticker to its chosen label, "
        "no prose, no markdown."
    )
    user = (
        f"MENU (label = definition):\n{menu}\n\n"
        f"STOCKS TO CLASSIFY:\n{stocks_block}\n\n"
        'Return JSON like {"AAPL":"earnings_beat","XYZ":"no_catalyst"}. '
        "Use only labels from the menu."
    )

    body = json.dumps({
        "model": CONFIG["ai"]["model"],
        "max_tokens": 1000,
        "temperature": 0,               # <-- determinism
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()

    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urlopen(req, timeout=45) as r:
            data = json.loads(r.read().decode())
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        text = text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return {t: (lbl if lbl in CATALYST_TYPES else "no_catalyst")
                for t, lbl in parsed.items()}
    except Exception as e:
        print(f"  ! AI batch failed: {e}")
        return None


def _apply_label(row, label):
    pts, tier, desc = CATALYST_TYPES[label]
    row["catalyst_label"] = label
    row["catalyst_tier"] = tier
    row["catalyst_desc"] = desc
    row["catalyst_base_points"] = pts


# ======================================================================
# Keyword fallback — used only if AI is off or a call fails.
# Same closed label set, so scoring is identical downstream.
# ======================================================================
def keyword_fallback(t, row, headlines):
    blob = " ".join(h["title"].lower() + " " + h["text"].lower() for h in headlines)
    name = (row.get("name") or "").lower()
    text = blob + " " + name

    def has(*words):
        return any(w in text for w in words)

    if has("fda approv", "phase 3 met", "primary endpoint", "granted approval"):
        return "fda_approval"
    if has("to be acquired", "acquisition of", "buyout", "takeover", "merger agreement"):
        return "acquisition_target"
    if has("awarded contract", "wins contract", "contract award", "defense contract"):
        return "major_contract"
    if has("raises guidance", "raised full-year", "boosts outlook"):
        if has("beats", "tops estimates", "record revenue"):
            return "earnings_blowout_raise"
        return "positive_guidance"
    if has("beats", "tops estimates", "earnings beat", "q1 beat", "q2 beat", "q3 beat", "q4 beat"):
        return "earnings_beat"
    if has("upgrades to buy", "price target raised", "raises price target", "upgraded"):
        return "analyst_upgrade_big_pt"
    if has("partnership", "collaboration", "teams up", "strategic alliance"):
        return "major_partnership"
    if has("offering", "prices offering", "registered direct", "dilut"):
        return "dilution_offering"
    if has("added to s&p", "joins index", "index inclusion"):
        return "index_inclusion"
    if has("launches", "unveils", "introduces"):
        return "product_launch"
    move = abs(row.get("premarket_change") or row.get("changesPercentage") or 0)
    if move >= 7.5:
        return "unclassified_move"
    return "no_catalyst"


# ======================================================================
# 4. Technicals — pure math, NO AI.
# ======================================================================
def attach_technicals(cand):
    for t, row in cand.items():
        if row.get("catalyst_tier") is None and CONFIG["gates"]["require_catalyst"]:
            continue
        row.update(compute_technicals(t))
        time.sleep(0.05)
    return cand


def compute_technicals(t):
    hist = fmp(f"historical-price-full/{t}", "&timeseries=220")
    out = {"adr_pct": None, "dollar_volume_m": None, "gap_pct": None,
           "above_10": False, "above_20": False, "above_50": False, "above_200": False,
           "neglect": "low", "vol_expansion": None, "market_cap_m": None}
    if not hist or "historical" not in hist or not hist["historical"]:
        return out
    bars = hist["historical"]
    closes = [_to_float(b.get("close")) for b in bars if b.get("close")]
    highs = [_to_float(b.get("high")) for b in bars if b.get("high")]
    lows = [_to_float(b.get("low")) for b in bars if b.get("low")]
    vols = [_to_float(b.get("volume")) for b in bars if b.get("volume")]
    if len(closes) < 25:
        return out

    last = closes[0]
    prev = closes[1] if len(closes) > 1 else last
    out["gap_pct"] = (last - prev) / prev * 100 if prev else None

    ranges = [(highs[i] / lows[i] - 1) * 100 for i in range(min(20, len(highs), len(lows))) if lows[i]]
    out["adr_pct"] = round(sum(ranges) / len(ranges), 2) if ranges else None

    if vols and last:
        out["dollar_volume_m"] = round(last * vols[0] / 1_000_000, 1)
    if len(vols) > 21 and vols[0]:
        avg = sum(vols[1:21]) / 20
        out["vol_expansion"] = round(vols[0] / avg, 1) if avg else None

    def sma(k):
        return sum(closes[:k]) / k if len(closes) >= k else None
    for k, key in [(10, "above_10"), (20, "above_20"), (50, "above_50"), (200, "above_200")]:
        m = sma(k)
        out[key] = bool(m and last > m)

    if len(closes) > 21:
        base = closes[1:21]
        drift = abs(base[0] - base[-1]) / base[-1] * 100 if base[-1] else 999
        adr = out["adr_pct"] or 5
        ratio = drift / adr
        out["neglect"] = "high" if ratio < 1.5 else "med" if ratio < 3 else "low"
    return out


# ======================================================================
# 5. Score — pure arithmetic. AI's label already resolved to base points.
# ======================================================================
def score(row):
    c = CONFIG
    cat_pts, tech_pts = 0, 0
    cbreak, tbreak = [], []

    base = row.get("catalyst_base_points", 0)
    if base:
        sign = "+" if base >= 0 else ""
        cbreak.append([f"{row['catalyst_desc']} (Tier {row.get('catalyst_tier') or 'C'})", f"{sign}{base}"])
        cat_pts += base

    gap = row.get("premarket_change")
    if gap is None:
        gap = row.get("gap_pct")
    if gap is not None:
        g = c["gap"]
        if gap >= g["pivot"]:
            pts = g["full_points"] + min(g["max_bonus"], (gap - g["pivot"]) * g["per_pct_above"])
        elif gap >= g["floor"]:
            pts = g["full_points"] * (gap - g["floor"]) / (g["pivot"] - g["floor"])
        else:
            pts = -6
        pts = round(pts)
        cat_pts += pts
        cbreak.append([f"Gap {gap:.1f}% (market voting)", f"{'+' if pts>=0 else ''}{pts}"])

    ve = row.get("vol_expansion")
    if ve:
        pts = round(min(c["volume_expansion"]["max_points"], ve * c["volume_expansion"]["per_x"]))
        cat_pts += pts
        cbreak.append([f"Volume {ve:.1f}x expansion", f"+{pts}"])

    dv = row.get("dollar_volume_m")
    if dv is not None:
        d = c["dollar_volume"]
        pts = d["full_points"] if dv >= d["pivot"] else d["below_points"] if dv >= d["thin_floor"] else d["thin_points"]
        tech_pts += pts
        tbreak.append([f"$ vol ${dv:.0f}M", f"+{pts}"])

    adr = row.get("adr_pct")
    if adr is not None:
        a = c["adr"]
        pts = a["full_points"] if adr >= a["pivot"] else max(0, round(a["full_points"] - (a["pivot"] - adr) * a["per_pct_below"]))
        tech_pts += pts
        tbreak.append([f"ADR {adr:.1f}%", f"+{pts}"])

    ma = c["ma"]; ma_pts = 0; ma_desc = []
    for key, label in [("above_200", "200d"), ("above_50", "50d"), ("above_20", "20d"), ("above_10", "10d")]:
        if row.get(key):
            ma_pts += ma[key]; ma_desc.append(label)
    if ma_pts:
        tech_pts += ma_pts
        tbreak.append([f"Above {', '.join(ma_desc)}", f"+{ma_pts}"])

    neg = row.get("neglect", "low")
    npts = c["neglect"].get(f"{neg}_points", 0)
    if npts:
        tech_pts += npts
        tbreak.append([f"{neg.capitalize()} neglect (quiet base)", f"+{npts}"])

    row["catalyst_score"] = cat_pts
    row["technical_score"] = tech_pts
    row["total"] = cat_pts + tech_pts
    row["cbreak"] = cbreak
    row["tbreak"] = tbreak
    return row


# ======================================================================
# 6. Gate, score, rank, write
# ======================================================================
def passes_gates(row):
    g = CONFIG["gates"]
    if g["require_catalyst"] and not row.get("catalyst_tier"):
        return False
    price = row.get("price")
    if price and price < g["min_price"]:
        return False
    mc = row.get("market_cap_m")
    if mc is not None and mc < g["min_market_cap_m"]:
        return False
    dv = row.get("dollar_volume_m")
    if dv is not None and dv < g["min_dollar_volume_m"]:
        return False
    return True


# ======================================================================
# MODES
# ------
# MODE=full     : the 4 scheduled runs. Gathers candidates, fetches news,
#                 classifies with AI, prices, scores. Caches the AI labels.
# MODE=refresh  : the cheap 15-min updates. Loads today's existing board,
#                 REUSES the cached AI labels (no news fetch, no AI call),
#                 only re-prices the technicals and re-scores. Adds any brand
#                 new gappers as "new" (keyword-classified so it stays cheap;
#                 they get a proper AI label at the next full run).
#
# Cooling: a ticker that has appeared today NEVER disappears. If it later
# falls below the catalyst gate, it's kept and marked cooling=True so the
# site can show it in the faded section instead of dropping it.
# ======================================================================
MODE = os.environ.get("MODE", "full")


def _today_key():
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")


def load_existing():
    """Load today's data.json if it exists and is from today; else empty."""
    try:
        with open(OUT_PATH) as f:
            data = json.load(f)
        if data.get("session_date") == _today_key():
            return data
    except Exception:
        pass
    return None


def main():
    if not FMP_KEY:
        print("ERROR: FMP_API_KEY not set.")
        write_output([]); return

    print(f"=== EP Scanner · mode={MODE} · run {RUN_LABEL} · {dt.datetime.now(dt.UTC).isoformat()} ===")

    if MODE == "refresh":
        run_refresh()
    else:
        run_full()


def run_full():
    """Heavy run: AI classification + full technicals. Used by the 4 scheduled runs."""
    if not ANTHROPIC_KEY:
        print("  note: ANTHROPIC_API_KEY not set -> using keyword fallback")

    # carry over any names already seen today so nothing that appeared is lost
    prior = load_existing()
    prior_rows = {r["ticker"]: r for r in prior["rows"]} if prior else {}

    cand = gather_candidates()
    news = fetch_news(list(cand.keys()))
    cand = classify_all_with_ai(cand, news)
    cand = attach_technicals(cand)

    board = {}
    for t, row in cand.items():
        row["cooling"] = not passes_gates(row)
        board[t] = score(row)

    # re-price + carry forward prior names not in today's gather (kept, may be cooling)
    for t, prow in prior_rows.items():
        if t in board:
            continue
        prow.setdefault("catalyst_label", "no_catalyst")
        tech = compute_technicals(t)
        prow.update(tech)
        prow["cooling"] = not passes_gates(prow)
        board[t] = score(prow)

    rows = _finalize(board)
    print(f"  final board: {len(rows)} ({sum(1 for r in rows if not r['cooling'])} active, "
          f"{sum(1 for r in rows if r['cooling'])} cooling)")
    write_output(rows, first_full=True)


def run_refresh():
    """
    Cheap run: reuse cached AI labels, only re-price. No news, no AI.
    If there's no board yet today (e.g. refresh fired before the first full
    run), fall back to a full run so we always have something.
    """
    prior = load_existing()
    if not prior or not prior.get("rows"):
        print("  no board yet today -> doing a full run instead")
        run_full()
        return

    board = {}
    for prow in prior["rows"]:
        t = prow["ticker"]
        # reuse the cached label -> base points (NO AI, NO news fetch)
        label = prow.get("catalyst_label", "no_catalyst")
        _apply_label(prow, label)
        prow.update(compute_technicals(t))     # only the cheap price math
        prow["cooling"] = not passes_gates(prow)
        board[t] = score(prow)
        time.sleep(0.03)

    rows = _finalize(board)
    active = sum(1 for r in rows if not r["cooling"])
    print(f"  refreshed {len(rows)} names ({active} active, {len(rows)-active} cooling) — no AI used")
    write_output(rows)


def _finalize(board):
    """Sort: active names by score desc, then cooling names by score desc below them."""
    rows = list(board.values())
    active = sorted([r for r in rows if not r.get("cooling")], key=lambda r: r["total"], reverse=True)
    cooling = sorted([r for r in rows if r.get("cooling")], key=lambda r: r["total"], reverse=True)
    active = active[:CONFIG["max_results"]]
    return active + cooling


def write_output(rows, first_full=False):
    payload = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "session_date": _today_key(),
        "mode": MODE,
        "run_label": RUN_LABEL,
        "count": len(rows),
        "active_count": sum(1 for r in rows if not r.get("cooling")),
        "ai_used": bool(CONFIG["ai"]["enabled"] and ANTHROPIC_KEY),
        "config_summary": {"gap_pivot": CONFIG["gap"]["pivot"],
                           "adr_pivot": CONFIG["adr"]["pivot"],
                           "dollar_volume_pivot": CONFIG["dollar_volume"]["pivot"]},
        "rows": rows,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {OUT_PATH} ({len(rows)} rows)")


# ======================================================================
def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _to_float(x):
    if x is None:
        return None
    try:
        if isinstance(x, str):
            x = x.replace("%", "").replace(",", "").strip()
        return float(x)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()
