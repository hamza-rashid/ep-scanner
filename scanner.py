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
from urllib.error import HTTPError
from urllib.parse import quote

# Market calendar for the freshness window. pandas_market_calendars knows every
# NYSE holiday and half-day, so "last close -> next open" is always correct,
# including across the July 4th weekend. If the import fails for any reason we
# fall back to a simple weekday rule so the scanner still runs.
try:
    import pandas_market_calendars as mcal
    _NYSE = mcal.get_calendar("XNYS")
    _HAS_CAL = True
except Exception:
    _NYSE = None
    _HAS_CAL = False

# US/Eastern handling without extra deps: use fixed offset via zoneinfo.
try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:
    _ET = None


def fresh_window_utc(now_utc=None):
    """
    Return (window_start_utc, window_end_utc): a catalyst is 'fresh' (day-1 EP)
    if its publish time falls in [previous session close, next session open].

    Uses the real NYSE calendar so holidays/weekends are handled. Example:
    on Monday premarket after a Friday-July-3rd holiday, the window opens at
    Thursday's 4pm ET close and runs to Monday's 9:30am ET open.
    """
    if now_utc is None:
        now_utc = dt.datetime.now(dt.UTC)

    if _HAS_CAL and _ET is not None:
        # Look at a generous span around now to find sessions.
        start = (now_utc - dt.timedelta(days=10)).date()
        end = (now_utc + dt.timedelta(days=3)).date()
        sched = _NYSE.schedule(start_date=start.isoformat(), end_date=end.isoformat())
        # sched has market_open / market_close as tz-aware UTC timestamps.
        opens = list(sched["market_open"])
        closes = list(sched["market_close"])
        # Find the most recent close at or before now = start of the fresh window.
        prev_close = None
        for c in closes:
            c_utc = c.to_pydatetime().astimezone(dt.UTC)
            if c_utc <= now_utc:
                prev_close = c_utc
            else:
                break
        # Find the next open at or after now = end of the fresh window.
        next_open = None
        for o in opens:
            o_utc = o.to_pydatetime().astimezone(dt.UTC)
            if o_utc >= now_utc:
                next_open = o_utc
                break
        if prev_close is None:
            prev_close = now_utc - dt.timedelta(days=4)
        if next_open is None:
            next_open = now_utc + dt.timedelta(days=1)
        return prev_close, next_open

    # Fallback (no calendar lib): last weekday 4pm ET -> next weekday 9:30am ET,
    # skipping Sat/Sun only (won't know holidays, but keeps running).
    ref = now_utc
    # crude ET offset (-4 or -5); use -4 as summer default for the fallback only
    et_off = dt.timedelta(hours=4)
    et_now = ref - et_off
    d = et_now.date()
    # walk back to the previous weekday close
    prev = d
    while True:
        prev_dt = dt.datetime.combine(prev, dt.time(16, 0)) + et_off
        if prev_dt.replace(tzinfo=dt.UTC) <= now_utc and prev.weekday() < 5:
            break
        prev = prev - dt.timedelta(days=1)
    nxt = d
    while nxt.weekday() >= 5:
        nxt = nxt + dt.timedelta(days=1)
    next_dt = (dt.datetime.combine(nxt, dt.time(9, 30)) + et_off).replace(tzinfo=dt.UTC)
    return prev_dt.replace(tzinfo=dt.UTC), next_dt


def parse_news_time(s):
    """FMP news publishedDate looks like '2026-07-02 09:25:00' (US/Eastern)."""
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            naive = dt.datetime.strptime(s.strip()[:19], fmt)
            if _ET is not None:
                return naive.replace(tzinfo=_ET).astimezone(dt.UTC)
            # assume ET as -4 if no zoneinfo
            return (naive + dt.timedelta(hours=4)).replace(tzinfo=dt.UTC)
        except ValueError:
            continue
    return None


# ======================================================================
# CATALYST TYPES  — the fixed menu the AI must choose from.
# The AI picks the LABEL. The points come from here. Tune points freely.
# `tier` is just for the badge colour on the site.
# ======================================================================
CATALYST_TYPES = {
    # Ordered by Stockbee/Bonde's own catalyst hierarchy: fundamental
    # re-ratings (earnings/sales acceleration, FDA, contracts) at the top;
    # media mentions at the bottom (Bonde rates those low-success / one-day).
    # Points are on a ~0-100 scale so catalyst dominates the total (~60%).
    # label                       points  tier   human description
    "earnings_accel_blowout":    (100, "S", "Earnings/sales acceleration, blowout + raised guidance"),
    "fda_approval":              (98,  "S", "FDA approval / positive pivotal trial"),
    "acquisition_target":        (96,  "S", "Acquisition / buyout (target)"),
    "biotech_bigpharma_deal":    (92,  "S", "Biotech tie-up with large pharma"),
    "major_contract":            (90,  "S", "Major new contract / government order"),
    "earnings_beat_growth":      (84,  "A", "Earnings beat with strong sales/EPS growth"),
    "sales_acceleration":        (82,  "A", "Sales acceleration"),
    "guidance_raised":           (78,  "A", "Company raised earnings guidance"),
    "new_orders":                (74,  "A", "New orders / demand surge"),
    "sector_runaway":            (70,  "A", "Sector runaway move"),
    "insider_buy_1m":            (66,  "A", "Insider buying > $1M"),
    "new_product":               (62,  "B", "New product launch / major product news"),
    "analyst_upgrade_big_pt":    (58,  "B", "Analyst upgrade + big price-target raise"),
    "ibd_highlight":             (54,  "B", "IBD rating / IBD 100 / IBD highlight"),
    "legal_ruling_favorable":    (60,  "B", "Favourable legal / regulatory ruling"),
    "index_inclusion":           (50,  "B", "Index inclusion"),
    "earnings_inline":           (44,  "B", "In-line earnings, strong reaction"),
    "media_mention":             (24,  "C", "Barron's / Cramer / media mention (low success)"),
    "vague_pr":                  (20,  "C", "Vague PR / promotional"),
    "unclassified_move":         (22,  "C", "Moving on unclear news"),
    "dilution_offering":         (-15, "C", "Share offering / dilution (bearish)"),
    "no_catalyst":               (0,   None, "No real catalyst"),
}

# The closed list of labels the AI is allowed to return.
ALLOWED_LABELS = list(CATALYST_TYPES.keys())

# Deterministic tie-break priority: prefer the type higher in this list when
# two could fit. Same input -> same winner, every time. Mirrors the hierarchy.
TIEBREAK_PRIORITY = [
    "earnings_accel_blowout", "fda_approval", "acquisition_target",
    "biotech_bigpharma_deal", "major_contract", "earnings_beat_growth",
    "sales_acceleration", "guidance_raised", "new_orders", "sector_runaway",
    "insider_buy_1m", "legal_ruling_favorable", "new_product",
    "analyst_upgrade_big_pt", "ibd_highlight", "index_inclusion",
    "earnings_inline", "media_mention", "unclassified_move", "vague_pr",
    "dilution_offering", "no_catalyst",
]


# ======================================================================
# CONFIG  — tunable weights. Single source of truth.
# Weighting target (Stockbee-informed): catalyst ~60%, neglect ~25%,
# gap/volume/MA ~15%. Catalyst points run 0-100 (above); the technical
# blocks below are scaled so their combined max is ~40% of a top score.
# ======================================================================
CONFIG = {
    # ---- Neglect (SECOND pillar — Bonde lists it as a core EP element) ----
    # Scaled up so a fully-neglected stock adds a big chunk (~25% of total).
    "neglect": {"high_points": 42, "med_points": 24, "low_points": 0},

    # ---- Gap % ramp (Stockbee ~7.5-10% pivot) ----
    "gap": {"pivot": 7.5, "floor": 5.0, "full_points": 12,
            "per_pct_above": 0.4, "max_bonus": 10},

    # ---- Volume-expansion bonus (massive volume near open = key) ----
    "volume_expansion": {"per_x": 2.5, "max_points": 10},

    # ---- Dollar volume ramp (>$100M for best) ----
    "dollar_volume": {"pivot": 100, "full_points": 8, "below_points": 4,
                      "thin_floor": 20, "thin_points": 2},

    # ---- Moving averages (200d most important, then 50d) — minor ----
    "ma": {"above_200": 5, "above_50": 4, "above_20": 2, "above_10": 1},

    # ---- ADR % ramp (Qullamaggie 4% min) — minor, informational-ish ----
    "adr": {"pivot": 4.0, "full_points": 5, "per_pct_below": 1.2},

    # ---- Low-float flag (Stockbee: <20M shares = explosive). INFO ONLY,
    #      does not change score — just flagged on the card. ----
    "low_float_shares_m": 20,

    # ---- Sanity gates (hard filters) ----
    "gates": {"min_price": 3.0, "min_market_cap_m": 100,
              "min_dollar_volume_m": 5, "require_catalyst": True},

    # ---- News freshness window (built on the market calendar) ----
    "freshness": {"enabled": True},

    # ---- AI classification ----
    "ai": {
        "model": "claude-sonnet-5",
        "enabled": True,
        "batch_size": 15,
    },

    "max_results": 40,
}

FMP_KEY = os.environ.get("FMP_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"
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
    """
    Gainers-first: pull what's actually MOVING in premarket, then (later) we
    check each for a fresh catalyst. No point classifying catalysts on stocks
    that aren't gapping. biggest-gainers gives ticker, name, price, change %.
    """
    cand = {}
    gainers = fmp("biggest-gainers") or []
    for g in gainers[:80]:
        t = g.get("symbol")
        if not t:
            continue
        chg = _to_float(g.get("changesPercentage"))
        cand[t] = {"ticker": t, "name": g.get("name", ""),
                   "changesPercentage": chg,
                   "price": _to_float(g.get("price"))}
        # gainers change % is the current move; treat as the premarket gap
        if chg is not None:
            cand[t]["premarket_change"] = chg

    print(f"  candidates gathered: {len(cand)}")
    return cand


def enrich_profile(cand):
    """
    Per-candidate company-profile enrichment (Premium): market cap, 52-week
    range, shares outstanding / float for the low-float flag. Degrades quietly
    if a field is missing so a thin plan never breaks the run.
    """
    for t, row in cand.items():
        prof = fmp("profile", f"&symbol={t}")
        p = prof[0] if isinstance(prof, list) and prof else (prof if isinstance(prof, dict) else None)
        if p:
            mc = _to_float(p.get("marketCap") or p.get("mktCap"))
            if mc:
                row["market_cap_m"] = mc / 1_000_000
            row["price"] = _to_float(p.get("price")) or row.get("price")
            # 52-week range comes as "12.34-56.78"
            rng = p.get("range") or ""
            if "-" in str(rng):
                try:
                    lo, hi = [float(x) for x in str(rng).split("-")[:2]]
                    row["wk52_high"] = hi
                    row["wk52_low"] = lo
                except ValueError:
                    pass
        # shares float (separate endpoint on Premium)
        fl = fmp("shares-float", f"&symbol={t}")
        f = fl[0] if isinstance(fl, list) and fl else (fl if isinstance(fl, dict) else None)
        if f:
            fs = _to_float(f.get("floatShares") or f.get("freeFloat") or f.get("outstandingShares"))
            if fs:
                row["float_m"] = fs / 1_000_000
        time.sleep(0.03)
    return cand


# ======================================================================
# 2. Fetch the NEWS for each candidate (so the AI has something to read)
# ======================================================================
def fetch_news(tickers):
    """
    FMP stock news for our candidates (stable: news/stock?symbols=).
    Returns ticker -> list of recent headline+text snippets for the AI to read.
    """
    news = {}
    for chunk in _chunks(tickers, 20):
        joined = ",".join(chunk)
        data = fmp("news/stock", f"&symbols={quote(joined)}&limit=100") or []
        for n in data:
            t = n.get("symbol")
            if not t:
                continue
            item = {
                "title": (n.get("title") or "").strip(),
                # stable uses "text" for the snippet body
                "text": (n.get("text") or n.get("content") or "")[:400].strip(),
                "published": n.get("publishedDate") or n.get("date") or "",
                "url": (n.get("url") or "").strip(),
                "site": (n.get("site") or n.get("publisher") or "").strip(),
            }
            item["published_utc"] = parse_news_time(item["published"])
            news.setdefault(t, []).append(item)
    for t in news:
        # sort newest-first by parsed time where available, else string
        news[t] = sorted(
            news[t],
            key=lambda x: x["published_utc"] or dt.datetime.min.replace(tzinfo=dt.UTC),
            reverse=True,
        )[:5]
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
        'Return JSON like {"AAPL":"earnings_beat_growth","XYZ":"no_catalyst"}. '
        "Use only labels from the menu."
    )

    body = json.dumps({
        "model": CONFIG["ai"]["model"],
        "max_tokens": 2000,
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
    result = _do_ai_request(req)
    if result is not None:
        return result

    # Retry once WITHOUT temperature — the newest thinking-on models can 400
    # on an explicit temperature. Determinism still holds via the closed menu
    # + tie-break rule, which don't depend on temperature.
    body2 = json.dumps({
        "model": CONFIG["ai"]["model"],
        "max_tokens": 2000,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }).encode()
    req2 = Request(
        "https://api.anthropic.com/v1/messages",
        data=body2,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_KEY,
            "anthropic-version": "2023-06-01",
        },
    )
    return _do_ai_request(req2, note="(retry without temperature)")


def _do_ai_request(req, note=""):
    try:
        with urlopen(req, timeout=45) as r:
            data = json.loads(r.read().decode())
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        text = text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return {t: (lbl if lbl in CATALYST_TYPES else "no_catalyst")
                for t, lbl in parsed.items()}
    except HTTPError as e:
        try:
            err_body = e.read().decode()[:500]
        except Exception:
            err_body = "(no body)"
        print(f"  ! AI batch HTTP {e.code} {note}: {err_body}")
        return None
    except Exception as e:
        print(f"  ! AI batch failed {note}: {e}")
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
    if has("collaboration with", "licensing deal", "partners with pfizer", "big pharma"):
        return "biotech_bigpharma_deal"
    if has("awarded contract", "wins contract", "contract award", "defense contract", "new order"):
        return "major_contract"
    if has("raises guidance", "raised full-year", "boosts outlook", "raises outlook"):
        return "guidance_raised"
    if has("beats", "tops estimates", "earnings beat", "record revenue",
           "sales growth", "revenue growth"):
        return "earnings_beat_growth"
    if has("upgrades to buy", "price target raised", "raises price target", "upgraded"):
        return "analyst_upgrade_big_pt"
    if has("insider buy", "ceo buys", "director buys"):
        return "insider_buy_1m"
    if has("offering", "prices offering", "registered direct", "dilut"):
        return "dilution_offering"
    if has("added to s&p", "joins index", "index inclusion"):
        return "index_inclusion"
    if has("launches", "unveils", "introduces", "new product"):
        return "new_product"
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
    # stable: historical-price-eod/full?symbol=X returns a FLAT array (newest first)
    hist = fmp("historical-price-eod/full", f"&symbol={t}")
    out = {"adr_pct": None, "dollar_volume_m": None, "gap_pct": None,
           "above_10": False, "above_20": False, "above_50": False, "above_200": False,
           "neglect": "low", "vol_expansion": None, "market_cap_m": None}
    # stable returns a plain list; guard for that + legacy dict shape just in case
    if isinstance(hist, dict) and "historical" in hist:
        bars = hist["historical"]
    elif isinstance(hist, list):
        bars = hist
    else:
        bars = None
    if not bars:
        return out
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

    # RVOL (relative volume): today's volume vs the 20-day average. The key
    # premarket participation tell — is the gap backed by real volume?
    if len(vols) > 21 and vols[0]:
        avg20 = sum(vols[1:21]) / 20
        out["rvol"] = round(vols[0] / avg20, 1) if avg20 else None

    # Distance from 52-week high, from the price history (fallback if profile
    # didn't supply it): >0 means below the high, ~0 means at/near blue-sky.
    if closes:
        hi_252 = max(closes[:252]) if len(closes) >= 2 else last
        if hi_252:
            out["pct_below_52w_high"] = round((hi_252 - last) / hi_252 * 100, 1)

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

    # Neglect is its own major pillar (~25%), not a small technical bonus.
    neg = row.get("neglect", "low")
    npts = c["neglect"].get(f"{neg}_points", 0)
    neglect_pts = npts
    nbreak = [[f"{neg.capitalize()} neglect (quiet base)", f"+{npts}"]] if npts else []

    # ---- Info-only enrichments: shown on the card, NOT added to the score ----
    info = []
    rvol = row.get("rvol")
    if rvol is not None:
        info.append(["RVOL", f"{rvol:.1f}x"])
    fl = row.get("float_m")
    if fl is not None:
        low = fl < c["low_float_shares_m"]
        row["low_float"] = low
        info.append(["Float", f"{fl:.0f}M{' · LOW' if low else ''}"])
    pbh = row.get("pct_below_52w_high")
    if pbh is not None:
        tag = "at highs" if pbh <= 2 else f"{pbh:.0f}% below high"
        info.append(["52wk", tag])
    row["info"] = info

    row["catalyst_score"] = cat_pts
    row["neglect_score"] = neglect_pts
    row["technical_score"] = tech_pts
    row["total"] = cat_pts + neglect_pts + tech_pts
    row["cbreak"] = cbreak
    row["nbreak"] = nbreak
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
    global MODE
    if not FMP_KEY:
        print("ERROR: FMP_API_KEY not set.")
        write_output([]); return

    # Auto-detect mode unless explicitly forced. First run of the day (no
    # board yet for today) = full scan; every later run = cheap refresh.
    forced = os.environ.get("MODE", "").strip()
    if forced in ("full", "refresh"):
        MODE = forced
    else:
        MODE = "refresh" if load_existing() else "full"

    print(f"=== EP Scanner · mode={MODE} · run {RUN_LABEL} · {dt.datetime.now(dt.UTC).isoformat()} ===")

    if MODE == "refresh":
        run_refresh()
    else:
        run_full()


def assign_bucket(row, news_items, window):
    """
    Decide the freshness bucket for a candidate:
      "fresh"    - has a catalyst-bearing news item inside [last close, next open]
      "no_news"  - gapping but NO datable news in the window (shown separately)
      "stale"    - newest news is OLDER than the window (discarded)
    Returns (bucket, freshest_item_or_None).
    """
    win_start, win_end = window
    dated = [n for n in news_items if n.get("published_utc")]
    if not dated:
        return "no_news", (news_items[0] if news_items else None)

    freshest = max(dated, key=lambda n: n["published_utc"])
    ts = freshest["published_utc"]
    if win_start <= ts <= win_end + dt.timedelta(hours=1):
        return "fresh", freshest
    # newest news predates the window entirely -> stale, already-played-out
    return "stale", freshest


def run_full():
    """Heavy run: gainers-first, fresh-catalyst filter, AI classification, score."""
    if not ANTHROPIC_KEY:
        print("  note: ANTHROPIC_API_KEY not set -> using keyword fallback")

    prior = load_existing()
    prior_rows = {r["ticker"]: r for r in prior["rows"]} if prior else {}

    window = fresh_window_utc()
    print(f"  fresh window: {window[0].isoformat()} -> {window[1].isoformat()}")

    cand = gather_candidates()

    # Price gate FIRST (cheap) — drop sub-$3 and missing-price before any paid calls
    cand = {t: r for t, r in cand.items()
            if (r.get("price") is not None and r["price"] >= CONFIG["gates"]["min_price"])}
    print(f"  after price >= ${CONFIG['gates']['min_price']:.0f} gate: {len(cand)}")

    news = fetch_news(list(cand.keys()))

    # Bucket by freshness
    fresh, no_news, stale = {}, {}, 0
    for t, row in cand.items():
        bucket, item = assign_bucket(row, news.get(t, []), window)
        if item:
            row["headline"] = item.get("title", "")
            row["news_url"] = item.get("url", "")
            row["news_site"] = item.get("site", "")
            row["news_time"] = item.get("published", "")
        if bucket == "fresh":
            fresh[t] = row
        elif bucket == "no_news":
            no_news[t] = row
        else:
            stale += 1
    print(f"  buckets: {len(fresh)} fresh · {len(no_news)} gapping-no-catalyst · {stale} stale (discarded)")

    # AI-classify ONLY the fresh ones (that's where catalysts matter)
    fresh = classify_all_with_ai(fresh, news)
    fresh = enrich_profile(fresh)
    fresh = attach_technicals(fresh)

    board = {}
    for t, row in fresh.items():
        row["section"] = "ep"
        row["cooling"] = not passes_gates(row)
        board[t] = score(row)

    # gapping-no-catalyst: keep, but mark section; light technicals, no AI
    for t, row in no_news.items():
        _apply_label(row, "no_catalyst")
        row.update(compute_technicals(t))
        row["section"] = "no_catalyst"
        row["cooling"] = False
        board[t] = score(row)

    # carry forward prior fresh names not in today's gather (re-price, keep)
    for t, prow in prior_rows.items():
        if t in board or prow.get("section") != "ep":
            continue
        prow.update(compute_technicals(t))
        prow["cooling"] = not passes_gates(prow)
        board[t] = score(prow)

    rows = _finalize(board)
    print(f"  final board: {len(rows)} rows")
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
        prow.setdefault("section", "ep")
        if prow["section"] == "ep":
            prow["cooling"] = not passes_gates(prow)
        else:
            prow["cooling"] = False
        board[t] = score(prow)
        time.sleep(0.03)

    rows = _finalize(board)
    ep_active = sum(1 for r in rows if r.get("section", "ep") == "ep" and not r["cooling"])
    print(f"  refreshed {len(rows)} names ({ep_active} active EP) — no AI used")
    write_output(rows)


def _finalize(board):
    """
    Order the board:
      1. EP section, active, by score desc
      2. EP section, cooling, by score desc
      3. gapping-no-catalyst section, by gap desc
    """
    rows = list(board.values())
    ep = [r for r in rows if r.get("section", "ep") == "ep"]
    noc = [r for r in rows if r.get("section") == "no_catalyst"]

    ep_active = sorted([r for r in ep if not r.get("cooling")], key=lambda r: r["total"], reverse=True)
    ep_cool = sorted([r for r in ep if r.get("cooling")], key=lambda r: r["total"], reverse=True)
    ep_active = ep_active[:CONFIG["max_results"]]
    noc = sorted(noc, key=lambda r: (r.get("premarket_change") or r.get("gap_pct") or 0), reverse=True)
    return ep_active + ep_cool + noc


def write_output(rows, first_full=False):
    ep = [r for r in rows if r.get("section", "ep") == "ep"]
    payload = {
        "generated_utc": dt.datetime.now(dt.UTC).isoformat(),
        "session_date": _today_key(),
        "mode": MODE,
        "run_label": RUN_LABEL,
        "count": len(rows),
        "active_count": sum(1 for r in ep if not r.get("cooling")),
        "no_catalyst_count": sum(1 for r in rows if r.get("section") == "no_catalyst"),
        "ai_used": bool(CONFIG["ai"]["enabled"] and ANTHROPIC_KEY),
        "fresh_window_ok": _HAS_CAL,
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
