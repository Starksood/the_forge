```markdown
# Enthusiast Car Auction Price Intelligence API
# Comprehensive System Specification v1.0
# Intended for: Kiro spec-driven development

---

## 1. PRODUCT VISION

### 1.1 What We Are Building

A data infrastructure product: a continuously-updated, structured database of completed
enthusiast car auction results, exposed via a queryable REST API. The primary data sources
are Bring a Trailer (bringatrailer.com) and Cars & Bids (carsandbids.com). Both publish
completed auction results publicly. Neither offers a developer API. That gap is the product.

The value proposition is pricing intelligence. Anyone building a car-adjacent app —
valuation tools, insurance products, market trackers, dealer inventory systems — needs
to answer the question: "What does this specific car actually sell for?" This API answers
that question with real transaction data, not estimates.

### 1.2 Who Pays and Why

Primary customer personas, in order of willingness to pay:

PERSONA A — Independent Developer Building a Car Valuation Tool
  Pain: Needs historical pricing data to show users "your car is worth X"
  Budget: $9–$29/month
  Use pattern: 1,000–50,000 API calls/month
  Where to find them: RapidAPI marketplace, r/webdev, IndieHackers

PERSONA B — Automotive Dealer or Broker
  Pain: Needs to price enthusiast inventory against real market comps
  Budget: $29–$99/month
  Use pattern: Low volume, high value per query
  Where to find them: LinkedIn, SEMA network, Cars & Bids comment sections

PERSONA C — Enthusiast Car Insurance Company (Hagerty-style)
  Pain: Needs transaction data to underwrite agreed-value policies
  Budget: $99–$500/month, possibly direct contract
  Use pattern: Batch queries, historical depth matters most
  Where to find them: Direct outreach after product is live

PERSONA D — Hobbyist Developer / Prosumer
  Pain: Wants to track a specific car's market over time
  Budget: Free tier or $9/month
  Use pattern: Very low volume, long tail
  Where to find them: r/cars, BaT comments, car forums

### 1.3 Revenue Model

Freemium API sold via RapidAPI marketplace.

  Tier        Price       Calls/month   Key Restrictions
  Free        $0          500           /auctions + /vehicles only, no /pricing
  Developer   $9/month    10,000        All endpoints, no trend data
  Pro         $29/month   100,000       All endpoints including trend + mileage bands
  Business    $99/month   Unlimited     All endpoints + SLA + email support

Revenue math to $1,000/month:
  Option A: 112 Developer subscribers
  Option B: 35 Pro subscribers  
  Option C: 10 Business + 20 Developer = $1,040/month
  Realistic mix at 6 months: ~20 Developer + ~15 Pro + ~2 Business = $895/month

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Component Map

```
EXTERNAL SOURCES
  bringatrailer.com     Public completed auction pages + embedded JSON
  carsandbids.com       Public past auction HTML pages

INGESTION PIPELINE
  Scraper Workers       Fetch raw HTML/JSON from sources
  Raw Storage           Append-only archive in PostgreSQL (raw_listings table)
  Parser                Extract structured fields from raw content
  Normalizer            Canonicalize make/model names, convert units, flag signals
  DB Writer             Upsert to auctions table, never duplicate

STORAGE
  PostgreSQL            Primary datastore — all auction records + vehicle catalog
  Redis                 Response cache — TTL-keyed per endpoint + query params

AGGREGATION
  Nightly Task          Rebuild price_snapshots table from auctions table
  Scheduler             APScheduler running inside API container

API SERVER
  FastAPI               Python 3.11, async, Pydantic v2 validation
  Four routers          /auctions, /pricing, /vehicles, /health
  Cache layer           Redis decorator on all read endpoints

DELIVERY
  RapidAPI Gateway      Handles API key auth, rate limiting, billing, discovery
  Railway.app           Hosts API container, Postgres, Redis

MONITORING
  UptimeRobot           Pings /v1/health every 5 minutes, emails on failure
  Railway Logs          Scraper run logs, error traces
```

### 2.2 Data Flow — Happy Path

1. APScheduler fires bat_scrape at 02:00 UTC
2. BaT scraper fetches page 1 of completed auctions from BaT JSON endpoint
3. Raw JSON blob saved to raw_listings (source="bat", source_id=listing_id)
4. Parser extracts fields: year, make, model, mileage, sale_price, transmission, etc.
5. Normalizer runs: lowercase make/model, price string to integer cents, parse date
6. DB Writer upserts to auctions table on (source, source_id) conflict → do nothing
7. Scraper sleeps SCRAPE_DELAY_SECONDS, fetches next page
8. Loop until SCRAPE_MAX_PAGES_PER_RUN reached or no new results found
9. Same flow repeats for cnb_scrape at 02:30 UTC
10. rebuild_price_snapshots fires at 03:30 UTC
11. Aggregation task queries auctions table, computes percentiles per make/model/window
12. Writes results to price_snapshots table (truncate + insert each night)
13. API requests hit FastAPI, which reads from price_snapshots (pricing) or auctions (search)
14. Redis cache stores response by hashed query key, TTL varies by endpoint
15. RapidAPI gateway validates API key, increments call counter, proxies to FastAPI

### 2.3 Infrastructure Sizing (MVP)

All services hosted on Railway.app:

  Service       Plan         Cost/month    Notes
  API           Starter      $5            FastAPI + APScheduler in one container
  PostgreSQL    Hobby        $5            5GB storage, enough for 500k+ records
  Redis         Hobby        $3            256MB, more than enough for cache
  Total                      ~$13/month

At this scale (pre-1M rows) there is no need for read replicas, connection pooling
middleware, or separate worker processes. Keep it simple. Add complexity only when
metrics demand it.

---

## 3. DATABASE SCHEMA

### 3.1 Design Principles

- Prices stored as INTEGER in USD cents. Never FLOAT or NUMERIC for money.
  Reason: 52500 (cents) not 525.00 (dollars). Avoids floating point precision bugs.
- raw_listings is append-only. Never update or delete. It is your recovery mechanism
  if a parser bug corrupts auctions data — you can re-parse from raw.
- auctions is the normalized working table. Upsert only, never bulk delete.
- price_snapshots is a materialized-style cache. Truncate + rebuild nightly. Treat
  it as disposable — it can always be regenerated from auctions.
- vehicles is a manually-curated reference table. Seed it at launch. Update manually
  as new generations are added. Do NOT auto-populate from scrapers.

### 3.2 Full Schema DDL

```sql
-- ============================================================
-- VEHICLES: Canonical vehicle reference table
-- Manually curated. All auctions optionally FK to this.
-- ============================================================
CREATE TABLE vehicles (
    id            SERIAL PRIMARY KEY,
    make          TEXT NOT NULL,
    model         TEXT NOT NULL,
    submodel      TEXT,
    generation    TEXT,
    year_min      INTEGER NOT NULL,
    year_max      INTEGER,           -- NULL if generation still in production
    body_style    TEXT,              -- coupe | convertible | sedan | wagon | hatchback
    drivetrain    TEXT,              -- RWD | AWD | FWD | 4WD
    engine        TEXT,              -- human-readable e.g. "flat-6 3.6L"
    is_turbo      BOOLEAN DEFAULT FALSE,
    displacement_cc INTEGER,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(make, model, submodel, generation)
);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER vehicles_updated_at
    BEFORE UPDATE ON vehicles
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- RAW_LISTINGS: Append-only archive of all raw fetched content
-- Written BEFORE parsing. Never updated. Never deleted.
-- Enables re-parsing if schema or parser logic changes.
-- ============================================================
CREATE TABLE raw_listings (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,       -- "bat" | "carsandbids"
    source_id   TEXT NOT NULL,       -- original listing ID from source
    raw_html    TEXT,                -- full HTML if HTML-parsed source
    raw_json    JSONB,               -- parsed JSON if JSON source (BaT)
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(source, source_id)
);

CREATE INDEX idx_raw_listings_source_id ON raw_listings(source, source_id);
CREATE INDEX idx_raw_listings_fetched   ON raw_listings(fetched_at DESC);

-- ============================================================
-- AUCTIONS: Normalized auction records
-- One row per completed auction listing.
-- Upsert only. Conflict target: (source, source_id).
-- ============================================================
CREATE TABLE auctions (
    id                  BIGSERIAL PRIMARY KEY,

    -- Source identity
    source              TEXT NOT NULL,   -- "bat" | "carsandbids"
    source_id           TEXT NOT NULL,   -- original listing ID
    source_url          TEXT NOT NULL,   -- canonical URL to listing

    -- Vehicle FK (nullable — not all listings match a known vehicle)
    vehicle_id          INTEGER REFERENCES vehicles(id) ON DELETE SET NULL,

    -- Vehicle attributes (denormalized for query performance)
    -- Do not normalize these to a join — query speed matters more than DRY
    year                SMALLINT,
    make                TEXT,            -- always lowercase e.g. "porsche"
    model               TEXT,            -- always lowercase e.g. "911"
    trim                TEXT,            -- e.g. "Carrera S", "GT3 RS"
    generation          TEXT,            -- e.g. "997.2", "NA", "A80"
    body_style          TEXT,

    -- Odometer
    mileage             INTEGER,         -- in miles, integer
    mileage_confidence  TEXT,            -- "exact" | "estimated" | "exempt" | "unknown"
    -- mileage_confidence="exempt" when listing says "TMU" or "Exempt"
    -- mileage_confidence="estimated" when listing says "approximately"

    -- Cosmetics
    color_exterior      TEXT,
    color_interior      TEXT,

    -- Drivetrain
    transmission        TEXT,            -- "manual" | "automatic" | "dct" | "unknown"
    drivetrain          TEXT,            -- "RWD" | "AWD" | "FWD"
    engine_description  TEXT,            -- raw string from listing

    -- Location
    location_state      TEXT,            -- US 2-letter state code or country name
    location_country    TEXT DEFAULT 'US',

    -- Auction result
    sale_price          INTEGER,         -- USD cents. NULL if not sold / passed.
    reserve_met         BOOLEAN,         -- NULL if no-reserve
    no_reserve          BOOLEAN NOT NULL DEFAULT FALSE,
    bid_count           INTEGER,
    sold                BOOLEAN NOT NULL, -- TRUE = sold, FALSE = passed/no sale
    auction_end_date    DATE NOT NULL,
    auction_end_ts      TIMESTAMPTZ,     -- full timestamp if available

    -- Condition signals (extracted from description text + title)
    has_service_records BOOLEAN,
    has_original_books  BOOLEAN,
    is_numbers_matching BOOLEAN,
    rust_noted          BOOLEAN NOT NULL DEFAULT FALSE,
    accident_noted      BOOLEAN NOT NULL DEFAULT FALSE,
    repaint_noted       BOOLEAN NOT NULL DEFAULT FALSE,

    -- Modification signals
    is_modified         BOOLEAN NOT NULL DEFAULT FALSE,
    mod_summary         TEXT[],          -- ["cold air intake", "coilovers", "wheels"]
    mod_engine          BOOLEAN NOT NULL DEFAULT FALSE,
    mod_suspension      BOOLEAN NOT NULL DEFAULT FALSE,
    mod_wheels          BOOLEAN NOT NULL DEFAULT FALSE,
    mod_exhaust         BOOLEAN NOT NULL DEFAULT FALSE,
    mod_forced_induction BOOLEAN NOT NULL DEFAULT FALSE,

    -- Raw content (stored for future NLP/search use)
    title               TEXT,
    description_text    TEXT,
    image_count         SMALLINT,
    image_urls          TEXT[],          -- first 5 image URLs only
    comment_count       INTEGER,

    -- Processing metadata
    scraped_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parsed_at           TIMESTAMPTZ,
    parse_version       SMALLINT DEFAULT 1, -- increment when parser logic changes
    parse_warnings      TEXT[],          -- non-fatal issues logged during parse

    UNIQUE(source, source_id)
);

-- Core query indexes
CREATE INDEX idx_auctions_make_model       ON auctions(make, model);
CREATE INDEX idx_auctions_make_model_year  ON auctions(make, model, year);
CREATE INDEX idx_auctions_year             ON auctions(year);
CREATE INDEX idx_auctions_end_date         ON auctions(auction_end_date DESC);
CREATE INDEX idx_auctions_sold             ON auctions(sold);
CREATE INDEX idx_auctions_transmission     ON auctions(transmission);
CREATE INDEX idx_auctions_source           ON auctions(source);
CREATE INDEX idx_auctions_no_reserve       ON auctions(no_reserve);
CREATE INDEX idx_auctions_is_modified      ON auctions(is_modified);
CREATE INDEX idx_auctions_mileage          ON auctions(mileage);
CREATE INDEX idx_auctions_sale_price       ON auctions(sale_price) WHERE sold = TRUE;
CREATE INDEX idx_auctions_location         ON auctions(location_state);

-- Composite index for the most common /pricing query pattern
CREATE INDEX idx_auctions_pricing_core
    ON auctions(make, model, year, sold, auction_end_date DESC)
    WHERE sold = TRUE;

-- Full-text search on title (for fuzzy vehicle matching, future use)
CREATE INDEX idx_auctions_title_fts
    ON auctions USING GIN(to_tsvector('english', COALESCE(title, '')));

-- ============================================================
-- PRICE_SNAPSHOTS: Pre-aggregated pricing stats
-- Rebuilt nightly by rebuild_price_snapshots task.
-- Treat as a materialized view — disposable + regenerable.
-- ============================================================
CREATE TABLE price_snapshots (
    id              BIGSERIAL PRIMARY KEY,

    -- Dimension keys (what was filtered to produce this snapshot)
    make            TEXT NOT NULL,
    model           TEXT NOT NULL,
    year_min        SMALLINT,           -- NULL means all years
    year_max        SMALLINT,           -- NULL means all years
    transmission    TEXT,               -- NULL means all transmissions
    condition       TEXT,               -- "stock" | "modified" | NULL (all)
    window_days     SMALLINT NOT NULL,  -- 90 | 180 | 365 | 0 (0 = all-time)

    -- Stats (all prices in USD cents)
    sample_size     INTEGER NOT NULL,
    avg_price       INTEGER,
    median_price    INTEGER,
    min_price       INTEGER,
    max_price       INTEGER,
    p10_price       INTEGER,
    p25_price       INTEGER,
    p75_price       INTEGER,
    p90_price       INTEGER,
    stddev_price    INTEGER,
    sell_through    NUMERIC(5,4),       -- ratio of sold / total (including passed)

    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Composite unique key for upsert
    UNIQUE(make, model, year_min, year_max, transmission, condition, window_days)
);

CREATE INDEX idx_snapshots_make_model ON price_snapshots(make, model);
CREATE INDEX idx_snapshots_computed   ON price_snapshots(computed_at DESC);

-- ============================================================
-- SCRAPE_RUNS: Audit log of every scraper execution
-- Used by /health endpoint and for debugging scraper issues.
-- ============================================================
CREATE TABLE scrape_runs (
    id              BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    status          TEXT,               -- "running" | "success" | "partial" | "failed"
    pages_fetched   INTEGER DEFAULT 0,
    records_seen    INTEGER DEFAULT 0,
    records_new     INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    error_message   TEXT,
    run_metadata    JSONB               -- any extra diagnostic info
);

CREATE INDEX idx_scrape_runs_source  ON scrape_runs(source);
CREATE INDEX idx_scrape_runs_started ON scrape_runs(started_at DESC);
```

### 3.3 Make/Model Canonicalization Rules

All make and model values stored in auctions must be lowercase, normalized.
The normalizer.py module handles this mapping before any DB write.

```
Raw Input         → Stored As
"Porsche"         → make="porsche"
"PORSCHE"         → make="porsche"
"porsche 911"     → make="porsche", model="911"
"Mazda Miata"     → make="mazda", model="miata"
"BMW M3"          → make="bmw", model="m3"
"Nissan GT-R"     → make="nissan", model="gt-r"
"Mercedes-Benz"   → make="mercedes-benz"
"VW"              → make="volkswagen"
"Chevy"           → make="chevrolet"
```

Build a static MAKE_ALIASES dict in normalizer.py:
```python
MAKE_ALIASES = {
    "vw": "volkswagen",
    "chevy": "chevrolet",
    "merc": "mercedes-benz",
    "mercedes": "mercedes-benz",
    "benz": "mercedes-benz",
    "alfa": "alfa romeo",
    "aston": "aston martin",
    "land rover": "land-rover",
    "rolls": "rolls-royce",
    "bentley": "bentley",
}
```

---

## 4. SCRAPER DESIGN

### 4.1 BaT Scraper Architecture

**Source behavior:**
BaT is a JS-heavy SPA. The completed auction listing data is embedded as JSON in the
page source under a `window.BATData` variable, AND exposed via an undocumented
WordPress REST endpoint. Target the REST endpoint — it is more stable and efficient.

**Endpoint (undocumented, subject to change):**
```
GET https://bringatrailer.com/wp-json/bat/v1/auctions/completed
  ?page=1
  &per_page=50
```
Returns JSON array of auction objects. Paginate by incrementing `page` until empty.

**Incremental scraping strategy:**
On first run: scrape all pages until empty (historical backfill).
On subsequent runs: scrape pages until we encounter a source_id that already exists
in raw_listings. Stop early to avoid unnecessary requests.

```python
# scraper/bat.py

import httpx
import re
import time
import logging
from datetime import datetime
from typing import Generator

from db.session import get_session
from db import crud
from scraper.normalizer import normalize_listing
from config import settings

logger = logging.getLogger(__name__)

BAT_COMPLETED_ENDPOINT = (
    "https://bringatrailer.com/wp-json/bat/v1/auctions/completed"
)

HEADERS = {
    "User-Agent": settings.USER_AGENT,
    "Accept": "application/json",
    "Referer": "https://bringatrailer.com/auctions/results/",
}


def fetch_page(page: int, per_page: int = 50) -> list[dict]:
    """
    Fetch one page of completed BaT auctions.
    Returns list of raw auction dicts.
    Raises httpx.HTTPStatusError on non-2xx responses.
    """
    resp = httpx.get(
        BAT_COMPLETED_ENDPOINT,
        params={"page": page, "per_page": per_page},
        headers=HEADERS,
        timeout=20,
        follow_redirects=True,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", []) or data.get("auctions", []) or []


def parse_listing(raw: dict) -> dict:
    """
    Map raw BaT API response dict to our normalized schema dict.
    All field names match auctions table columns.
    Does NOT write to DB — pure transformation.
    """
    source_id = str(raw.get("id") or raw.get("listing_id", ""))

    # Price: BaT returns integer dollars, we store cents
    sold_price_raw = raw.get("sold_price") or raw.get("high_bid")
    sale_price_cents = int(sold_price_raw) * 100 if sold_price_raw else None

    sold = bool(raw.get("sold") or raw.get("is_sold"))

    return {
        "source": "bat",
        "source_id": source_id,
        "source_url": raw.get("url") or f"https://bringatrailer.com/listing/{source_id}/",
        "year": _safe_int(raw.get("model_year") or raw.get("year")),
        "make": raw.get("make", "").lower().strip() or None,
        "model": raw.get("model", "").lower().strip() or None,
        "trim": raw.get("trim") or None,
        "mileage": parse_mileage(raw.get("mileage_description", "")),
        "mileage_confidence": parse_mileage_confidence(raw.get("mileage_description", "")),
        "color_exterior": raw.get("exterior_color") or None,
        "color_interior": raw.get("interior_color") or None,
        "transmission": parse_transmission(raw.get("transmission", "")),
        "location_state": parse_state(raw.get("location", "")),
        "sale_price": sale_price_cents,
        "no_reserve": bool(raw.get("noReserve") or raw.get("no_reserve")),
        "sold": sold,
        "bid_count": _safe_int(raw.get("bid_count") or raw.get("bids")),
        "auction_end_date": parse_date(raw.get("ended_at_date") or raw.get("end_date")),
        "title": raw.get("title", "").strip() or None,
        "image_urls": (raw.get("images") or raw.get("photos", []))[:5],
        "image_count": len(raw.get("images") or raw.get("photos", [])),
        "comment_count": _safe_int(raw.get("comment_count")),
    }


def parse_mileage(text: str) -> int | None:
    """Extract integer miles from strings like '67,400 miles' or '~50k miles'."""
    if not text:
        return None
    text = text.strip()
    if any(x in text.lower() for x in ["exempt", "tmu", "not actual", "unknown"]):
        return None
    # Handle "approximately 50k" style
    approx = re.search(r"(\d+\.?\d*)\s*k\s*mile", text, re.IGNORECASE)
    if approx:
        return int(float(approx.group(1)) * 1000)
    exact = re.search(r"([\d,]+)\s*mile", text, re.IGNORECASE)
    if exact:
        return int(exact.group(1).replace(",", ""))
    return None


def parse_mileage_confidence(text: str) -> str:
    """Return 'exact' | 'estimated' | 'exempt' | 'unknown'."""
    if not text:
        return "unknown"
    lower = text.lower()
    if any(x in lower for x in ["exempt", "tmu", "not actual"]):
        return "exempt"
    if any(x in lower for x in ["approximately", "approx", "~", "about"]):
        return "estimated"
    if re.search(r"[\d,]+\s*mile", lower):
        return "exact"
    return "unknown"


def parse_transmission(text: str) -> str:
    """Normalize transmission string to 'manual' | 'automatic' | 'dct' | 'unknown'."""
    lower = (text or "").lower()
    if any(x in lower for x in ["manual", "6-speed", "5-speed", "4-speed", "gearbox", "stick"]):
        return "manual"
    if any(x in lower for x in ["dct", "pdk", "s-tronic", "dsg", "dual-clutch", "dual clutch"]):
        return "dct"
    if any(x in lower for x in ["automatic", "auto", "slushbox", "torque converter"]):
        return "automatic"
    return "unknown"


def parse_state(text: str) -> str | None:
    """Extract 2-letter US state code from location strings like 'Los Angeles, CA'."""
    if not text:
        return None
    match = re.search(r"\b([A-Z]{2})\b", text)
    return match.group(1) if match else None


def parse_date(text: str) -> str | None:
    """Parse date string to YYYY-MM-DD. Returns None on failure."""
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _safe_int(val) -> int | None:
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def run_incremental_scrape(max_pages: int = None) -> dict:
    """
    Main entry point for scheduled scrape runs.
    Fetches new completed listings and upserts to DB.
    Returns summary stats dict.
    """
    max_pages = max_pages or settings.SCRAPE_MAX_PAGES_PER_RUN
    session = next(get_session())
    run = crud.create_scrape_run(session, source="bat")

    pages_fetched = 0
    records_seen = 0
    records_new = 0
    stop_early = False

    try:
        for page in range(1, max_pages + 1):
            logger.info(f"BaT scraper: fetching page {page}")
            raw_listings = fetch_page(page)

            if not raw_listings:
                logger.info(f"BaT scraper: empty page {page}, stopping")
                break

            for raw in raw_listings:
                source_id = str(raw.get("id") or "")
                records_seen += 1

                # Check if already in raw_listings table (incremental stop)
                if crud.raw_listing_exists(session, "bat", source_id):
                    logger.info(f"BaT scraper: found existing source_id {source_id}, stopping early")
                    stop_early = True
                    break

                # Store raw first
                crud.upsert_raw_listing(session, source="bat", source_id=source_id, raw_json=raw)

                # Parse and normalize
                parsed = parse_listing(raw)
                normalized = normalize_listing(parsed)

                # Upsert to auctions
                is_new = crud.upsert_auction(session, normalized)
                if is_new:
                    records_new += 1

            pages_fetched += 1
            if stop_early:
                break

            time.sleep(settings.SCRAPE_DELAY_SECONDS)

        crud.finish_scrape_run(
            session, run.id,
            status="success",
            pages_fetched=pages_fetched,
            records_seen=records_seen,
            records_new=records_new,
        )
        logger.info(f"BaT scrape complete: {records_new} new records from {pages_fetched} pages")

    except Exception as e:
        logger.error(f"BaT scrape failed: {e}", exc_info=True)
        crud.finish_scrape_run(session, run.id, status="failed", error_message=str(e))
        raise
    finally:
        session.close()

    return {"pages_fetched": pages_fetched, "records_new": records_new}
```

### 4.2 Cars & Bids Scraper Architecture

**Source behavior:**
C&B is server-rendered HTML. Past auction results are paginated at /past-auctions/.
Each card on the listing page contains most fields needed. Individual listing pages
have more detail (full description, all images) but are not needed for MVP — the
card data is sufficient.

**Fragility note:** C&B HTML selectors WILL change. Any time C&B deploys a frontend
update, the scraper may break silently. Mitigation: save fixture HTML in tests/fixtures/
and test parse_card() against it. Monitor scrape_runs table for records_new=0 anomalies.

```python
# scraper/carsandbids.py

import httpx
import re
import time
import logging
from bs4 import BeautifulSoup, Tag
from datetime import datetime

from db.session import get_session
from db import crud
from scraper.normalizer import normalize_listing
from config import settings

logger = logging.getLogger(__name__)

CNB_BASE = "https://carsandbids.com"
CNB_PAST_URL = f"{CNB_BASE}/past-auctions/"

HEADERS = {
    "User-Agent": settings.USER_AGENT,
    "Accept": "text/html,application/xhtml+xml",
}


def fetch_page(page: int) -> str:
    """Fetch raw HTML of a past auctions page. Returns HTML string."""
    resp = httpx.get(
        CNB_PAST_URL,
        params={"page": page},
        headers=HEADERS,
        timeout=20,
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text


def parse_page(html: str) -> list[dict]:
    """Parse all auction cards from a past auctions HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("ul.auction-list li.auction-item")
    # Fallback selector if primary fails
    if not cards:
        cards = soup.select(".auction-card")
    return [parse_card(c) for c in cards if c]


def parse_card(card: Tag) -> dict:
    """
    Extract fields from a single auction card element.
    Returns dict with fields matching auctions table columns.
    Safe: all field extractions catch AttributeError and return None.
    """
    def text(selector: str) -> str | None:
        el = card.select_one(selector)
        return el.get_text(strip=True) if el else None

    def attr(selector: str, attribute: str) -> str | None:
        el = card.select_one(selector)
        return el.get(attribute) if el else None

    source_id = card.get("data-id") or card.get("data-auction-id") or ""
    href = attr("a.title-link", "href") or attr("h2 a", "href") or ""
    title_text = text("h2") or text(".title") or ""

    # Sale price: look for .bid-value or .sold-price
    price_raw = text(".bid-value") or text(".sold-price") or text(".final-bid") or ""
    sale_price_cents = _parse_price_to_cents(price_raw)

    # Sold status: check for "sold" class or badge
    classes = " ".join(card.get("class", []))
    sold_badge = text(".badge-sold") or text(".result-sold") or ""
    sold = "sold" in classes.lower() or "sold" in sold_badge.lower()

    # Date
    time_el = card.select_one("time")
    date_str = time_el.get("datetime") if time_el else None

    # Stats
    bid_count_raw = text(".bid-count") or text(".num-bids") or ""
    bid_count = _extract_int(bid_count_raw)

    # Thumbnail image
    img_el = card.select_one("img.main-image") or card.select_one("img")
    img_url = img_el.get("src") or img_el.get("data-src") if img_el else None

    return {
        "source": "carsandbids",
        "source_id": source_id.strip(),
        "source_url": (CNB_BASE + href) if href.startswith("/") else href,
        "title": title_text or None,
        "sale_price": sale_price_cents,
        "sold": sold,
        "bid_count": bid_count,
        "auction_end_date": _parse_date(date_str),
        "image_urls": [img_url] if img_url else [],
        "image_count": 1 if img_url else 0,
    }


def _parse_price_to_cents(text: str) -> int | None:
    """Convert '$52,500' or '52500' to 5250000 (cents)."""
    if not text:
        return None
    digits = re.sub(r"[^\d]", "", text)
    if not digits:
        return None
    return int(digits) * 100  # assumes price is in dollars


def _parse_date(text: str) -> str | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _extract_int(text: str) -> int | None:
    match = re.search(r"\d+", text or "")
    return int(match.group()) if match else None


def run_incremental_scrape(max_pages: int = None) -> dict:
    """Main entry point. Mirrors bat.run_incremental_scrape() structure exactly."""
    max_pages = max_pages or settings.SCRAPE_MAX_PAGES_PER_RUN
    session = next(get_session())
    run = crud.create_scrape_run(session, source="carsandbids")

    pages_fetched = records_seen = records_new = 0
    stop_early = False

    try:
        for page in range(1, max_pages + 1):
            logger.info(f"C&B scraper: fetching page {page}")
            html = fetch_page(page)
            listings = parse_page(html)

            if not listings:
                logger.info(f"C&B scraper: empty page {page}, stopping")
                break

            for listing in listings:
                source_id = listing.get("source_id", "")
                records_seen += 1

                if crud.raw_listing_exists(session, "carsandbids", source_id):
                    stop_early = True
                    break

                crud.upsert_raw_listing(
                    session, source="carsandbids",
                    source_id=source_id, raw_html=html
                )
                normalized = normalize_listing(listing)
                is_new = crud.upsert_auction(session, normalized)
                if is_new:
                    records_new += 1

            pages_fetched += 1
            if stop_early:
                break
            time.sleep(settings.SCRAPE_DELAY_SECONDS)

        crud.finish_scrape_run(
            session, run.id, status="success",
            pages_fetched=pages_fetched,
            records_seen=records_seen,
            records_new=records_new,
        )

    except Exception as e:
        logger.error(f"C&B scrape failed: {e}", exc_info=True)
        crud.finish_scrape_run(session, run.id, status="failed", error_message=str(e))
        raise
    finally:
        session.close()

    return {"pages_fetched": pages_fetched, "records_new": records_new}
```

### 4.3 Normalizer

```python
# scraper/normalizer.py

import re
from typing import Any

MAKE_ALIASES: dict[str, str] = {
    "vw": "volkswagen",
    "chevy": "chevrolet",
    "cheverolet": "chevrolet",
    "merc": "mercedes-benz",
    "mercedes": "mercedes-benz",
    "benz": "mercedes-benz",
    "alfa": "alfa romeo",
    "aston": "aston martin",
    "rolls": "rolls-royce",
    "lambo": "lamborghini",
    "ferrari": "ferrari",
    "porche": "porsche",
    "bmw": "bmw",
    "subaru": "subaru",
    "subi": "subaru",
}

MOD_KEYWORDS: list[str] = [
    "turbo", "supercharged", "swap", "swapped", "cammed", "built",
    "coilovers", "lowered", "widebody", "wide body", "roll cage",
    "roll bar", "big brake", "bbk", "forged", "sleeved", "bored",
    "stroker", "intake", "intercooler", "downpipe", "exhaust",
    "tune", "tuned", "remap", "ecu", "standalone", "megasquirt",
    "aftermarket wheels", "rims", "spacers", "flares",
]

RUST_KEYWORDS = ["rust", "rusty", "surface rust", "frame rust", "rocker rust"]
ACCIDENT_KEYWORDS = ["accident", "collision", "structural", "airbag deployed",
                     "carfax shows", "autocheck shows", "prior damage"]
REPAINT_KEYWORDS = ["repaint", "resprayed", "respray", "overspray", "color change"]


def normalize_listing(data: dict) -> dict:
    """
    Apply all normalization rules to a parsed listing dict.
    Returns a new dict safe to write to the auctions table.
    Mutates nothing in-place.
    """
    result = dict(data)

    result["make"] = _normalize_make(data.get("make"))
    result["model"] = _normalize_model(data.get("model"))
    result["transmission"] = _normalize_transmission(data.get("transmission"))

    description = (data.get("description_text") or "") + " " + (data.get("title") or "")
    result["is_modified"] = _detect_modifications(description)
    result["mod_summary"] = _extract_mod_keywords(description)
    result["rust_noted"] = _contains_any(description, RUST_KEYWORDS)
    result["accident_noted"] = _contains_any(description, ACCIDENT_KEYWORDS)
    result["repaint_noted"] = _contains_any(description, REPAINT_KEYWORDS)

    return result


def _normalize_make(make: str | None) -> str | None:
    if not make:
        return None
    cleaned = make.lower().strip()
    return MAKE_ALIASES.get(cleaned, cleaned)


def _normalize_model(model: str | None) -> str | None:
    if not model:
        return None
    return model.lower().strip()


def _normalize_transmission(trans: str | None) -> str:
    if not trans:
        return "unknown"
    lower = trans.lower()
    if any(x in lower for x in ["manual", "stick", "row your own", "gated"]):
        return "manual"
    if any(x in lower for x in ["pdk", "dct", "dsg", "s-tronic", "dual-clutch"]):
        return "dct"
    if any(x in lower for x in ["automatic", "auto", "slushbox"]):
        return "automatic"
    return "unknown"


def _detect_modifications(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in MOD_KEYWORDS)


def _extract_mod_keywords(text: str) -> list[str]:
    lower = text.lower()
    return [kw for kw in MOD_KEYWORDS if kw in lower]


def _contains_any(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)
```

---

## 5. AGGREGATION PIPELINE

### 5.1 rebuild_price_snapshots

This task runs nightly after both scrapers complete. It:
1. Queries all sold auctions grouped by make/model
2. For each significant make/model combo (10+ records), computes price stats
3. Computes for each window: 90 days, 180 days, 365 days, all-time
4. Computes for each condition: stock only, modified only, combined
5. Truncates price_snapshots and inserts fresh rows

```python
# tasks/aggregate.py

import logging
from sqlalchemy import text
from db.session import get_session

logger = logging.getLogger(__name__)

WINDOWS = [90, 180, 365, 0]  # 0 = all-time
CONDITIONS = ["stock", "modified", None]  # None = all

AGGREGATION_SQL = """
INSERT INTO price_snapshots (
    make, model, year_min, year_max, transmission, condition, window_days,
    sample_size, avg_price, median_price, min_price, max_price,
    p10_price, p25_price, p75_price, p90_price, stddev_price, sell_through
)
SELECT
    make,
    model,
    MIN(year) as year_min,
    MAX(year) as year_max,
    NULL as transmission,
    :condition_label as condition,
    :window_days as window_days,
    COUNT(*) FILTER (WHERE sold = TRUE) as sample_size,
    AVG(sale_price) FILTER (WHERE sold = TRUE) as avg_price,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price)
        FILTER (WHERE sold = TRUE) as median_price,
    MIN(sale_price) FILTER (WHERE sold = TRUE) as min_price,
    MAX(sale_price) FILTER (WHERE sold = TRUE) as max_price,
    percentile_cont(0.1) WITHIN GROUP (ORDER BY sale_price)
        FILTER (WHERE sold = TRUE) as p10_price,
    percentile_cont(0.25) WITHIN GROUP (ORDER BY sale_price)
        FILTER (WHERE sold = TRUE) as p25_price,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY sale_price)
        FILTER (WHERE sold = TRUE) as p75_price,
    percentile_cont(0.9) WITHIN GROUP (ORDER BY sale_price)
        FILTER (WHERE sold = TRUE) as p90_price,
    STDDEV(sale_price) FILTER (WHERE sold = TRUE) as stddev_price,
    COUNT(*) FILTER (WHERE sold = TRUE)::FLOAT /
        NULLIF(COUNT(*), 0) as sell_through
FROM auctions
WHERE
    make IS NOT NULL
    AND model IS NOT NULL
    AND (:window_days = 0 OR
         auction_end_date >= CURRENT_DATE - INTERVAL '1 day' * :window_days)
    AND (:condition_label IS NULL OR
         (:condition_label = 'stock' AND is_modified = FALSE) OR
         (:condition_label = 'modified' AND is_modified = TRUE))
GROUP BY make, model
HAVING COUNT(*) FILTER (WHERE sold = TRUE) >= 3
ON CONFLICT (make, model, year_min, year_max, transmission, condition, window_days)
DO UPDATE SET
    sample_size  = EXCLUDED.sample_size,
    avg_price    = EXCLUDED.avg_price,
    median_price = EXCLUDED.median_price,
    min_price    = EXCLUDED.min_price,
    max_price    = EXCLUDED.max_price,
    p10_price    = EXCLUDED.p10_price,
    p25_price    = EXCLUDED.p25_price,
    p75_price    = EXCLUDED.p75_price,
    p90_price    = EXCLUDED.p90_price,
    stddev_price = EXCLUDED.stddev_price,
    sell_through = EXCLUDED.sell_through,
    computed_at  = NOW()
"""


def rebuild_price_snapshots():
    """Rebuild all price_snapshots combinations. Runs nightly."""
    session = next(get_session())
    total_rows = 0
    try:
        for window in WINDOWS:
            for condition in CONDITIONS:
                logger.info(f"Aggregating: window={window}d condition={condition}")
                result = session.execute(
                    text(AGGREGATION_SQL),
                    {"window_days": window, "condition_label": condition}
                )
                session.commit()
                total_rows += result.rowcount
                logger.info(f"  → {result.rowcount} snapshot rows upserted")

        logger.info(f"Aggregation complete: {total_rows} total rows")
    except Exception as e:
        session.rollback()
        logger.error(f"Aggregation failed: {e}", exc_info=True)
        raise
    finally:
        session.close()
```

---

## 6. API SPECIFICATION

### 6.1 FastAPI Application Setup

```python
# api/main.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from api.routers import auctions, pricing, vehicles, health
from api.middleware import RateLimitMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API starting up")
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Enthusiast Car Auction Price Intelligence API",
    description="Historical pricing data from BaT and Cars & Bids.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/v1")
app.include_router(auctions.router, prefix="/v1")
app.include_router(pricing.router, prefix="/v1")
app.include_router(vehicles.router, prefix="/v1")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred."}}
    )
```

### 6.2 Pydantic Schemas

```python
# api/schemas.py

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date


# ── Request query param models ────────────────────────────────────

class AuctionQuery(BaseModel):
    make: Optional[str] = None
    model: Optional[str] = None
    year_min: Optional[int] = Field(None, ge=1950, le=2030)
    year_max: Optional[int] = Field(None, ge=1950, le=2030)
    miles_min: Optional[int] = Field(None, ge=0)
    miles_max: Optional[int] = Field(None, ge=0)
    transmission: Optional[str] = None
    sold: Optional[bool] = None
    no_reserve: Optional[bool] = None
    source: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    modified: Optional[bool] = None
    sort: str = Field("date_desc", pattern="^(date_desc|date_asc|price_asc|price_desc)$")
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)

    @field_validator("make", "model", mode="before")
    @classmethod
    def lowercase_strings(cls, v):
        return v.lower().strip() if v else v

    @field_validator("transmission", mode="before")
    @classmethod
    def valid_transmission(cls, v):
        allowed = {"manual", "automatic", "dct", "unknown"}
        if v and v.lower() not in allowed:
            raise ValueError(f"transmission must be one of: {allowed}")
        return v.lower() if v else v

    @field_validator("source", mode="before")
    @classmethod
    def valid_source(cls, v):
        allowed = {"bat", "carsandbids"}
        if v and v.lower() not in allowed:
            raise ValueError(f"source must be 'bat' or 'carsandbids'")
        return v.lower() if v else v


class PricingQuery(BaseModel):
    make: str
    model: str
    year_min: Optional[int] = Field(None, ge=1950, le=2030)
    year_max: Optional[int] = Field(None, ge=1950, le=2030)
    transmission: Optional[str] = None
    modified: Optional[bool] = None
    window_days: int = Field(365, ge=0)
    miles_max: Optional[int] = Field(None, ge=0)

    @field_validator("make", "model", mode="before")
    @classmethod
    def lowercase_strings(cls, v):
        return v.lower().strip() if v else v

    @field_validator("window_days", mode="before")
    @classmethod
    def valid_window(cls, v):
        if v not in (0, 90, 180, 365):
            raise ValueError("window_days must be 0, 90, 180, or 365")
        return v


# ── Response models ───────────────────────────────────────────────

class AuctionResult(BaseModel):
    id: int
    source: str
    source_url: str
    year: Optional[int]
    make: Optional[str]
    model: Optional[str]
    trim: Optional[str]
    mileage: Optional[int]
    mileage_confidence: Optional[str]
    transmission: Optional[str]
    color_exterior: Optional[str]
    sale_price: Optional[int]  # USD cents
    sold: bool
    no_reserve: bool
    bid_count: Optional[int]
    auction_end_date: Optional[date]
    is_modified: bool
    has_service_records: Optional[bool]
    location_state: Optional[str]
    image_urls: Optional[list[str]]


class AuctionsResponse(BaseModel):
    total: int
    limit: int
    offset: int
    results: list[AuctionResult]


class MileageBand(BaseModel):
    band: str
    median_price: Optional[int]
    avg_price: Optional[int]
    sample_size: int


class TrendPoint(BaseModel):
    period: str          # "2024-Q1"
    median_price: Optional[int]
    avg_price: Optional[int]
    sample_size: int


class PricingSummary(BaseModel):
    sample_size: int
    avg_price: Optional[int]
    median_price: Optional[int]
    min_price: Optional[int]
    max_price: Optional[int]
    p25_price: Optional[int]
    p75_price: Optional[int]
    p10_price: Optional[int]
    p90_price: Optional[int]
    sell_through_rate: Optional[float]


class PricingResponse(BaseModel):
    make: str
    model: str
    year_range: Optional[str]
    filters: dict
    summary: PricingSummary
    trend: list[TrendPoint]
    by_mileage_band: list[MileageBand]


class VehicleResult(BaseModel):
    id: int
    make: str
    model: str
    submodel: Optional[str]
    generation: Optional[str]
    year_min: int
    year_max: Optional[int]
    body_style: Optional[str]
    drivetrain: Optional[str]
    engine: Optional[str]


class ErrorResponse(BaseModel):
    error: dict  # {code, message, field?}
```

### 6.3 Pricing Router (Most Complex Endpoint)

```python
# api/routers/pricing.py

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text

from db.session import get_session
from api.schemas import PricingQuery, PricingResponse
from api.cache import cache_response

router = APIRouter()


@router.get("/pricing", response_model=PricingResponse)
@cache_response(ttl=3600)
async def get_pricing(
    make: str = Query(..., description="Vehicle make e.g. 'mazda'"),
    model: str = Query(..., description="Vehicle model e.g. 'miata'"),
    year_min: int = Query(None, ge=1950, le=2030),
    year_max: int = Query(None, ge=1950, le=2030),
    transmission: str = Query(None),
    modified: bool = Query(None),
    window_days: int = Query(365),
    miles_max: int = Query(None),
    db: Session = Depends(get_session),
):
    # Normalize inputs
    make = make.lower().strip()
    model = model.lower().strip()

    # Build filter conditions
    filters = {"make": make, "model": model}
    conditions = ["make = :make", "model = :model", "sold = TRUE"]
    params = {"make": make, "model": model}

    if year_min:
        conditions.append("year >= :year_min")
        params["year_min"] = year_min
    if year_max:
        conditions.append("year <= :year_max")
        params["year_max"] = year_max
    if transmission:
        conditions.append("transmission = :transmission")
        params["transmission"] = transmission.lower()
        filters["transmission"] = transmission
    if modified is not None:
        conditions.append("is_modified = :modified")
        params["modified"] = modified
        filters["modified"] = modified
    if window_days and window_days > 0:
        conditions.append(
            "auction_end_date >= CURRENT_DATE - INTERVAL '1 day' * :window_days"
        )
        params["window_days"] = window_days
        filters["window_days"] = window_days
    else:
        filters["window_days"] = 0
    if miles_max:
        conditions.append("mileage <= :miles_max")
        params["miles_max"] = miles_max

    where_clause = " AND ".join(conditions)

    # Main stats query
    stats_sql = f"""
    SELECT
        COUNT(*) as sample_size,
        AVG(sale_price) as avg_price,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) as median_price,
        MIN(sale_price) as min_price,
        MAX(sale_price) as max_price,
        percentile_cont(0.1) WITHIN GROUP (ORDER BY sale_price) as p10_price,
        percentile_cont(0.25) WITHIN GROUP (ORDER BY sale_price) as p25_price,
        percentile_cont(0.75) WITHIN GROUP (ORDER BY sale_price) as p75_price,
        percentile_cont(0.9) WITHIN GROUP (ORDER BY sale_price) as p90_price,
        STDDEV(sale_price) as stddev_price
    FROM auctions
    WHERE {where_clause} AND sale_price IS NOT NULL
    """

    # Sell-through query (all records, not just sold)
    all_conditions = [c for c in conditions if "sold = TRUE" not in c]
    all_where = " AND ".join(all_conditions) if all_conditions else "TRUE"
    sell_through_sql = f"""
    SELECT
        COUNT(*) FILTER (WHERE sold = TRUE)::FLOAT /
        NULLIF(COUNT(*), 0) as sell_through
    FROM auctions
    WHERE {all_where}
    """

    stats_row = db.execute(text(stats_sql), params).fetchone()

    if not stats_row or stats_row.sample_size < 3:
        raise HTTPException(
            status_code=404,
            detail={"code": "NOT_FOUND",
                    "message": f"Insufficient data for {make} {model} with the given filters. Try broader parameters."}
        )

    sell_through_row = db.execute(text(sell_through_sql), params).fetchone()

    # Trend query — group by quarter
    trend_sql = f"""
    SELECT
        EXTRACT(YEAR FROM auction_end_date) as yr,
        EXTRACT(QUARTER FROM auction_end_date) as qtr,
        COUNT(*) as sample_size,
        AVG(sale_price) as avg_price,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) as median_price
    FROM auctions
    WHERE {where_clause} AND sale_price IS NOT NULL
    GROUP BY yr, qtr
    ORDER BY yr ASC, qtr ASC
    """
    trend_rows = db.execute(text(trend_sql), params).fetchall()

    # Mileage band query
    mileage_sql = f"""
    SELECT
        CASE
            WHEN mileage < 30000  THEN '0-30k'
            WHEN mileage < 60000  THEN '30k-60k'
            WHEN mileage < 100000 THEN '60k-100k'
            ELSE '100k+'
        END as band,
        CASE
            WHEN mileage < 30000  THEN 1
            WHEN mileage < 60000  THEN 2
            WHEN mileage < 100000 THEN 3
            ELSE 4
        END as band_order,
        COUNT(*) as sample_size,
        AVG(sale_price) as avg_price,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY sale_price) as median_price
    FROM auctions
    WHERE {where_clause} AND sale_price IS NOT NULL AND mileage IS NOT NULL
    GROUP BY band, band_order
    ORDER BY band_order
    """
    mileage_rows = db.execute(text(mileage_sql), params).fetchall()

    # Year range
    year_range = None
    if year_min and year_max:
        year_range = f"{year_min}–{year_max}"
    elif year_min:
        year_range = f"{year_min}+"
    elif year_max:
        year_range = f"up to {year_max}"

    return PricingResponse(
        make=make.title(),
        model=model.upper() if len(model) <= 4 else model.title(),
        year_range=year_range,
        filters=filters,
        summary=dict(
            sample_size=stats_row.sample_size,
            avg_price=int(stats_row.avg_price) if stats_row.avg_price else None,
            median_price=int(stats_row.median_price) if stats_row.median_price else None,
            min_price=stats_row.min_price,
            max_price=stats_row.max_price,
            p10_price=int(stats_row.p10_price) if stats_row.p10_price else None,
            p25_price=int(stats_row.p25_price) if stats_row.p25_price else None,
            p75_price=int(stats_row.p75_price) if stats_row.p75_price else None,
            p90_price=int(stats_row.p90_price) if stats_row.p90_price else None,
            sell_through_rate=float(sell_through_row.sell_through) if sell_through_row.sell_through else None,
        ),
        trend=[
            dict(
                period=f"{int(r.yr)}-Q{int(r.qtr)}",
                median_price=int(r.median_price) if r.median_price else None,
                avg_price=int(r.avg_price) if r.avg_price else None,
                sample_size=r.sample_size,
            )
            for r in trend_rows
        ],
        by_mileage_band=[
            dict(
                band=r.band,
                median_price=int(r.median_price) if r.median_price else None,
                avg_price=int(r.avg_price) if r.avg_price else None,
                sample_size=r.sample_size,
            )
            for r in mileage_rows
        ],
    )
```

---

## 7. KNOWN FAILURE POINTS AND MITIGATIONS

This section is for Kiro. These are the places most likely to break, and what to do.

### 7.1 BaT API Endpoint Changes

**Risk:** BaT's WP REST endpoint is undocumented. It may change URL, response shape,
or disappear entirely. BaT has no obligation to maintain it.

**Detection:** scrape_runs.records_new = 0 for 3+ consecutive days while
scrape_runs.status = "success". This means scraper runs but finds nothing new.

**Mitigation:**
- Store fixture JSON from the endpoint in tests/fixtures/bat_page.json at launch
- Write test_bat_parser() against fixture, not live endpoint
- If endpoint breaks: fall back to scraping the HTML page and extracting
  window.BATData JSON object embedded in <script> tags using regex
- Fallback regex: `re.search(r'window\.BATData\s*=\s*({.+?});', html, re.DOTALL)`

### 7.2 C&B HTML Selector Changes

**Risk:** Any C&B frontend deployment can break BeautifulSoup CSS selectors.
This will cause parse_card() to return dicts with mostly None values.

**Detection:** scrape_runs.records_new / records_seen ratio drops below 0.1
(most records parsing as invalid). Alert if title is None for >50% of a page.

**Mitigation:**
- Save fixture HTML page in tests/fixtures/cnb_page.html at launch
- On selector failure, log full card HTML as a warning for manual inspection
- Build parse_card() with multiple fallback selectors (already done above)
- Add a post-parse validator: if title is None AND sale_price is None, discard record
  and log as parse_warning rather than writing garbage to DB

### 7.3 Price as Float Instead of Cents Integer

**Risk:** Any code path that does `int(price_string.replace("$","").replace(",",""))`
is correct. But if someone does `float("$52,500".replace(...))` the result stored
is 52500.0 which PostgreSQL will accept in an INTEGER column by truncation. Subtle.

**Detection:** Add a test: assert type(parsed["sale_price"]) is int

**Mitigation:** Always multiply by 100 immediately after parsing dollar string.
All internal code treats prices as cents. API responses divide by 100 for display
only if the API consumer wants dollars — or document that all prices are cents.

### 7.4 Duplicate Records From Race Conditions

**Risk:** If two scraper runs overlap (e.g., scheduler fires while previous run still
running), the same source_id could be inserted twice before the UNIQUE constraint check.

**Mitigation:** Use PostgreSQL upsert (ON CONFLICT DO NOTHING) in crud.upsert_auction().
Also add an APScheduler `max_instances=1` setting per job so overlapping runs are
prevented at the scheduler level.

### 7.5 Redis Cache Serving Stale Data After Schema Change

**Risk:** If you add a new field to the /pricing response, existing Redis cache
entries from before the deployment will return the old shape to API consumers for
up to TTL seconds (1 hour).

**Mitigation:** Include a cache_version integer in all cache keys.
Cache key pattern: `cache:v{CACHE_VERSION}:{endpoint}:{hash}`
Bump CACHE_VERSION in config whenever response schema changes. Old keys expire
naturally. This is a single env var change.

### 7.6 Aggregation Producing Wrong Medians on Small Samples

**Risk:** percentile_cont() on 3 records produces a "median" that is misleading.
A 911 with only 3 sales in a window could show median=$180,000 if one was a
high-spec GT3 RS.

**Mitigation:** Enforce minimum sample_size >= 3 in HAVING clause (already done).
Consider raising to 5 for production. Return NOT_FOUND rather than misleading stats.
Always include sample_size in every response so consumers can judge quality.

### 7.7 Make/Model Normalization Missing an Alias

**Risk:** "Chevy" and "Chevrolet" will be treated as separate makes. "VW" and
"Volkswagen" will be separate. This splits data and produces wrong stats.

**Mitigation:** MAKE_ALIASES dict in normalizer.py (already included). The bigger
risk is model name variations: "Miata" vs "MX-5" vs "MX-5 Miata". Handle this with
a MODEL_ALIASES dict too. Seed it with the most common variants at launch:
```python
MODEL_ALIASES = {
    "mx-5": "miata",
    "mx5": "miata",
    "miata mx-5": "miata",
    "boxster": "boxster",
    "gt-r": "gt-r",
    "gtr": "gt-r",
}
```

### 7.8 Database Connection Pool Exhaustion Under Load

**Risk:** Under sustained API load, FastAPI's async request handlers + SQLAlchemy
synchronous sessions can exhaust the connection pool.

**Mitigation for MVP:** Use SQLAlchemy connection pooling with pool_size=5,
max_overflow=10. This is fine for MVP load. If needed later: switch to
asyncpg + SQLAlchemy async sessions. Do not optimize prematurely.

### 7.9 RapidAPI Tier Bypass

**Risk:** A developer discovers the raw Railway URL (not the RapidAPI URL) and
makes unlimited free requests, bypassing billing entirely.

**Mitigation:** Add middleware that checks for X-RapidAPI-Proxy-Secret header
(set by RapidAPI gateway) on all non-/health endpoints. Reject requests without it.
Railway URL stays hidden — only the RapidAPI URL is documented publicly.

```python
# api/middleware.py
from fastapi import Request
from fastapi.responses import JSONResponse
from config import settings

class RapidAPIAuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            path = request.url.path
            # Skip auth for health endpoint
            if path != "/v1/health":
                proxy_secret = request.headers.get("X-RapidAPI-Proxy-Secret")
                if settings.RAPIDAPI_PROXY_SECRET and \
                   proxy_secret != settings.RAPIDAPI_PROXY_SECRET:
                    response = JSONResponse(
                        status_code=403,
                        content={"error": {"code": "FORBIDDEN",
                                          "message": "Access via RapidAPI only."}}
                    )
                    await response(scope, receive, send)
                    return
        await self.app(scope, receive, send)
```

---

## 8. CONFIGURATION

```python
# config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    REDIS_URL: str

    # Scraper
    SCRAPE_DELAY_SECONDS: float = 2.0
    SCRAPE_MAX_PAGES_PER_RUN: int = 50
    USER_AGENT: str = "Mozilla/5.0 (compatible; AuctionPriceBot/1.0)"

    # API
    CACHE_VERSION: int = 1
    CACHE_TTL_PRICING: int = 3600      # 1 hour
    CACHE_TTL_AUCTIONS: int = 900      # 15 minutes
    CACHE_TTL_VEHICLES: int = 86400    # 24 hours
    CACHE_TTL_HEALTH: int = 60         # 1 minute

    # Security
    RAPIDAPI_PROXY_SECRET: str = ""    # Set from RapidAPI dashboard

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 9. DOCKER AND DEPLOYMENT

### 9.1 Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim as base
WORKDIR /app

FROM base as builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM base as runtime
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

### 9.2 docker-compose.yml (Local Development)

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://postgres:password@postgres:5432/auction_db
      REDIS_URL: redis://redis:6379/0
      SCRAPE_DELAY_SECONDS: "1"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - .:/app    # hot reload in dev only

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: auction_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

---

## 10. SEED DATA

At launch, the vehicles table must be seeded with major enthusiast platforms.
This is what populates /v1/vehicles and what API consumers use to discover
valid make/model combinations.

```sql
INSERT INTO vehicles (make, model, submodel, generation, year_min, year_max,
                      body_style, drivetrain, engine) VALUES
-- Mazda Miata
('mazda', 'miata', NULL, 'NA', 1990, 1997, 'convertible', 'RWD', '1.6L / 1.8L inline-4'),
('mazda', 'miata', NULL, 'NB', 1999, 2005, 'convertible', 'RWD', '1.8L inline-4'),
('mazda', 'miata', NULL, 'NC', 2006, 2015, 'convertible', 'RWD', '2.0L inline-4'),
('mazda', 'miata', NULL, 'ND', 2016, NULL, 'convertible', 'RWD', '2.0L inline-4'),
-- Porsche 911
('porsche', '911', 'Carrera', '996', 1999, 2004, 'coupe', 'RWD', 'flat-6 3.4L/3.6L'),
('porsche', '911', 'Carrera', '997.1', 2005, 2008, 'coupe', 'RWD', 'flat-6 3.6L'),
('porsche', '911', 'Carrera', '997.2', 2009, 2012, 'coupe', 'RWD', 'flat-6 3.6L'),
('porsche', '911', 'Carrera', '991.1', 2012, 2016, 'coupe', 'RWD', 'flat-6 3.4L/3.8L'),
('porsche', '911', 'Carrera', '991.2', 2016, 2019, 'coupe', 'RWD', 'flat-6 3.0L Turbo'),
-- Toyota Supra
('toyota', 'supra', 'Turbo', 'A80', 1993, 1998, 'coupe', 'RWD', '2JZ-GTE 3.0L Twin Turbo'),
('toyota', 'supra', 'Naturally Aspirated', 'A80', 1993, 1998, 'coupe', 'RWD', '2JZ-GE 3.0L'),
-- BMW E30
('bmw', 'm3', NULL, 'E30', 1987, 1991, 'coupe', 'RWD', 'S14 2.3L inline-4'),
('bmw', '3 series', NULL, 'E30', 1982, 1991, 'coupe', 'RWD', 'inline-6'),
-- Honda S2000
('honda', 's2000', NULL, 'AP1', 1999, 2003, 'convertible', 'RWD', 'F20C 2.0L inline-4'),
('honda', 's2000', NULL, 'AP2', 2004, 2009, 'convertible', 'RWD', 'F22C 2.2L inline-4'),
-- Acura NSX
('acura', 'nsx', NULL, 'NA1', 1990, 2005, 'coupe', 'RWD', 'C30A/C32B 3.0L/3.2L V6'),
-- Nissan GT-R / Skyline
('nissan', 'gt-r', NULL, 'R34', 1999, 2002, 'coupe', 'AWD', 'RB26DETT 2.6L Twin Turbo'),
('nissan', 'gt-r', NULL, 'R35', 2007, NULL, 'coupe', 'AWD', 'VR38DETT 3.8L Twin Turbo'),
-- Subaru WRX/STI
('subaru', 'wrx', NULL, 'GD', 2002, 2007, 'sedan', 'AWD', 'EJ205 2.0L Turbo'),
('subaru', 'wrx sti', NULL, 'GD', 2004, 2007, 'sedan', 'AWD', 'EJ257 2.5L Turbo'),
('subaru', 'wrx sti', NULL, 'GR', 2008, 2014, 'sedan', 'AWD', 'EJ257 2.5L Turbo'),
-- Mitsubishi Lancer Evolution
('mitsubishi', 'lancer evolution', 'GSR', 'EVO X', 2008, 2015, 'sedan', 'AWD', '4B11T 2.0L Turbo'),
-- BMW M2/M3/M4
('bmw', 'm2', NULL, 'F87', 2016, 2021, 'coupe', 'RWD', 'S55 / N55 3.0L Turbo'),
('bmw', 'm3', NULL, 'E46', 2001, 2006, 'coupe', 'RWD', 'S54 3.2L inline-6'),
('bmw', 'm3', NULL, 'E92', 2008, 2013, 'coupe', 'RWD', 'S65 4.0L V8'),
-- Ferrari (sampler)
('ferrari', '308', NULL, NULL, 1975, 1985, 'coupe', 'RWD', 'F106 3.0L V8'),
('ferrari', '348', NULL, NULL, 1989, 1995, 'coupe', 'RWD', 'F119 3.4L V8'),
('ferrari', '360', NULL, NULL, 1999, 2005, 'coupe', 'RWD', 'F131 3.6L V8'),
-- Porsche Boxster/Cayman
('porsche', 'boxster', NULL, '986', 1997, 2004, 'convertible', 'RWD', 'flat-6 2.5L/2.7L'),
('porsche', 'cayman', NULL, '987', 2006, 2012, 'coupe', 'RWD', 'flat-6 2.7L/3.4L');
```

---

## 11. BUILD PHASES WITH DETAILED TASK BREAKDOWN

### Phase 0 — Foundation (Days 1–3)

Goal: A running local environment with schema + health endpoint.

- [ ] `git init auction-api && cd auction-api`
- [ ] Create directory structure as specified in Section 2.1
- [ ] Write requirements.txt with all dependencies
- [ ] Write docker-compose.yml (postgres + redis + api)
- [ ] Write config.py with Settings class
- [ ] Write db/session.py (SQLAlchemy engine + SessionLocal)
- [ ] Write all CREATE TABLE DDL in db/migrations/versions/001_initial.py via Alembic
- [ ] Run `alembic upgrade head` — verify all tables created
- [ ] Write SQLAlchemy ORM models in db/models.py mirroring schema exactly
- [ ] Write api/main.py skeleton with FastAPI app + CORS + /v1/health router
- [ ] Write api/routers/health.py returning {"status": "ok"}
- [ ] Verify: `docker compose up` → `curl localhost:8000/v1/health` returns 200
- [ ] Write .env.example with all variables documented
- [ ] Write README.md with `docker compose up` quickstart

Completion criteria: `docker compose up && curl localhost:8000/v1/health` returns JSON 200.

### Phase 1 — Scrapers (Days 4–8)

Goal: 1,000+ real records in the auctions table from each source.

- [ ] Write scraper/normalizer.py with MAKE_ALIASES, MODEL_ALIASES, MOD_KEYWORDS
- [ ] Write scraper/bat.py: fetch_page(), parse_listing(), parse_mileage(),
      parse_mileage_confidence(), parse_transmission(), parse_state(), parse_date()
- [ ] Save real BaT API response page to tests/fixtures/bat_page_1.json
- [ ] Write tests/test_bat_scraper.py: test parse_listing() against fixture
- [ ] Write db/crud.py: raw_listing_exists(), upsert_raw_listing(), upsert_auction(),
      create_scrape_run(), finish_scrape_run()
- [ ] Write scraper/bat.py: run_incremental_scrape() with full loop logic
- [ ] Manual run: `python -c "from scraper.bat import run_incremental_scrape; run_incremental_scrape(max_pages=5)"`
- [ ] Verify: 200+ rows in auctions table, spot-check 5 records by hand
- [ ] Write scraper/carsandbids.py: fetch_page(), parse_page(), parse_card()
- [ ] Save real C&B HTML page to tests/fixtures/cnb_page_1.html
- [ ] Write tests/test_cnb_scraper.py: test parse_card() against fixture
- [ ] Write scraper/carsandbids.py: run_incremental_scrape()
- [ ] Manual run: ingest 5 pages of C&B results
- [ ] Verify: 200+ C&B rows in auctions table, check source="carsandbids" records
- [ ] Test dedup: run same scrape again → 0 new records inserted
- [ ] Validate: all sale_price values are integers, not floats
      `SELECT sale_price, pg_typeof(sale_price) FROM auctions LIMIT 10;`

Completion criteria: 500+ rows in auctions, both sources, dedup confirmed.

### Phase 2 — API Endpoints (Days 9–14)

Goal: All four endpoints working with validation and caching.

- [ ] Write api/cache.py with cache_response() decorator
- [ ] Write db/crud.py: query_auctions(), get_pricing_stats(), list_vehicles()
- [ ] Write api/schemas.py: all request + response Pydantic models
- [ ] Write api/routers/vehicles.py: GET /v1/vehicles
- [ ] Run seed SQL for vehicles table (Section 10 above)
- [ ] Write api/routers/auctions.py: GET /v1/auctions with all filters + pagination
- [ ] Write api/routers/pricing.py: GET /v1/pricing with stats + trend + mileage bands
- [ ] Add all CREATE INDEX statements — verify with EXPLAIN ANALYZE
- [ ] Write api/middleware.py: RapidAPIAuthMiddleware (proxy secret check)
- [ ] Write tests/test_api.py: integration tests for all endpoints
      - /auctions: happy path, pagination, sold filter, source filter
      - /pricing: happy path, NOT_FOUND for sparse data, trend sorted correctly
      - /vehicles: known vehicle returns, unknown returns 404
      - /health: returns 200
- [ ] Test cache: hit /pricing twice, second response faster + identical
- [ ] Test error shapes: invalid year_min → 422 with "field" in response

Completion criteria: All tests pass, /pricing returns correct stats for seeded data.

### Phase 3 — Aggregation (Days 15–17)

Goal: price_snapshots table populated and /pricing reading from it.

- [ ] Write tasks/aggregate.py: rebuild_price_snapshots() with window loop
- [ ] Run manually: `python -c "from tasks.aggregate import rebuild_price_snapshots; rebuild_price_snapshots()"`
- [ ] Verify price_snapshots has rows for all major make/model combos
- [ ] Update /pricing router to check price_snapshots first, fall back to live SQL
- [ ] Write tasks/scheduler.py: APScheduler with bat, cnb, aggregate jobs
- [ ] Add max_instances=1 to each job to prevent overlap
- [ ] Test: scheduler fires correctly, check logs
- [ ] Update /health endpoint to include last scrape run timestamps

Completion criteria: price_snapshots populated, /pricing serves from cache table.

### Phase 4 — Deploy (Days 18–21)

Goal: Live on Railway, UptimeRobot monitoring, 6 months of historical data.

- [ ] Write Dockerfile (multi-stage)
- [ ] Push to GitHub
- [ ] Create Railway project
- [ ] Add PostgreSQL service in Railway, copy DATABASE_URL
- [ ] Add Redis service in Railway, copy REDIS_URL
- [ ] Add API service in Railway, point to GitHub repo
- [ ] Set all env vars in Railway dashboard
- [ ] Run `alembic upgrade head` via Railway shell on first deploy
- [ ] Run seed SQL for vehicles table via Railway shell
- [ ] Run historical backfill: set SCRAPE_MAX_PAGES_PER_RUN=200, trigger scrape manually
- [ ] Verify 10,000+ auction records in production DB
- [ ] Set up UptimeRobot: HTTP monitor on /v1/health, 5-minute interval, email alert
- [ ] Verify cron fires at scheduled time by checking scrape_runs table next morning

Completion criteria: API live, monitoring active, 10k+ records, cron confirmed.

### Phase 5 — Marketplace Launch (Days 22–28)

Goal: Listed on RapidAPI with docs, first users, first feedback.

- [ ] Create RapidAPI account at rapidapi.com/provider
- [ ] New API listing: "Enthusiast Car Auction Price Intelligence"
- [ ] Set base URL to Railway API URL
- [ ] Configure 4 tiers: Free / Developer / Pro / Business
- [ ] Set RAPIDAPI_PROXY_SECRET in Railway env vars
- [ ] Test: make a request via RapidAPI test console → confirm it hits Railway
- [ ] Write RapidAPI documentation for each endpoint
      - Description of every parameter
      - Example request + response for each
      - Note that prices are in USD cents
- [ ] Add 10 example queries in RapidAPI console:
      - Mazda Miata NA manual sold 365 days
      - Porsche 911 996 all time
      - Toyota Supra A80 turbo manual
      - Honda S2000 AP1
      - BMW E30 M3
      - Acura NSX NA1 sold only
      - Nissan GT-R R35
      - Subaru WRX STI GR
      - Ferrari 308 all time
      - BMW M3 E46 manual
- [ ] Post on r/webdev: "I built an API for enthusiast car auction pricing — feedback?"
- [ ] Post on r/cars: "BaT and Cars & Bids data as an API — anyone find this useful?"
- [ ] Post on IndieHackers: Show HN-style post with revenue model
- [ ] Post on HackerNews: Show HN: Enthusiast Car Auction Price API
- [ ] DM 5 developers in car-adjacent space on Twitter/X: offer free Pro trial
- [ ] Monitor RapidAPI analytics daily for first 2 weeks

---

## 12. COMPLETE TEST CHECKLIST

### Scraper Tests (run with pytest, fixture files only — never hit live sites in CI)

BaT Parser:
  [ ] parse_listing() with complete fixture record returns all expected fields
  [ ] parse_mileage("67,400 miles") returns 67400
  [ ] parse_mileage("~50k miles") returns 50000
  [ ] parse_mileage("Exempt") returns None
  [ ] parse_mileage("TMU") returns None
  [ ] parse_mileage("") returns None
  [ ] parse_mileage_confidence("67,400 miles") returns "exact"
  [ ] parse_mileage_confidence("~50k miles") returns "estimated"
  [ ] parse_mileage_confidence("Exempt") returns "exempt"
  [ ] parse_transmission("6-speed manual") returns "manual"
  [ ] parse_transmission("PDK") returns "dct"
  [ ] parse_transmission("Automatic") returns "automatic"
  [ ] parse_transmission("") returns "unknown"
  [ ] sale_price stored as integer, not float
  [ ] No-reserve flag correctly parsed
  [ ] Missing bid_count returns None not 0

C&B Parser:
  [ ] parse_card() with fixture HTML returns source="carsandbids"
  [ ] Unsold card (no "sold" class) returns sold=False
  [ ] Sold card returns sold=True
  [ ] source_id extracted correctly
  [ ] source_url correctly prepends base domain

Normalizer:
  [ ] "VW" normalized to "volkswagen"
  [ ] "Chevy" normalized to "chevrolet"
  [ ] "PORSCHE" normalized to "porsche"
  [ ] Description with "coilovers" sets is_modified=True
  [ ] Description with "rust" sets rust_noted=True
  [ ] Description with "accident" sets accident_noted=True
  [ ] Clean description sets all flags False

Deduplication:
  [ ] Running same scrape twice → second run inserts 0 rows
  [ ] upsert_auction() with existing source_id returns False (not new)
  [ ] upsert_auction() with new source_id returns True (is new)

### API Tests (integration tests against test database)

/v1/health:
  [ ] Returns 200 with {"status": "ok"}
  [ ] Returns last_ingestion timestamps from scrape_runs

/v1/vehicles:
  [ ] GET /vehicles?make=mazda&model=miata returns at least one result
  [ ] GET /vehicles?make=doesnotexist returns 404
  [ ] Response includes generation field

/v1/auctions:
  [ ] Returns paginated results with total, limit, offset fields
  [ ] sold=true returns only records where sold=TRUE
  [ ] sold=false returns only records where sold=FALSE
  [ ] make+model filter returns only matching records
  [ ] year_min + year_max filter works correctly
  [ ] transmission=manual returns only manual records
  [ ] source=bat returns only BaT records
  [ ] Pagination: offset=20 returns next 20 records
  [ ] sort=price_asc returns records in ascending price order
  [ ] Invalid transmission value returns 422
  [ ] Invalid year_min (e.g. 1800) returns 422

/v1/pricing:
  [ ] Returns summary, trend, by_mileage_band when data exists
  [ ] Returns 404 when fewer than 3 matching sold records
  [ ] trend array sorted chronologically (ascending by period)
  [ ] by_mileage_band returns 4 bands when data covers all ranges
  [ ] median_price != avg_price when distribution is skewed
  [ ] sell_through_rate between 0 and 1
  [ ] window_days=90 returns fewer records than window_days=365
  [ ] window_days=999 returns 422 (not in allowed values)
  [ ] modified=false excludes records where is_modified=TRUE
  [ ] sample_size matches COUNT of qualifying sold records

Cache:
  [ ] /pricing hit twice: second response has identical body
  [ ] After Redis flush: /pricing regenerates correctly
  [ ] Cache key includes make, model, year_min, year_max, window_days

Error format:
  [ ] All errors return {"error": {"code": ..., "message": ...}}
  [ ] 422 validation errors include field name
  [ ] 500 errors return INTERNAL_ERROR code without stack trace

---

## 13. DEPENDENCIES (requirements.txt)

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
sqlalchemy==2.0.30
alembic==1.13.1
psycopg2-binary==2.9.9
redis==5.0.4
httpx==0.27.0
beautifulsoup4==4.12.3
lxml==5.2.2
pydantic==2.7.1
pydantic-settings==2.3.0
apscheduler==3.10.4
pytest==8.2.2
pytest-asyncio==0.23.7
httpx==0.27.0
python-dotenv==1.0.1
```

---

## 14. POST-MVP ROADMAP (Out of Scope, Schema Supports These)

VIN Lookup Integration
  Cross-reference NHTSA vPIC API to auto-populate trim, engine, drivetrain
  for auctions where that data is missing. Free API, no key required.
  Endpoint: https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVin/{vin}

Comment Sentiment / Condition Signals
  BaT auctions have 20–200 comments containing condition judgments from
  expert buyers. NLP pass over comment text to extract: flood damage mentions,
  known issues, PPI results, restoration quality ratings. Adds "community_score"
  field to auction records.

Webhook Alert System
  Let subscribers define a watch: {make, model, year_min, year_max, max_price}.
  When a matching auction closes, POST to their webhook URL with the result.
  Requires: watches table, background checker job, webhook delivery queue.

Additional Sources
  Hemmings: classic car auctions with different demographic
  Collecting Cars: UK/European enthusiast auctions
  AutoHunter: budget enthusiast market
  Each needs its own scraper module following the same interface.

Price Estimate Endpoint
  GET /v1/estimate?make=porsche&model=911&year=1999&mileage=67000&transmission=manual
  Returns estimated sale price range based on comparable sold auctions.
  Uses weighted nearest-neighbor: most weight on year/mileage proximity.

Direct Stripe Billing
  For enterprise customers who want to bypass RapidAPI (higher volume, lower
  per-call cost). Build a thin subscription layer: API key table + Stripe webhooks
  + usage counting. Gives 100% revenue share vs 80% through RapidAPI.

---

Spec version: 2.0
Status: Ready for Kiro input
Rule: Start at Phase 0. Do not skip phases.
The scraper integrity and schema correctness are the foundation of everything.
```

---

That's the full spec. A few things to highlight as you hand this to Kiro:

**Section 7 (Known Failure Points)** is the most important section for Kiro to read before writing any code. It tells the tool exactly where things break and why, so it can build defensively from the start rather than patching later.

**The scraper architecture** uses the `run_incremental_scrape()` pattern with `scrape_runs` table logging — this means every run is auditable, and you can detect silent failures (scraper runs but finds nothing new) before you lose weeks of data.

**All prices are stored in cents as INTEGER** — this is the single most common bug in financial systems and it's called out in 3 places intentionally so Kiro doesn't accidentally drift it.

**The fixture file pattern** (saving real BaT JSON + C&B HTML to tests/fixtures/) means your CI tests never hit live sites, and when BaT or C&B change their markup you'll know immediately because the fixture tests will still pass while the live scraper fails.