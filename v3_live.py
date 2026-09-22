
#!/usr/bin/env python3
"""
===============================================================================
UPSTOX NSE 15-MINUTE CONTINUOUS BREAKOUT SCANNER
===============================================================================

Strategy characteristics
------------------------
1. Official Upstox NSE instrument master.
2. V2 Full Market Quote API for fast universe pre-screening.
3. Quote batching capped at 50 instruments.
4. Daily turnover pre-screen:
       cumulative_volume * LTP >= INR 50 crore
5. V3 15-minute candles for actual strategy analysis.
6. Historical 15-minute warm-up before indicators are calculated.
7. Latest COMPLETED 15-minute candle T only.
8. T is compared strictly against T-1.
9. Isolated turnover:
       T.volume * T.close >= INR 50 crore
10. Session VWAP anchored to 09:15.
11. EMA 9 / EMA 20.
12. RSI 14.
13. MACD 12/26/9.
14. Wilder-style ATR 14.
15. ATR-normalised price displacement.
16. Exact-period NIFTY 50 relative strength / weakness.
17. Continuous six-family scoring model.
18. LONG score >= 70 and shift >= +0.30%.
19. SHORT score >= 60 and shift <= -0.20%.
20. CSV append-only signal journal.
21. Duplicate signal suppression.
22. Auto-refreshing terminal dashboard.
23. Network/API failures do not terminate the scanner.
24. No hard-coded "scan only at 09:15/09:45" logic.

INSTALL
-------
    python3 -m pip install requests pandas numpy

RUN
---
    export UPSTOX_ACCESS_TOKEN="YOUR_ACCESS_TOKEN"
    python3 scanner.py

OPTIONAL ENVIRONMENT VARIABLES
------------------------------
    UPSTOX_POLL_SECONDS=30
    UPSTOX_MAX_WORKERS=16
    UPSTOX_QUOTE_BATCH_SIZE=50
    UPSTOX_HISTORICAL_DAYS=30
    UPSTOX_MIN_DAILY_TURNOVER=500000000
    UPSTOX_MIN_ISOLATED_TURNOVER=500000000
    UPSTOX_MIN_SHORT_SHIFT_PCT=0.20
    UPSTOX_MIN_LONG_SHIFT_PCT=0.30
    UPSTOX_SHORT_SCORE=60
    UPSTOX_LONG_SCORE=70
    UPSTOX_CSV_PATH=upstox_breakouts.csv

IMPORTANT
---------
The V2 intraday candle endpoint does not provide a 15-minute interval.
Therefore:

    V2 -> market quote pre-screen
    V3 -> 15-minute historical/intraday candles

This is intentional and follows the current Upstox API structure.
===============================================================================
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zoneinfo import ZoneInfo


# =============================================================================
# CONFIGURATION
# =============================================================================

IST = ZoneInfo("Asia/Kolkata")

UPSTOX_V2_BASE = "https://api.upstox.com/v2"
UPSTOX_V3_BASE = "https://api.upstox.com/v3"

# Official Upstox NSE CSV instrument master.
NSE_MASTER_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
)

NIFTY_50_INSTRUMENT_KEY = "NSE_INDEX|Nifty 50"

MARKET_OPEN = datetime_time(9, 15)
MARKET_CLOSE = datetime_time(15, 30)

DEFAULT_POLL_SECONDS = 30
DEFAULT_MAX_WORKERS = 16
DEFAULT_QUOTE_BATCH_SIZE = 50
DEFAULT_HISTORICAL_DAYS = 30

DEFAULT_MIN_DAILY_TURNOVER = 500_000_000.0
DEFAULT_MIN_ISOLATED_TURNOVER = 500_000_000.0

DEFAULT_MIN_SHORT_SHIFT = 0.0020
DEFAULT_MIN_LONG_SHIFT = 0.0030

DEFAULT_SHORT_SCORE = 60.0
DEFAULT_LONG_SCORE = 70.0

EMA_FAST_PERIOD = 9
EMA_SLOW_PERIOD = 20

RSI_PERIOD = 14
ATR_PERIOD = 14

MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9

RELATIVE_VOLUME_LOOKBACK = 20

EPSILON = 1e-12


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================

def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "").strip()

POLL_SECONDS = max(
    5,
    get_env_int(
        "UPSTOX_POLL_SECONDS",
        DEFAULT_POLL_SECONDS,
    ),
)

MAX_WORKERS = max(
    1,
    get_env_int(
        "UPSTOX_MAX_WORKERS",
        DEFAULT_MAX_WORKERS,
    ),
)

QUOTE_BATCH_SIZE = min(
    50,
    max(
        1,
        get_env_int(
            "UPSTOX_QUOTE_BATCH_SIZE",
            DEFAULT_QUOTE_BATCH_SIZE,
        ),
    ),
)

HISTORICAL_DAYS = max(
    7,
    get_env_int(
        "UPSTOX_HISTORICAL_DAYS",
        DEFAULT_HISTORICAL_DAYS,
    ),
)

MIN_DAILY_TURNOVER = get_env_float(
    "UPSTOX_MIN_DAILY_TURNOVER",
    DEFAULT_MIN_DAILY_TURNOVER,
)

MIN_ISOLATED_TURNOVER = get_env_float(
    "UPSTOX_MIN_ISOLATED_TURNOVER",
    DEFAULT_MIN_ISOLATED_TURNOVER,
)

MIN_SHORT_SHIFT = (
    get_env_float(
        "UPSTOX_MIN_SHORT_SHIFT_PCT",
        DEFAULT_MIN_SHORT_SHIFT * 100,
    )
    / 100.0
)

MIN_LONG_SHIFT = (
    get_env_float(
        "UPSTOX_MIN_LONG_SHIFT_PCT",
        DEFAULT_MIN_LONG_SHIFT * 100,
    )
    / 100.0
)

SHORT_SCORE_THRESHOLD = get_env_float(
    "UPSTOX_SHORT_SCORE",
    DEFAULT_SHORT_SCORE,
)

LONG_SCORE_THRESHOLD = get_env_float(
    "UPSTOX_LONG_SCORE",
    DEFAULT_LONG_SCORE,
)

CSV_PATH = Path(
    os.getenv(
        "UPSTOX_CSV_PATH",
        "upstox_breakouts.csv",
    )
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

LOGGER = logging.getLogger("upstox_scanner")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass(frozen=True)
class Instrument:
    """
    Canonical instrument representation used internally.
    """

    trading_symbol: str
    instrument_key: str
    name: str
    isin: str = ""


@dataclass(frozen=True)
class QuoteSnapshot:
    """
    Minimal quote representation required by the scanner.
    """

    instrument_key: str
    symbol: str
    ltp: float
    volume: float
    timestamp: str = ""


@dataclass(frozen=True)
class Signal:
    """
    Immutable strategy signal.
    """

    scan_timestamp: datetime
    candle_timestamp: datetime

    symbol: str
    instrument_key: str

    direction: str
    score: float

    shift_pct: float
    nifty_shift_pct: float
    relative_shift_pct: float

    close: float
    previous_close: float

    isolated_turnover: float
    daily_turnover: float

    vwap: float
    ema9: float
    ema20: float

    rsi: float
    macd: float
    macd_signal: float

    atr14: float
    atr_normalized_shift: float

    candle_range: float
    relative_volume: float

    family_vwap: float
    family_ema: float
    family_momentum: float
    family_candle: float
    family_relative: float
    family_volume: float


# =============================================================================
# HTTP CLIENT
# =============================================================================

def create_http_session(
    access_token: str,
) -> requests.Session:
    """
    Create a reusable HTTP session.

    GET-only retry policy:
        429
        500
        502
        503
        504
    """

    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.35,
        status_forcelist=(
            429,
            500,
            502,
            503,
            504,
        ),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=MAX_WORKERS + 4,
        pool_maxsize=MAX_WORKERS + 4,
    )

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}",
            "User-Agent": "Upstox-NSE-15m-Breakout-Scanner/2.0",
        }
    )

    return session


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def now_ist() -> datetime:
    return datetime.now(IST)


def chunks(
    sequence: Sequence[str],
    size: int,
) -> Iterable[list[str]]:
    for start in range(
        0,
        len(sequence),
        size,
    ):
        yield list(
            sequence[
                start:start + size
            ]
        )


def safe_float(
    value: Any,
    default: float = math.nan,
) -> float:
    try:
        if value is None:
            return default

        result = float(value)

        if math.isfinite(result):
            return result

        return default

    except (
        TypeError,
        ValueError,
    ):
        return default


def clip01(value: float) -> float:
    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def is_finite(*values: float) -> bool:
    return all(
        math.isfinite(value)
        for value in values
    )


def format_pct(
    value: float,
) -> str:
    if not math.isfinite(value):
        return "NA"

    return f"{value * 100:+.2f}%"


def format_inr(
    value: float,
) -> str:
    if not math.isfinite(value):
        return "NA"

    crore = value / 10_000_000.0

    return f"₹{crore:,.2f}Cr"


def market_is_open(
    current_time: datetime,
) -> bool:
    current = current_time.time()

    return (
        MARKET_OPEN
        <= current
        < MARKET_CLOSE
    )


# =============================================================================
# INSTRUMENT MASTER
# =============================================================================

def download_instrument_master(
    session: requests.Session,
) -> pd.DataFrame:
    """
    Download the official Upstox NSE CSV master.

    The server normally returns gzip-compressed CSV. The function also
    tolerates an uncompressed response.
    """

    response = session.get(
        NSE_MASTER_URL,
        timeout=(5, 30),
    )

    response.raise_for_status()

    raw_data = response.content

    try:
        raw_data = gzip.decompress(
            raw_data
        )
    except OSError:
        pass

    dataframe = pd.read_csv(
        io.BytesIO(raw_data),
        low_memory=False,
    )

    dataframe.columns = [
        str(column)
        .strip()
        .lower()
        .replace(
            " ",
            "_",
        )
        .replace(
            "-",
            "_",
        )
        for column in dataframe.columns
    ]

    if dataframe.empty:
        raise RuntimeError(
            "Upstox instrument master downloaded successfully "
            "but contains zero rows."
        )

    LOGGER.info(
        "Downloaded instrument master: %d rows.",
        len(dataframe),
    )

    LOGGER.debug(
        "Instrument-master columns: %s",
        list(dataframe.columns),
    )

    return dataframe


def normalize_instrument_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize known CSV/JSON naming differences.

    CSV commonly uses:
        tradingsymbol

    JSON commonly uses:
        trading_symbol
    """

    dataframe = dataframe.copy()

    rename_map: dict[str, str] = {}

    if (
        "trading_symbol" not in dataframe.columns
        and "tradingsymbol" in dataframe.columns
    ):
        rename_map["tradingsymbol"] = "trading_symbol"

    if (
        "instrument_key" not in dataframe.columns
        and "instrumentkey" in dataframe.columns
    ):
        rename_map["instrumentkey"] = "instrument_key"

    if (
        "exchange_token" not in dataframe.columns
        and "exchangetoken" in dataframe.columns
    ):
        rename_map["exchangetoken"] = "exchange_token"

    dataframe = dataframe.rename(
        columns=rename_map
    )

    return dataframe


def load_nse_equity_instruments(
    session: requests.Session,
) -> dict[str, Instrument]:
    """
    Load NSE cash-equity instruments.

    The parser deliberately does NOT require a specific combination of
    segment/instrument_type/security_type because the CSV schema can evolve.

    The instrument-key prefix is used as the primary NSE cash-equity filter.
    """

    dataframe = download_instrument_master(
        session
    )

    dataframe = normalize_instrument_columns(
        dataframe
    )

    required_columns = {
        "instrument_key",
        "trading_symbol",
        "name",
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        LOGGER.error(
            "Instrument master columns received: %s",
            list(dataframe.columns),
        )

        raise RuntimeError(
            "Instrument master missing required columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------------------
    # Clean strings.
    # -------------------------------------------------------------------------

    for column in (
        "instrument_key",
        "trading_symbol",
        "name",
    ):
        dataframe[column] = (
            dataframe[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    if "isin" in dataframe.columns:
        dataframe["isin"] = (
            dataframe["isin"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        dataframe["isin"] = ""

    # -------------------------------------------------------------------------
    # Primary filter: NSE_EQ instrument-key namespace.
    # -------------------------------------------------------------------------

    dataframe = dataframe[
        dataframe["instrument_key"]
        .str.startswith(
            "NSE_EQ|",
            na=False,
        )
    ]

    # -------------------------------------------------------------------------
    # If exchange exists, enforce NSE.
    # -------------------------------------------------------------------------

    if "exchange" in dataframe.columns:
        exchange_mask = (
            dataframe["exchange"]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq("NSE")
        )

        if exchange_mask.any():
            dataframe = dataframe[
                exchange_mask
            ]

    # -------------------------------------------------------------------------
    # If segment exists, prefer NSE_EQ.
    # -------------------------------------------------------------------------

    if "segment" in dataframe.columns:
        segment_mask = (
            dataframe["segment"]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq("NSE_EQ")
        )

        if segment_mask.any():
            dataframe = dataframe[
                segment_mask
            ]

    # -------------------------------------------------------------------------
    # Prefer normal securities when security_type is present.
    #
    # We only apply this filter when NORMAL rows actually exist.
    # This avoids accidentally producing an empty universe because a future
    # schema changes the terminology.
    # -------------------------------------------------------------------------

    if "security_type" in dataframe.columns:

        security_type = (
            dataframe["security_type"]
            .fillna("")
            .astype(str)
            .str.upper()
        )

        normal_mask = security_type.eq(
            "NORMAL"
        )

        if normal_mask.any():
            dataframe = dataframe[
                normal_mask
            ]

    # -------------------------------------------------------------------------
    # Remove malformed records.
    # -------------------------------------------------------------------------

    dataframe = dataframe[
        dataframe["instrument_key"].ne("")
        & dataframe["trading_symbol"].ne("")
        & dataframe["instrument_key"].ne("nan")
        & dataframe["trading_symbol"].ne("NAN")
    ]

    if dataframe.empty:
        raise RuntimeError(
            "No NSE_EQ instruments remained after instrument-master filtering."
        )

    # -------------------------------------------------------------------------
    # Construct canonical mapping.
    # -------------------------------------------------------------------------

    instruments: dict[str, Instrument] = {}

    for row in dataframe.itertuples(
        index=False
    ):
        symbol = str(
            getattr(
                row,
                "trading_symbol",
                "",
            )
        ).strip().upper()

        instrument_key = str(
            getattr(
                row,
                "instrument_key",
                "",
            )
        ).strip()

        name = str(
            getattr(
                row,
                "name",
                "",
            )
        ).strip()

        isin = str(
            getattr(
                row,
                "isin",
                "",
            )
        ).strip()

        if not symbol:
            continue

        if not instrument_key:
            continue

        if not instrument_key.startswith(
            "NSE_EQ|"
        ):
            continue

        # First valid record wins.
        instruments.setdefault(
            symbol,
            Instrument(
                trading_symbol=symbol,
                instrument_key=instrument_key,
                name=name,
                isin=isin,
            ),
        )

    if not instruments:
        raise RuntimeError(
            "Instrument mapping contains zero valid NSE equities."
        )

    LOGGER.info(
        "Loaded %d NSE equity instruments.",
        len(instruments),
    )

    LOGGER.info(
        "Sample symbols: %s",
        ", ".join(
            list(instruments.keys())[:12]
        ),
    )

    return instruments


# =============================================================================
# V2 FULL MARKET QUOTES
# =============================================================================

def fetch_market_quotes(
    session: requests.Session,
    instruments: Sequence[Instrument],
) -> dict[str, QuoteSnapshot]:
    """
    V2 full market quote.

    Returns:
        instrument_key -> QuoteSnapshot

    We use quote["instrument_token"] as the authoritative mapping back to
    the instrument master.
    """

    quotes: dict[
        str,
        QuoteSnapshot
    ] = {}

    instrument_keys = [
        instrument.instrument_key
        for instrument in instruments
    ]

    for batch in chunks(
        instrument_keys,
        QUOTE_BATCH_SIZE,
    ):

        params = {
            "instrument_key": ",".join(
                batch
            )
        }

        response = session.get(
            f"{UPSTOX_V2_BASE}/market-quote/quotes",
            params=params,
            timeout=(4, 15),
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") != "success":
            raise RuntimeError(
                "Full market quote API returned non-success response: "
                f"{payload}"
            )

        data = payload.get(
            "data",
            {},
        )

        if not isinstance(
            data,
            dict,
        ):
            continue

        for _, raw_quote in data.items():

            if not isinstance(
                raw_quote,
                dict,
            ):
                continue

            instrument_key = str(
                raw_quote.get(
                    "instrument_token",
                    "",
                )
            ).strip()

            symbol = str(
                raw_quote.get(
                    "symbol",
                    "",
                )
            ).strip().upper()

            ltp = safe_float(
                raw_quote.get(
                    "last_price"
                )
            )

            volume = safe_float(
                raw_quote.get(
                    "volume"
                )
            )

            timestamp = str(
                raw_quote.get(
                    "timestamp",
                    "",
                )
            )

            if not instrument_key:
                continue

            if not symbol:
                continue

            if not math.isfinite(ltp):
                continue

            if not math.isfinite(volume):
                continue

            quotes[instrument_key] = QuoteSnapshot(
                instrument_key=instrument_key,
                symbol=symbol,
                ltp=ltp,
                volume=volume,
                timestamp=timestamp,
            )

    return quotes


# =============================================================================
# DAILY TURNOVER PRE-SCREEN
# =============================================================================

def prescreen_instruments(
    instruments: Mapping[str, Instrument],
    quotes: Mapping[str, QuoteSnapshot],
) -> tuple[
    list[Instrument],
    dict[str, float],
]:
    """
    Apply:

        daily cumulative volume * current LTP >= ₹50 crore
    """

    candidates: list[Instrument] = []

    daily_turnover: dict[
        str,
        float
    ] = {}

    for instrument in instruments.values():

        quote = quotes.get(
            instrument.instrument_key
        )

        if quote is None:
            continue

        if quote.volume <= 0:
            continue

        if quote.ltp <= 0:
            continue

        turnover = (
            quote.volume
            * quote.ltp
        )

        daily_turnover[
            instrument.instrument_key
        ] = turnover

        if turnover >= MIN_DAILY_TURNOVER:
            candidates.append(
                instrument
            )

    return (
        candidates,
        daily_turnover,
    )


# =============================================================================
# CANDLE DATA PARSING
# =============================================================================

CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "oi",
]


def candles_to_dataframe(
    candles: Any,
) -> pd.DataFrame:
    """
    Convert Upstox candle arrays to a chronological dataframe.

    IMPORTANT:
    Upstox candle responses are returned newest -> oldest.

    The scanner explicitly reverses with iloc[::-1] before feature engineering.
    """

    if not isinstance(
        candles,
        list,
    ):
        return pd.DataFrame(
            columns=CANDLE_COLUMNS
        )

    rows: list[list[Any]] = []

    for candle in candles:

        if not isinstance(
            candle,
            (list, tuple),
        ):
            continue

        if len(candle) < 6:
            continue

        rows.append(
            [
                candle[0],
                candle[1],
                candle[2],
                candle[3],
                candle[4],
                candle[5],
                candle[6]
                if len(candle) >= 7
                else 0,
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=CANDLE_COLUMNS
        )

    dataframe = pd.DataFrame(
        rows,
        columns=CANDLE_COLUMNS,
    )

    # -------------------------------------------------------------------------
    # Parse timestamp.
    # -------------------------------------------------------------------------

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    # Normalize timezone to IST.
    if dataframe["timestamp"].dt.tz is None:
        dataframe["timestamp"] = (
            dataframe["timestamp"]
            .dt.tz_localize(IST)
        )
    else:
        dataframe["timestamp"] = (
            dataframe["timestamp"]
            .dt.tz_convert(IST)
        )

    # -------------------------------------------------------------------------
    # Numeric fields.
    # -------------------------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "oi",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    if dataframe.empty:
        return dataframe.reset_index(
            drop=True
        )

    # -------------------------------------------------------------------------
    # REQUIRED TEMPORAL ALIGNMENT:
    #
    # Upstox -> newest to oldest
    # Scanner -> oldest to newest
    #
    # Explicit iloc[::-1] is retained rather than relying only on sort_values.
    # -------------------------------------------------------------------------

    dataframe = dataframe.iloc[::-1].copy()

    # -------------------------------------------------------------------------
    # Deduplicate and make chronological order explicit.
    # -------------------------------------------------------------------------

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


# =============================================================================
# V3 15-MINUTE HISTORICAL CANDLES
# =============================================================================

def fetch_historical_15m(
    session: requests.Session,
    instrument_key: str,
    to_date: date,
    days: int,
) -> pd.DataFrame:
    """
    Fetch historical 15-minute candles.

    V3:
        /historical-candle/{instrument_key}/minutes/15/{to_date}/{from_date}

    Upstox currently limits <=15-minute historical intervals to one month,
    hence the default 30-calendar-day warm-up.
    """

    # Keep within one-month API history boundary.
    days = min(
        days,
        30,
    )

    from_date = (
        to_date
        - timedelta(
            days=days
        )
    )

    encoded_key = requests.utils.quote(
        instrument_key,
        safe="",
    )

    url = (
        f"{UPSTOX_V3_BASE}/historical-candle/"
        f"{encoded_key}/minutes/15/"
        f"{to_date.isoformat()}/"
        f"{from_date.isoformat()}"
    )

    response = session.get(
        url,
        timeout=(5, 20),
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(
            f"Historical candle API failure for "
            f"{instrument_key}: {payload}"
        )

    candles = (
        payload
        .get("data", {})
        .get("candles", [])
    )

    return candles_to_dataframe(
        candles
    )


# =============================================================================
# V3 15-MINUTE INTRADAY CANDLES
# =============================================================================

def fetch_intraday_15m(
    session: requests.Session,
    instrument_key: str,
) -> pd.DataFrame:
    """
    Fetch current trading-day 15-minute candles.
    """

    encoded_key = requests.utils.quote(
        instrument_key,
        safe="",
    )

    url = (
        f"{UPSTOX_V3_BASE}/historical-candle/"
        f"intraday/"
        f"{encoded_key}/minutes/15"
    )

    response = session.get(
        url,
        timeout=(4, 15),
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("status") != "success":
        raise RuntimeError(
            f"Intraday candle API failure for "
            f"{instrument_key}: {payload}"
        )

    candles = (
        payload
        .get("data", {})
        .get("candles", [])
    )

    return candles_to_dataframe(
        candles
    )


# =============================================================================
# HISTORICAL CACHE
# =============================================================================

class HistoricalCache:
    """
    Thread-safe in-memory cache.

    Historical warm-up is reused throughout the trading day.
    """

    def __init__(
        self,
        session: requests.Session,
        days: int,
    ) -> None:

        self._session = session
        self._days = days

        self._cache: dict[
            str,
            pd.DataFrame
        ] = {}

        self._cache_date: date | None = None

        self._lock = threading.Lock()

    def reset_for_date(
        self,
        current_date: date,
    ) -> None:

        with self._lock:

            if self._cache_date != current_date:

                self._cache.clear()

                self._cache_date = current_date

    def contains(
        self,
        instrument_key: str,
    ) -> bool:

        with self._lock:
            return instrument_key in self._cache

    def get(
        self,
        instrument_key: str,
    ) -> pd.DataFrame:

        with self._lock:

            dataframe = self._cache.get(
                instrument_key
            )

            if dataframe is None:
                return pd.DataFrame(
                    columns=CANDLE_COLUMNS
                )

            return dataframe

    def put(
        self,
        instrument_key: str,
        dataframe: pd.DataFrame,
    ) -> None:

        with self._lock:
            self._cache[
                instrument_key
            ] = dataframe

    def preload(
        self,
        instruments: Sequence[Instrument],
        current_date: date,
    ) -> None:

        self.reset_for_date(
            current_date
        )

        missing = [
            instrument
            for instrument in instruments
            if not self.contains(
                instrument.instrument_key
            )
        ]

        if not missing:
            return

        LOGGER.info(
            "Historical warm-up required for %d instruments.",
            len(missing),
        )

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix="history",
        ) as executor:

            futures = {
                executor.submit(
                    fetch_historical_15m,
                    self._session,
                    instrument.instrument_key,
                    current_date,
                    self._days,
                ): instrument
                for instrument in missing
            }

            for future in as_completed(
                futures
            ):

                instrument = futures[
                    future
                ]

                try:

                    dataframe = future.result()

                    if not dataframe.empty:

                        self.put(
                            instrument.instrument_key,
                            dataframe,
                        )

                except Exception as exc:

                    LOGGER.warning(
                        "Historical warm-up failed: "
                        "%s -> %s",
                        instrument.trading_symbol,
                        exc,
                    )


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_ema(
    close: pd.Series,
    period: int,
) -> pd.Series:

    return close.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_rsi(
    close: pd.Series,
    period: int = RSI_PERIOD,
) -> pd.Series:

    delta = close.diff()

    gains = delta.clip(
        lower=0.0
    )

    losses = (
        -delta.clip(
            upper=0.0
        )
    )

    average_gain = gains.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = losses.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        average_gain
        / average_loss.replace(
            0.0,
            np.nan,
        )
    )

    rsi = (
        100.0
        - (
            100.0
            / (1.0 + rs)
        )
    )

    # Handle no-loss / no-gain edge cases.
    rsi = rsi.where(
        average_loss > EPSILON,
        100.0,
    )

    rsi = rsi.where(
        average_gain > EPSILON,
        0.0,
    )

    return rsi


def calculate_atr(
    dataframe: pd.DataFrame,
    period: int = ATR_PERIOD,
) -> pd.Series:

    previous_close = (
        dataframe["close"]
        .shift(1)
    )

    true_ranges = pd.concat(
        [
            dataframe["high"]
            - dataframe["low"],

            (
                dataframe["high"]
                - previous_close
            ).abs(),

            (
                dataframe["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    )

    true_range = true_ranges.max(
        axis=1
    )

    # Wilder smoothing.
    return true_range.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_macd(
    close: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
]:

    fast = close.ewm(
        span=MACD_FAST_PERIOD,
        adjust=False,
        min_periods=MACD_FAST_PERIOD,
    ).mean()

    slow = close.ewm(
        span=MACD_SLOW_PERIOD,
        adjust=False,
        min_periods=MACD_SLOW_PERIOD,
    ).mean()

    macd = fast - slow

    signal = macd.ewm(
        span=MACD_SIGNAL_PERIOD,
        adjust=False,
        min_periods=MACD_SIGNAL_PERIOD,
    ).mean()

    histogram = (
        macd
        - signal
    )

    return (
        macd,
        signal,
        histogram,
    )


def add_indicators(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["ema9"] = calculate_ema(
        dataframe["close"],
        EMA_FAST_PERIOD,
    )

    dataframe["ema20"] = calculate_ema(
        dataframe["close"],
        EMA_SLOW_PERIOD,
    )

    dataframe["rsi14"] = calculate_rsi(
        dataframe["close"],
        RSI_PERIOD,
    )

    (
        dataframe["macd"],
        dataframe["macd_signal"],
        dataframe["macd_hist"],
    ) = calculate_macd(
        dataframe["close"]
    )

    dataframe["atr14"] = calculate_atr(
        dataframe,
        ATR_PERIOD,
    )

    # Median historical volume baseline.
    #
    # Shift(1) is critical:
    # T must NOT contribute to its own relative-volume baseline.
    dataframe["volume_median_20"] = (
        dataframe["volume"]
        .rolling(
            RELATIVE_VOLUME_LOOKBACK,
            min_periods=10,
        )
        .median()
        .shift(1)
    )

    dataframe["relative_volume"] = (
        dataframe["volume"]
        / dataframe["volume_median_20"].replace(
            0.0,
            np.nan,
        )
    )

    return dataframe


# =============================================================================
# SESSION VWAP
# =============================================================================

def add_session_vwap(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate true daily anchored VWAP.

    For each trading date:

        cumulative(TypicalPrice * Volume)
        --------------------------------
              cumulative Volume

    The first NSE 15-minute candle begins at 09:15, so VWAP naturally resets
    with the trading date.
    """

    dataframe = dataframe.copy()

    dataframe["trade_date"] = (
        dataframe["timestamp"]
        .dt.date
    )

    typical_price = (
        dataframe["high"]
        + dataframe["low"]
        + dataframe["close"]
    ) / 3.0

    dataframe["price_volume"] = (
        typical_price
        * dataframe["volume"]
    )

    dataframe["session_volume"] = (
        dataframe
        .groupby(
            "trade_date",
            sort=False,
        )["volume"]
        .cumsum()
    )

    dataframe["session_price_volume"] = (
        dataframe
        .groupby(
            "trade_date",
            sort=False,
        )["price_volume"]
        .cumsum()
    )

    dataframe["session_vwap"] = (
        dataframe["session_price_volume"]
        / dataframe["session_volume"].replace(
            0.0,
            np.nan,
        )
    )

    return dataframe


# =============================================================================
# CANDLE COMPLETION
# =============================================================================

def get_latest_completed_candle_index(
    dataframe: pd.DataFrame,
    current_time: datetime,
) -> int | None:
    """
    A candle timestamp represents the opening time.

    Example:
        10:15 candle -> complete at 10:30.

    Therefore:

        candle_timestamp + 15min <= now

    """

    if dataframe.empty:
        return None

    timestamps = dataframe[
        "timestamp"
    ]

    completion_times = (
        timestamps
        + pd.Timedelta(
            minutes=15
        )
    )

    current_timestamp = pd.Timestamp(
        current_time
    )

    completed = (
        completion_times
        <= current_timestamp
    )

    indices = np.flatnonzero(
        completed.to_numpy()
    )

    if len(indices) == 0:
        return None

    return int(
        indices[-1]
    )


# =============================================================================
# SCORING ENGINE
# =============================================================================

def score_vwap_family(
    close: float,
    vwap: float,
    atr: float,
    direction: str,
) -> float:
    """
    VWAP family: 15 points.
    """

    if not is_finite(
        close,
        vwap,
        atr,
    ):
        return 0.0

    if atr <= 0:
        return 0.0

    if direction == "LONG":

        distance = (
            close - vwap
        ) / atr

    else:

        distance = (
            vwap - close
        ) / atr

    normalized = clip01(
        distance / 1.50
    )

    return 15.0 * normalized


def score_ema_family(
    close: float,
    ema9: float,
    ema20: float,
    atr: float,
    direction: str,
) -> float:
    """
    EMA family: 15 points.

    Combines:
        price vs EMA9
        price vs EMA20
        EMA9 vs EMA20

    This is deliberately one family to avoid triple-counting.
    """

    if not is_finite(
        close,
        ema9,
        ema20,
        atr,
    ):
        return 0.0

    if atr <= 0:
        return 0.0

    if direction == "LONG":

        components = [
            (close - ema9) / atr,
            (close - ema20) / atr,
            (ema9 - ema20) / atr,
        ]

    else:

        components = [
            (ema9 - close) / atr,
            (ema20 - close) / atr,
            (ema20 - ema9) / atr,
        ]

    scaled = [
        clip01(
            component / 1.25
        )
        for component in components
    ]

    return (
        15.0
        * float(
            np.mean(
                scaled
            )
        )
    )


def score_momentum_family(
    rsi: float,
    macd: float,
    macd_signal: float,
    atr: float,
    direction: str,
) -> float:
    """
    Momentum family: 20 points.
    """

    if not is_finite(
        rsi,
        macd,
        macd_signal,
        atr,
    ):
        return 0.0

    if atr <= 0:
        return 0.0

    if direction == "LONG":

        rsi_component = clip01(
            (rsi - 50.0)
            / 25.0
        )

        macd_component = clip01(
            (
                (macd - macd_signal)
                / atr
            )
            / 0.50
        )

    else:

        rsi_component = clip01(
            (50.0 - rsi)
            / 25.0
        )

        macd_component = clip01(
            (
                (macd_signal - macd)
                / atr
            )
            / 0.50
        )

    combined = (
        0.50 * rsi_component
        + 0.50 * macd_component
    )

    return 20.0 * combined


def score_candle_family(
    current: pd.Series,
    previous: pd.Series,
    atr: float,
    direction: str,
) -> float:
    """
    Candle structure family: 15 points.

    Includes:
        body quality
        close location
        higher-high / lower-low confirmation
    """

    if not math.isfinite(atr):
        return 0.0

    if atr <= 0:
        return 0.0

    candle_open = safe_float(
        current["open"]
    )

    candle_high = safe_float(
        current["high"]
    )

    candle_low = safe_float(
        current["low"]
    )

    candle_close = safe_float(
        current["close"]
    )

    previous_high = safe_float(
        previous["high"]
    )

    previous_low = safe_float(
        previous["low"]
    )

    candle_range = (
        candle_high
        - candle_low
    )

    if candle_range <= EPSILON:
        return 0.0

    if direction == "LONG":

        body = (
            candle_close
            - candle_open
        )

        body_quality = clip01(
            body / atr
        )

        close_location = clip01(
            (
                candle_close
                - candle_low
            )
            / candle_range
        )

        higher_high = (
            candle_high
            - previous_high
        )

        structure = clip01(
            higher_high
            / atr
        )

    else:

        body = (
            candle_open
            - candle_close
        )

        body_quality = clip01(
            body / atr
        )

        close_location = clip01(
            (
                candle_high
                - candle_close
            )
            / candle_range
        )

        lower_low = (
            previous_low
            - candle_low
        )

        structure = clip01(
            lower_low
            / atr
        )

    return 15.0 * (
        0.35 * body_quality
        + 0.35 * close_location
        + 0.30 * structure
    )


def score_relative_family(
    stock_shift: float,
    nifty_shift: float,
    direction: str,
) -> float:
    """
    Market-relative family: 15 points.

    Relative movement:

        stock T/T-1 return
        -
        NIFTY T/T-1 return
    """

    relative_shift = (
        stock_shift
        - nifty_shift
    )

    if direction == "LONG":

        directional_relative = (
            relative_shift
        )

    else:

        directional_relative = (
            -relative_shift
        )

    normalized = clip01(
        directional_relative
        / 0.010
    )

    return 15.0 * normalized


def score_volume_family(
    volume: float,
    median_volume: float,
) -> float:
    """
    Relative-volume family: 20 points.

    1x historical median -> 0
    4x historical median -> 20
    >4x -> clipped
    """

    if not is_finite(
        volume,
        median_volume,
    ):
        return 0.0

    if median_volume <= 0:
        return 0.0

    relative_volume = (
        volume
        / median_volume
    )

    normalized = clip01(
        (
            relative_volume
            - 1.0
        )
        / 3.0
    )

    return 20.0 * normalized


def calculate_score(
    stock_dataframe: pd.DataFrame,
    nifty_dataframe: pd.DataFrame,
    candle_index: int,
    direction: str,
) -> dict[str, float] | None:
    """
    Calculate the complete 100-point score for T.
    """

    if candle_index < 1:
        return None

    if candle_index >= len(
        stock_dataframe
    ):
        return None

    current = stock_dataframe.iloc[
        candle_index
    ]

    previous = stock_dataframe.iloc[
        candle_index - 1
    ]

    timestamp = current[
        "timestamp"
    ]

    # -------------------------------------------------------------------------
    # Exact timestamp alignment with NIFTY.
    # -------------------------------------------------------------------------

    matching_indices = np.flatnonzero(
        (
            nifty_dataframe[
                "timestamp"
            ]
            == timestamp
        ).to_numpy()
    )

    if len(matching_indices) == 0:
        return None

    nifty_index = int(
        matching_indices[0]
    )

    if nifty_index < 1:
        return None

    nifty_current = (
        nifty_dataframe.iloc[
            nifty_index
        ]
    )

    nifty_previous = (
        nifty_dataframe.iloc[
            nifty_index - 1
        ]
    )

    # -------------------------------------------------------------------------
    # Stock return T-1 -> T.
    # -------------------------------------------------------------------------

    close = safe_float(
        current["close"]
    )

    previous_close = safe_float(
        previous["close"]
    )

    if not is_finite(
        close,
        previous_close,
    ):
        return None

    if close <= 0 or previous_close <= 0:
        return None

    stock_shift = (
        close
        / previous_close
        - 1.0
    )

    # -------------------------------------------------------------------------
    # NIFTY return over EXACT SAME candle interval.
    # -------------------------------------------------------------------------

    nifty_close = safe_float(
        nifty_current["close"]
    )

    nifty_previous_close = safe_float(
        nifty_previous["close"]
    )

    if not is_finite(
        nifty_close,
        nifty_previous_close,
    ):
        return None

    if nifty_close <= 0:
        return None

    if nifty_previous_close <= 0:
        return None

    nifty_shift = (
        nifty_close
        / nifty_previous_close
        - 1.0
    )

    relative_shift = (
        stock_shift
        - nifty_shift
    )

    # -------------------------------------------------------------------------
    # Indicator values.
    # -------------------------------------------------------------------------

    vwap = safe_float(
        current["session_vwap"]
    )

    ema9 = safe_float(
        current["ema9"]
    )

    ema20 = safe_float(
        current["ema20"]
    )

    rsi = safe_float(
        current["rsi14"]
    )

    macd = safe_float(
        current["macd"]
    )

    macd_signal = safe_float(
        current["macd_signal"]
    )

    atr = safe_float(
        current["atr14"]
    )

    median_volume = safe_float(
        current["volume_median_20"]
    )

    current_volume = safe_float(
        current["volume"]
    )

    if not is_finite(
        atr,
        vwap,
        ema9,
        ema20,
    ):
        return None

    if atr <= 0:
        return None

    # -------------------------------------------------------------------------
    # ATR-normalised movement.
    # -------------------------------------------------------------------------

    atr_normalized_shift = (
        close
        - previous_close
    ) / atr

    # -------------------------------------------------------------------------
    # Factor families.
    # -------------------------------------------------------------------------

    family_vwap = score_vwap_family(
        close,
        vwap,
        atr,
        direction,
    )

    family_ema = score_ema_family(
        close,
        ema9,
        ema20,
        atr,
        direction,
    )

    family_momentum = score_momentum_family(
        rsi,
        macd,
        macd_signal,
        atr,
        direction,
    )

    family_candle = score_candle_family(
        current,
        previous,
        atr,
        direction,
    )

    family_relative = score_relative_family(
        stock_shift,
        nifty_shift,
        direction,
    )

    family_volume = score_volume_family(
        current_volume,
        median_volume,
    )

    total_score = (
        family_vwap
        + family_ema
        + family_momentum
        + family_candle
        + family_relative
        + family_volume
    )

    candle_range = (
        safe_float(
            current["high"]
        )
        - safe_float(
            current["low"]
        )
    )

    relative_volume = (
        current_volume
        / median_volume
        if median_volume > 0
        else math.nan
    )

    return {
        "score": float(
            np.clip(
                total_score,
                0.0,
                100.0,
            )
        ),

        "shift_pct": stock_shift,

        "nifty_shift_pct": nifty_shift,

        "relative_shift_pct": relative_shift,

        "vwap": vwap,

        "ema9": ema9,

        "ema20": ema20,

        "rsi": rsi,

        "macd": macd,

        "macd_signal": macd_signal,

        "atr14": atr,

        "atr_normalized_shift": atr_normalized_shift,

        "candle_range": candle_range,

        "relative_volume": relative_volume,

        "family_vwap": family_vwap,

        "family_ema": family_ema,

        "family_momentum": family_momentum,

        "family_candle": family_candle,

        "family_relative": family_relative,

        "family_volume": family_volume,
    }


# =============================================================================
# DATAFRAME PREPARATION
# =============================================================================

def combine_historical_and_intraday(
    historical: pd.DataFrame,
    intraday: pd.DataFrame,
) -> pd.DataFrame:
    """
    Historical data provides indicator warm-up.

    Intraday data overrides historical records if a timestamp overlaps.
    """

    frames = []

    if not historical.empty:
        frames.append(
            historical
        )

    if not intraday.empty:
        frames.append(
            intraday
        )

    if not frames:
        return pd.DataFrame(
            columns=CANDLE_COLUMNS
        )

    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values(
            "timestamp"
        )
        .reset_index(
            drop=True
        )
    )

    return dataframe


def prepare_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if dataframe.empty:
        return dataframe

    dataframe = add_indicators(
        dataframe
    )

    dataframe = add_session_vwap(
        dataframe
    )

    return dataframe


# =============================================================================
# SIGNAL CREATION
# =============================================================================

def evaluate_instrument(
    instrument: Instrument,
    historical: pd.DataFrame,
    intraday: pd.DataFrame,
    nifty_historical: pd.DataFrame,
    nifty_intraday: pd.DataFrame,
    daily_turnover: float,
    current_time: datetime,
) -> list[Signal]:
    """
    Evaluate exactly one stock.

    The sequence is intentionally:

        historical warm-up
                +
        current intraday
                |
                v
        chronological dataframe
                |
                v
        indicators
                |
                v
        latest completed T
                |
                v
        isolated turnover
                |
                v
        T vs T-1 shift
                |
                v
        score
    """

    if intraday.empty:
        return []

    stock_dataframe = combine_historical_and_intraday(
        historical,
        intraday,
    )

    if stock_dataframe.empty:
        return []

    stock_dataframe = prepare_dataframe(
        stock_dataframe
    )

    nifty_dataframe = combine_historical_and_intraday(
        nifty_historical,
        nifty_intraday,
    )

    if nifty_dataframe.empty:
        return []

    nifty_dataframe = prepare_dataframe(
        nifty_dataframe
    )

    # -------------------------------------------------------------------------
    # Latest fully completed 15-minute candle.
    # -------------------------------------------------------------------------

    candle_index = (
        get_latest_completed_candle_index(
            stock_dataframe,
            current_time,
        )
    )

    if candle_index is None:
        return []

    if candle_index < 1:
        return []

    current = stock_dataframe.iloc[
        candle_index
    ]

    previous = stock_dataframe.iloc[
        candle_index - 1
    ]

    candle_timestamp = current[
        "timestamp"
    ]

    # -------------------------------------------------------------------------
    # Never evaluate yesterday's candle during today's session.
    # -------------------------------------------------------------------------

    if candle_timestamp.date() != current_time.date():
        return []

    # -------------------------------------------------------------------------
    # Isolated turnover.
    #
    # THIS IS THE CRITICAL FILTER.
    # -------------------------------------------------------------------------

    candle_volume = safe_float(
        current["volume"]
    )

    candle_close = safe_float(
        current["close"]
    )

    if not is_finite(
        candle_volume,
        candle_close,
    ):
        return []

    if candle_volume <= 0:
        return []

    if candle_close <= 0:
        return []

    isolated_turnover = (
        candle_volume
        * candle_close
    )

    if (
        isolated_turnover
        < MIN_ISOLATED_TURNOVER
    ):
        return []

    # -------------------------------------------------------------------------
    # Isolated T vs T-1 return.
    # -------------------------------------------------------------------------

    previous_close = safe_float(
        previous["close"]
    )

    if not math.isfinite(
        previous_close
    ):
        return []

    if previous_close <= 0:
        return []

    shift = (
        candle_close
        / previous_close
        - 1.0
    )

    # -------------------------------------------------------------------------
    # LONG candidate.
    # -------------------------------------------------------------------------

    if shift >= MIN_LONG_SHIFT:

        components = calculate_score(
            stock_dataframe,
            nifty_dataframe,
            candle_index,
            "LONG",
        )

        if (
            components is None
            or components["score"]
            < LONG_SCORE_THRESHOLD
        ):
            return []

        return [
            Signal(
                scan_timestamp=current_time,
                candle_timestamp=candle_timestamp.to_pydatetime(),
                symbol=instrument.trading_symbol,
                instrument_key=instrument.instrument_key,
                direction="LONG",
                score=components["score"],
                shift_pct=components["shift_pct"],
                nifty_shift_pct=components["nifty_shift_pct"],
                relative_shift_pct=components["relative_shift_pct"],
                close=candle_close,
                previous_close=previous_close,
                isolated_turnover=isolated_turnover,
                daily_turnover=daily_turnover,
                vwap=components["vwap"],
                ema9=components["ema9"],
                ema20=components["ema20"],
                rsi=components["rsi"],
                macd=components["macd"],
                macd_signal=components["macd_signal"],
                atr14=components["atr14"],
                atr_normalized_shift=components[
                    "atr_normalized_shift"
                ],
                candle_range=components[
                    "candle_range"
                ],
                relative_volume=components[
                    "relative_volume"
                ],
                family_vwap=components[
                    "family_vwap"
                ],
                family_ema=components[
                    "family_ema"
                ],
                family_momentum=components[
                    "family_momentum"
                ],
                family_candle=components[
                    "family_candle"
                ],
                family_relative=components[
                    "family_relative"
                ],
                family_volume=components[
                    "family_volume"
                ],
            )
        ]

    # -------------------------------------------------------------------------
    # SHORT candidate.
    # -------------------------------------------------------------------------

    if shift <= -MIN_SHORT_SHIFT:

        components = calculate_score(
            stock_dataframe,
            nifty_dataframe,
            candle_index,
            "SHORT",
        )

        if (
            components is None
            or components["score"]
            < SHORT_SCORE_THRESHOLD
        ):
            return []

        return [
            Signal(
                scan_timestamp=current_time,
                candle_timestamp=candle_timestamp.to_pydatetime(),
                symbol=instrument.trading_symbol,
                instrument_key=instrument.instrument_key,
                direction="SHORT",
                score=components["score"],
                shift_pct=components["shift_pct"],
                nifty_shift_pct=components["nifty_shift_pct"],
                relative_shift_pct=components["relative_shift_pct"],
                close=candle_close,
                previous_close=previous_close,
                isolated_turnover=isolated_turnover,
                daily_turnover=daily_turnover,
                vwap=components["vwap"],
                ema9=components["ema9"],
                ema20=components["ema20"],
                rsi=components["rsi"],
                macd=components["macd"],
                macd_signal=components["macd_signal"],
                atr14=components["atr14"],
                atr_normalized_shift=components[
                    "atr_normalized_shift"
                ],
                candle_range=components[
                    "candle_range"
                ],
                relative_volume=components[
                    "relative_volume"
                ],
                family_vwap=components[
                    "family_vwap"
                ],
                family_ema=components[
                    "family_ema"
                ],
                family_momentum=components[
                    "family_momentum"
                ],
                family_candle=components[
                    "family_candle"
                ],
                family_relative=components[
                    "family_relative"
                ],
                family_volume=components[
                    "family_volume"
                ],
            )
        ]

    return []


# =============================================================================
# CONCURRENT INTRADAY FETCHING
# =============================================================================

def fetch_candidate_intraday_data(
    session: requests.Session,
    candidates: Sequence[Instrument],
) -> dict[
    str,
    pd.DataFrame,
]:
    """
    Concurrently download current 15-minute candles.

    A single bad instrument never kills the cycle.
    """

    result: dict[
        str,
        pd.DataFrame
    ] = {}

    if not candidates:
        return result

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
        thread_name_prefix="intraday",
    ) as executor:

        futures = {
            executor.submit(
                fetch_intraday_15m,
                session,
                instrument.instrument_key,
            ): instrument
            for instrument in candidates
        }

        for future in as_completed(
            futures
        ):

            instrument = futures[
                future
            ]

            try:

                dataframe = future.result()

                if not dataframe.empty:

                    result[
                        instrument.instrument_key
                    ] = dataframe

            except requests.RequestException as exc:

                LOGGER.warning(
                    "Intraday API error [%s]: %s",
                    instrument.trading_symbol,
                    exc,
                )

            except Exception as exc:

                LOGGER.warning(
                    "Intraday processing error [%s]: %s",
                    instrument.trading_symbol,
                    exc,
                )

    return result


# =============================================================================
# SIGNAL REGISTRY
# =============================================================================

class SignalRegistry:
    """
    Prevent duplicate CSV entries.

    Same:
        instrument
        direction
        completed candle

    is written only once.
    """

    def __init__(self) -> None:

        self._keys: set[
            tuple[
                str,
                str,
                datetime,
            ]
        ] = set()

        self._lock = threading.Lock()

    def is_new(
        self,
        signal: Signal,
    ) -> bool:

        key = (
            signal.instrument_key,
            signal.direction,
            signal.candle_timestamp,
        )

        with self._lock:

            if key in self._keys:
                return False

            self._keys.add(
                key
            )

            return True


# =============================================================================
# CSV PERSISTENCE
# =============================================================================

CSV_FIELDS = [
    "scan_timestamp",
    "candle_timestamp",
    "symbol",
    "instrument_key",
    "direction",
    "score",
    "shift_pct",
    "nifty_shift_pct",
    "relative_shift_pct",
    "close",
    "previous_close",
    "isolated_turnover",
    "daily_turnover",
    "vwap",
    "ema9",
    "ema20",
    "rsi",
    "macd",
    "macd_signal",
    "atr14",
    "atr_normalized_shift",
    "candle_range",
    "relative_volume",
    "family_vwap",
    "family_ema",
    "family_momentum",
    "family_candle",
    "family_relative",
    "family_volume",
]


def signal_to_row(
    signal: Signal,
) -> dict[str, Any]:

    return {
        "scan_timestamp":
            signal.scan_timestamp.isoformat(),

        "candle_timestamp":
            signal.candle_timestamp.isoformat(),

        "symbol":
            signal.symbol,

        "instrument_key":
            signal.instrument_key,

        "direction":
            signal.direction,

        "score":
            round(
                signal.score,
                4,
            ),

        "shift_pct":
            round(
                signal.shift_pct * 100,
                6,
            ),

        "nifty_shift_pct":
            round(
                signal.nifty_shift_pct * 100,
                6,
            ),

        "relative_shift_pct":
            round(
                signal.relative_shift_pct * 100,
                6,
            ),

        "close":
            round(
                signal.close,
                4,
            ),

        "previous_close":
            round(
                signal.previous_close,
                4,
            ),

        "isolated_turnover":
            round(
                signal.isolated_turnover,
                2,
            ),

        "daily_turnover":
            round(
                signal.daily_turnover,
                2,
            ),

        "vwap":
            round(
                signal.vwap,
                4,
            ),

        "ema9":
            round(
                signal.ema9,
                4,
            ),

        "ema20":
            round(
                signal.ema20,
                4,
            ),

        "rsi":
            round(
                signal.rsi,
                4,
            ),

        "macd":
            round(
                signal.macd,
                6,
            ),

        "macd_signal":
            round(
                signal.macd_signal,
                6,
            ),

        "atr14":
            round(
                signal.atr14,
                6,
            ),

        "atr_normalized_shift":
            round(
                signal.atr_normalized_shift,
                6,
            ),

        "candle_range":
            round(
                signal.candle_range,
                4,
            ),

        "relative_volume":
            round(
                signal.relative_volume,
                4,
            ),

        "family_vwap":
            round(
                signal.family_vwap,
                4,
            ),

        "family_ema":
            round(
                signal.family_ema,
                4,
            ),

        "family_momentum":
            round(
                signal.family_momentum,
                4,
            ),

        "family_candle":
            round(
                signal.family_candle,
                4,
            ),

        "family_relative":
            round(
                signal.family_relative,
                4,
            ),

        "family_volume":
            round(
                signal.family_volume,
                4,
            ),
    }


def append_signals(
    signals: Sequence[Signal],
    path: Path,
) -> None:

    if not signals:
        return

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = (
        path.exists()
        and path.stat().st_size > 0
    )

    with path.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=CSV_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        for signal in signals:
            writer.writerow(
                signal_to_row(
                    signal
                )
            )

        handle.flush()


# =============================================================================
# NIFTY DATA
# =============================================================================

def fetch_nifty_data(
    session: requests.Session,
    cache: HistoricalCache,
    current_date: date,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    historical = cache.get(
        NIFTY_50_INSTRUMENT_KEY
    )

    if historical.empty:

        historical = fetch_historical_15m(
            session,
            NIFTY_50_INSTRUMENT_KEY,
            current_date,
            HISTORICAL_DAYS,
        )

        if not historical.empty:

            cache.put(
                NIFTY_50_INSTRUMENT_KEY,
                historical,
            )

    intraday = fetch_intraday_15m(
        session,
        NIFTY_50_INSTRUMENT_KEY,
    )

    return (
        historical,
        intraday,
    )


# =============================================================================
# DASHBOARD
# =============================================================================

def clear_terminal() -> None:
    sys.stdout.write(
        "\033[H\033[J"
    )


def render_dashboard(
    scan_time: datetime,
    candidates: int,
    quote_count: int,
    signals: Sequence[Signal],
    elapsed: float,
) -> None:

    clear_terminal()

    print(
        "=" * 125
    )

    print(
        " UPSTOX NSE 15-MINUTE CONTINUOUS BREAKOUT SCANNER"
    )

    print(
        "=" * 125
    )

    print(
        f"Time: {scan_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )

    print(
        f"Quotes: {quote_count:,} | "
        f"Daily turnover candidates: {candidates:,} | "
        f"Cycle: {elapsed:.2f}s | "
        f"Poll: {POLL_SECONDS}s"
    )

    print(
        f"Daily turnover >= {format_inr(MIN_DAILY_TURNOVER)} | "
        f"15m turnover >= {format_inr(MIN_ISOLATED_TURNOVER)}"
    )

    print(
        f"SHORT: score >= {SHORT_SCORE_THRESHOLD:.1f}, "
        f"shift <= {-MIN_SHORT_SHIFT * 100:.2f}% | "
        f"LONG: score >= {LONG_SCORE_THRESHOLD:.1f}, "
        f"shift >= {MIN_LONG_SHIFT * 100:.2f}%"
    )

    print(
        "=" * 125
    )

    if not signals:

        print()
        print(
            "No qualifying signals in the latest completed 15-minute candle."
        )
        print()

        return

    # -------------------------------------------------------------------------
    # Sort by absolute factor score.
    # -------------------------------------------------------------------------

    ordered = sorted(
        signals,
        key=lambda signal: abs(
            signal.score
        ),
        reverse=True,
    )

    # -------------------------------------------------------------------------
    # Group by completed candle timestamp.
    # -------------------------------------------------------------------------

    grouped: dict[
        datetime,
        list[Signal]
    ] = {}

    for signal in ordered:

        grouped.setdefault(
            signal.candle_timestamp,
            [],
        ).append(
            signal
        )

    for anchor in sorted(
        grouped.keys(),
        reverse=True,
    ):

        print()

        print(
            f"--- TIME ANCHOR "
            f"{anchor.strftime('%H:%M')} "
            f"(COMPLETED 15M CANDLE) ---"
        )

        print(
            f"{'SYMBOL':<14}"
            f"{'SIDE':<7}"
            f"{'SCORE':>8}"
            f"{'SHIFT':>10}"
            f"{'REL':>10}"
            f"{'15M TO':>12}"
            f"{'DAY TO':>12}"
            f"{'RVOL':>8}"
            f"{'ATR':>9}"
            f"{'RSI':>8}"
            f"{'VWAP':>12}"
        )

        print(
            "-" * 125
        )

        for signal in sorted(
            grouped[anchor],
            key=lambda item: abs(
                item.score
            ),
            reverse=True,
        ):

            print(
                f"{signal.symbol:<14}"
                f"{signal.direction:<7}"
                f"{signal.score:>8.1f}"
                f"{format_pct(signal.shift_pct):>10}"
                f"{format_pct(signal.relative_shift_pct):>10}"
                f"{format_inr(signal.isolated_turnover):>12}"
                f"{format_inr(signal.daily_turnover):>12}"
                f"{signal.relative_volume:>8.2f}"
                f"{signal.atr14:>9.2f}"
                f"{signal.rsi:>8.1f}"
                f"{signal.vwap:>12.2f}"
            )

    print()

    print(
        "Score families:"
    )

    print(
        "VWAP=15 | EMA=15 | Momentum=20 | "
        "Candle=15 | Relative=15 | Volume=20 | Total=100"
    )

    print(
        f"CSV: {CSV_PATH.resolve()}"
    )


# =============================================================================
# MARKET-CLOSED DASHBOARD
# =============================================================================

def render_market_closed(
    current_time: datetime,
) -> None:

    clear_terminal()

    print(
        "=" * 100
    )

    print(
        " UPSTOX NSE 15-MINUTE BREAKOUT SCANNER"
    )

    print(
        "=" * 100
    )

    print(
        f"Market closed | "
        f"Current time: "
        f"{current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    )

    print(
        "Scanner remains alive and will automatically resume "
        "during the next NSE session."
    )

    print(
        "=" * 100
    )


# =============================================================================
# MAIN SCANNER
# =============================================================================

def run_scanner() -> None:

    if not ACCESS_TOKEN:

        raise RuntimeError(
            "UPSTOX_ACCESS_TOKEN environment variable is not set."
        )

    session = create_http_session(
        ACCESS_TOKEN
    )

    # -------------------------------------------------------------------------
    # Instrument master.
    # -------------------------------------------------------------------------

    LOGGER.info(
        "Downloading official Upstox NSE instrument master..."
    )

    instruments = load_nse_equity_instruments(
        session
    )

    # -------------------------------------------------------------------------
    # Historical cache.
    # -------------------------------------------------------------------------

    historical_cache = HistoricalCache(
        session=session,
        days=HISTORICAL_DAYS,
    )

    signal_registry = SignalRegistry()

    # -------------------------------------------------------------------------
    # Infinite scanner loop.
    # -------------------------------------------------------------------------

    while True:

        cycle_start = time.perf_counter()

        current_time = now_ist()

        try:

            # =================================================================
            # MARKET CLOSED
            # =================================================================

            if not market_is_open(
                current_time
            ):

                render_market_closed(
                    current_time
                )

                elapsed = (
                    time.perf_counter()
                    - cycle_start
                )

                time.sleep(
                    max(
                        1.0,
                        POLL_SECONDS
                        - elapsed,
                    )
                )

                continue

            current_date = (
                current_time.date()
            )

            # =================================================================
            # RESET DAILY CACHE.
            # =================================================================

            historical_cache.reset_for_date(
                current_date
            )

            # =================================================================
            # STEP 1
            #
            # V2 full-market quotes.
            # =================================================================

            LOGGER.debug(
                "Fetching V2 market quotes..."
            )

            all_instruments = list(
                instruments.values()
            )

            quotes = fetch_market_quotes(
                session,
                all_instruments,
            )

            # =================================================================
            # STEP 2
            #
            # Daily turnover pre-screen.
            # =================================================================

            candidates, daily_turnover = (
                prescreen_instruments(
                    instruments,
                    quotes,
                )
            )

            LOGGER.info(
                "Quote snapshot: %d | "
                "Daily-turnover candidates: %d",
                len(quotes),
                len(candidates),
            )

            # =================================================================
            # STEP 3
            #
            # Historical warm-up.
            # =================================================================

            historical_cache.preload(
                candidates,
                current_date,
            )

            # =================================================================
            # STEP 4
            #
            # NIFTY 50 historical + current candles.
            # =================================================================

            nifty_historical, nifty_intraday = (
                fetch_nifty_data(
                    session,
                    historical_cache,
                    current_date,
                )
            )

            if (
                nifty_historical.empty
                or nifty_intraday.empty
            ):

                LOGGER.warning(
                    "NIFTY data unavailable for this cycle."
                )

                elapsed = (
                    time.perf_counter()
                    - cycle_start
                )

                time.sleep(
                    max(
                        1.0,
                        POLL_SECONDS
                        - elapsed,
                    )
                )

                continue

            # =================================================================
            # STEP 5
            #
            # Candidate current 15-minute candles.
            # =================================================================

            intraday_data = (
                fetch_candidate_intraday_data(
                    session,
                    candidates,
                )
            )

            # =================================================================
            # STEP 6
            #
            # Evaluate candidates.
            # =================================================================

            signals: list[Signal] = []

            for instrument in candidates:

                intraday = intraday_data.get(
                    instrument.instrument_key
                )

                if (
                    intraday is None
                    or intraday.empty
                ):
                    continue

                historical = historical_cache.get(
                    instrument.instrument_key
                )

                try:

                    instrument_signals = (
                        evaluate_instrument(
                            instrument=instrument,
                            historical=historical,
                            intraday=intraday,
                            nifty_historical=nifty_historical,
                            nifty_intraday=nifty_intraday,
                            daily_turnover=daily_turnover.get(
                                instrument.instrument_key,
                                math.nan,
                            ),
                            current_time=current_time,
                        )
                    )

                    signals.extend(
                        instrument_signals
                    )

                except Exception as exc:

                    LOGGER.warning(
                        "Evaluation failure [%s]: %s",
                        instrument.trading_symbol,
                        exc,
                    )

            # =================================================================
            # STEP 7
            #
            # Deduplicate before CSV.
            # =================================================================

            new_signals = [
                signal
                for signal in signals
                if signal_registry.is_new(
                    signal
                )
            ]

            # =================================================================
            # STEP 8
            #
            # Append-only CSV.
            # =================================================================

            append_signals(
                new_signals,
                CSV_PATH,
            )

            # =================================================================
            # STEP 9
            #
            # Dashboard.
            # =================================================================

            elapsed = (
                time.perf_counter()
                - cycle_start
            )

            render_dashboard(
                scan_time=current_time,
                candidates=len(
                    candidates
                ),
                quote_count=len(
                    quotes
                ),
                signals=signals,
                elapsed=elapsed,
            )

        # =====================================================================
        # USER EXIT.
        # =====================================================================

        except KeyboardInterrupt:

            print(
                "\nScanner stopped by user."
            )

            break

        # =====================================================================
        # NETWORK/API ERROR.
        # =====================================================================

        except requests.RequestException as exc:

            LOGGER.error(
                "Network/API failure in scan cycle: %s",
                exc,
            )

        # =====================================================================
        # UNEXPECTED ERROR.
        # =====================================================================

        except Exception as exc:

            LOGGER.exception(
                "Unexpected scan-cycle failure: %s",
                exc,
            )

        # =====================================================================
        # POLLING CONTROL.
        # =====================================================================

        elapsed = (
            time.perf_counter()
            - cycle_start
        )

        sleep_seconds = max(
            1.0,
            POLL_SECONDS
            - elapsed,
        )

        try:

            time.sleep(
                sleep_seconds
            )

        except KeyboardInterrupt:

            print(
                "\nScanner stopped by user."
            )

            break


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    run_scanner()


