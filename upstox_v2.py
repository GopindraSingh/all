#!/usr/bin/env python3
"""
NSE 500 Intraday 9/21 EMA Crossover Bot
========================================

Broker:
    Upstox

Universe:
    NSE Nifty 500 constituents

Strategy timeframe:
    5-minute candles

Signals:
    LONG:
        9 EMA crosses above 21 EMA
        Close > VWAP
        MACD histogram > 0
        RVOL > 1.5

    SHORT:
        9 EMA crosses below 21 EMA
        Close < VWAP
        MACD histogram < 0
        RVOL > 1.5

Risk:
    Target = +1.00%
    Stop   = -0.50%

Execution:
    Upstox intraday MARKET orders, product "I"

IMPORTANT:
    This program can submit LIVE orders.

    Start with:
        PAPER_TRADING=true

    Only set:
        PAPER_TRADING=false

    after independently validating the strategy, API permissions,
    quantities, order status handling, and broker behavior.

Python:
    3.10+

Install:
    pip install -U upstox-python-sdk pandas numpy requests python-dotenv

Environment:
    UPSTOX_ACCESS_TOKEN=your_access_token
    PAPER_TRADING=true

Optional:
    ORDER_QUANTITY=1
    MAX_OPEN_POSITIONS=3
    PREMARKET_TOP_N=10
    MAX_DATA_ROWS=200
    LOG_LEVEL=INFO

Notes:
    - Upstox currently exposes MarketDataStreamerV3 through its
      official Python SDK even though the SDK itself is the v2 SDK.
    - Current-day 5-minute candles are obtained through Upstox's
      V3 intraday candle API.
    - Live WebSocket data is used for immediate price monitoring.
    - Target/SL are monitored locally and a market exit is sent when
      the threshold is breached.
    - This is NOT a guaranteed-execution stop loss. A network outage,
      API outage, exchange halt, slippage, gap, or order rejection
      can cause actual execution to differ from the requested level.
"""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
import upstox_client
from dotenv import load_dotenv
from upstox_client.rest import ApiException


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IST = "Asia/Kolkata"

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)

PREMARKET_START = dt_time(9, 0)
PREMARKET_END = dt_time(9, 15)

SIGNAL_START = dt_time(9, 30)
FORCE_EXIT_TIME = dt_time(15, 20)

TARGET_PCT = 0.0100
STOP_PCT = 0.0050

RVOL_THRESHOLD = 1.5

EMA_FAST = 9
EMA_SLOW = 21

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

RVOL_PERIOD = 20

TOP_N = 10
MAX_OPEN_POSITIONS = 3
DEFAULT_QUANTITY = 1

MAX_ROWS = 200

CANDLE_REFRESH_SECONDS = 10

# NSE's public index page exposes the current Nifty 500 constituent CSV.
#
# The URL can change as NSE modifies its frontend. Keep it configurable.
NIFTY500_CSV_URL = os.getenv(
    "NIFTY500_CSV_URL",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
)

# Upstox BOD instrument files are published here.
# This is configurable because broker instrument-master URLs can change.
UPSTOX_INSTRUMENT_URL = os.getenv(
    "UPSTOX_INSTRUMENT_URL",
    "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz",
)

UPSTOX_API_BASE = "https://api.upstox.com"

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"

ORDER_QUANTITY = int(os.getenv("ORDER_QUANTITY", str(DEFAULT_QUANTITY)))
MAX_OPEN_POSITIONS = int(
    os.getenv("MAX_OPEN_POSITIONS", str(MAX_OPEN_POSITIONS))
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("MorningEMABot")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
)

_console = logging.StreamHandler(sys.stdout)
_console.setFormatter(_formatter)

_file = logging.FileHandler(
    LOG_DIR / "morning_ema_bot.log",
    encoding="utf-8",
)
_file.setFormatter(_formatter)

logger.addHandler(_console)
logger.addHandler(_file)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Instrument:
    symbol: str
    instrument_key: str
    isin: str
    exchange: str = "NSE_EQ"


@dataclass
class Position:
    symbol: str
    instrument_key: str
    side: str                 # LONG / SHORT
    quantity: int
    entry_price: float
    target_price: float
    stop_price: float
    entry_order_id: Optional[str]
    opened_at: datetime
    exit_order_id: Optional[str] = None
    closed_at: Optional[datetime] = None


@dataclass
class Candidate:
    symbol: str
    instrument_key: str
    previous_close: float
    reference_price: float
    gap_pct: float
    previous_day_volume: float
    previous_day_value: float
    score: float


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def now_ist() -> datetime:
    return datetime.now().astimezone(
        __import__("zoneinfo").ZoneInfo(IST)
    )


def current_ist_time() -> dt_time:
    return now_ist().time().replace(tzinfo=None)


def is_weekday(d: date) -> bool:
    return d.weekday() < 5


def sleep_until(target: dt_time) -> None:
    """
    Sleep until target local IST time.

    If target has already passed, return immediately.
    """
    while True:
        now = now_ist()

        target_dt = datetime.combine(
            now.date(),
            target,
            tzinfo=now.tzinfo,
        )

        remaining = (target_dt - now).total_seconds()

        if remaining <= 0:
            return

        logger.info(
            "Waiting %.1f seconds until %s IST",
            remaining,
            target.strftime("%H:%M:%S"),
        )

        time.sleep(min(remaining, 30))


# ---------------------------------------------------------------------------
# HTTP client with retries
# ---------------------------------------------------------------------------

class HTTPClient:
    def __init__(
        self,
        access_token: str,
        timeout: int = 15,
        max_retries: int = 4,
    ):
        self.access_token = access_token
        self.timeout = timeout
        self.max_retries = max_retries

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
                "User-Agent": "NSE500-MorningEMA-Bot/1.0",
            }
        )

    def get_json(
        self,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:

        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )

                response.raise_for_status()

            except Exception as exc:
                last_exception = exc

                logger.warning(
                    "HTTP GET failed attempt=%d/%d url=%s error=%s",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                )

                if attempt < self.max_retries:
                    time.sleep(2 ** (attempt - 1))

        raise RuntimeError(
            f"HTTP request failed after {self.max_retries} attempts"
        ) from last_exception


# ---------------------------------------------------------------------------
# Universe manager
# ---------------------------------------------------------------------------

class UniverseManager:
    """
    Resolves NSE Nifty 500 constituents to Upstox instrument keys.

    Nifty 500 membership comes from NSE/NSE Indices.

    Upstox's instrument master is used for broker-specific instrument keys.
    """

    def __init__(self, http: HTTPClient):
        self.http = http
        self.instrument_map: dict[str, Instrument] = {}

    def download_nifty500(self) -> pd.DataFrame:
        logger.info("Downloading Nifty 500 constituent list")

        # NSE/NiftyIndices can occasionally reject requests without browser-like
        # headers, so use a separate public session.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120 Safari/537.36"
            ),
            "Accept": "text/csv,application/csv,text/plain,*/*",
        }

        response = requests.get(
            NIFTY500_CSV_URL,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        # Some NSE/Nifty files contain BOMs.
        from io import BytesIO

        df = pd.read_csv(
            BytesIO(response.content),
            encoding="utf-8-sig",
        )

        df.columns = [
            str(c).strip().upper().replace(" ", "_")
            for c in df.columns
        ]

        logger.info(
            "Downloaded Nifty 500 file: rows=%d columns=%s",
            len(df),
            list(df.columns),
        )

        return df

    def download_upstox_instruments(self) -> pd.DataFrame:
        logger.info("Downloading Upstox instrument master")

        response = requests.get(
            UPSTOX_INSTRUMENT_URL,
            timeout=60,
            headers={
                "User-Agent": "NSE500-MorningEMA-Bot/1.0",
            },
        )

        response.raise_for_status()

        # Upstox commonly publishes compressed CSV instrument masters.
        import gzip
        from io import BytesIO

        raw = response.content

        try:
            raw = gzip.decompress(raw)
        except gzip.BadGzipFile:
            pass

        df = pd.read_csv(
            BytesIO(raw),
            low_memory=False,
        )

        df.columns = [
            str(c).strip().lower()
            for c in df.columns
        ]

        logger.info(
            "Downloaded Upstox instrument master: rows=%d",
            len(df),
        )

        return df

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
        for candidate in candidates:
            if candidate in df.columns:
                return candidate

        raise KeyError(
            f"None of columns {candidates} exist. "
            f"Available={list(df.columns)}"
        )

    def build_instrument_map(self) -> dict[str, Instrument]:
        nifty = self.download_nifty500()
        instruments = self.download_upstox_instruments()

        nifty_symbol_col = self._find_column(
            nifty,
            ["SYMBOL", "SYMBOL_NAME"],
        )

        instrument_symbol_col = self._find_column(
            instruments,
            ["trading_symbol", "tradingsymbol", "symbol"],
        )

        instrument_key_col = self._find_column(
            instruments,
            ["instrument_key"],
        )

        # Prefer ISIN for exact identity where available.
        isin_col = next(
            (
                c for c in
                ["isin", "isin_code"]
                if c in instruments.columns
            ),
            None,
        )

        segment_col = next(
            (
                c for c in
                ["segment", "exchange_segment"]
                if c in instruments.columns
            ),
            None,
        )

        instrument_type_col = next(
            (
                c for c in
                ["instrument_type", "instrument_type_name"]
                if c in instruments.columns
            ),
            None,
        )

        n500_symbols = set(
            nifty[nifty_symbol_col]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        filtered = instruments[
            instruments[instrument_symbol_col]
            .astype(str)
            .str.upper()
            .isin(n500_symbols)
        ].copy()

        # Restrict to NSE equity where the instrument master provides enough
        # information to identify the segment.
        if segment_col:
            filtered = filtered[
                filtered[segment_col]
                .astype(str)
                .str.upper()
                .isin(
                    {
                        "NSE_EQ",
                        "NSE_EQ|",
                        "NSE_EQ",
                    }
                )
                |
                filtered[segment_col]
                .astype(str)
                .str.upper()
                .str.contains("NSE_EQ", na=False)
            ]

        if instrument_type_col:
            equity_mask = (
                filtered[instrument_type_col]
                .astype(str)
                .str.upper()
                .isin({"EQ", "EQUITY"})
            )

            # Don't throw everything away if the current instrument master
            # uses another equity classification.
            if equity_mask.any():
                filtered = filtered[equity_mask]

        result: dict[str, Instrument] = {}

        for _, row in filtered.iterrows():
            symbol = str(
                row[instrument_symbol_col]
            ).strip().upper()

            key = str(
                row[instrument_key_col]
            ).strip()

            if not symbol or not key:
                continue

            isin = (
                str(row[isin_col]).strip()
                if isin_col and pd.notna(row[isin_col])
                else ""
            )

            # Only accept standard NSE_EQ instrument keys.
            if not key.startswith("NSE_EQ|"):
                continue

            # Avoid accidentally selecting derivatives / duplicates.
            if symbol in result:
                continue

            result[symbol] = Instrument(
                symbol=symbol,
                instrument_key=key,
                isin=isin,
            )

        logger.info(
            "Resolved %d Nifty 500 equities to Upstox instruments",
            len(result),
        )

        if len(result) < 450:
            logger.warning(
                "Only %d Nifty 500 constituents resolved. "
                "Check the current NSE/Upstox instrument-master schema.",
                len(result),
            )

        self.instrument_map = result

        return result


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------

class MarketDataService:
    """
    Upstox market-data wrapper.

    Responsibilities:
        - Full quotes
        - Intraday 5-minute candles
        - Historical candles
    """

    def __init__(self, http: HTTPClient):
        self.http = http

    def full_quotes(
        self,
        instrument_keys: list[str],
    ) -> dict[str, dict[str, Any]]:

        if not instrument_keys:
            return {}

        result: dict[str, dict[str, Any]] = {}

        # API accepts up to 500 instruments. Keep batches conservative.
        for start in range(0, len(instrument_keys), 400):
            batch = instrument_keys[start:start + 400]

            response = self.http.get_json(
                f"{UPSTOX_API_BASE}/v2/market-quote/quotes",
                params={
                    "instrument_key": ",".join(batch),
                },
            )

            data = response.get("data", {})

            for key, quote in data.items():
                result[key] = quote

        return result

    def _extract_ltp(self, quote: dict[str, Any]) -> Optional[float]:
        for path in (
            ("last_price",),
            ("ltp",),
            ("last_traded_price",),
            ("ohlc", "close"),
        ):
            value: Any = quote

            try:
                for item in path:
                    value = value[item]
                value = float(value)

                if math.isfinite(value) and value > 0:
                    return value
            except (KeyError, TypeError, ValueError):
                pass

        return None

    def reference_prices(
        self,
        instruments: dict[str, Instrument],
    ) -> dict[str, tuple[float, float, float]]:

        """
        Returns:
            symbol -> (reference/current price, previous close, volume)

        The full quote endpoint provides an exchange snapshot.
        """

        keys = [i.instrument_key for i in instruments.values()]
        quotes = self.full_quotes(keys)

        by_key = {
            instrument.instrument_key: instrument.symbol
            for instrument in instruments.values()
        }

        result = {}

        for key, quote in quotes.items():
            symbol = by_key.get(key)

            if not symbol:
                continue

            ltp = self._extract_ltp(quote)

            previous_close = None

            for candidate_path in (
                ("ohlc", "close"),
                ("cp",),
                ("prev_close",),
            ):
                value: Any = quote

                try:
                    for item in candidate_path:
                        value = value[item]

                    value = float(value)

                    if value > 0:
                        previous_close = value
                        break
                except (KeyError, TypeError, ValueError):
                    continue

            volume = 0.0

            for candidate_path in (
                ("volume",),
                ("vtt",),
                ("ohlc", "volume"),
            ):
                value: Any = quote

                try:
                    for item in candidate_path:
                        value = value[item]

                    volume = float(value)
                    break
                except (KeyError, TypeError, ValueError):
                    continue

            if ltp and previous_close:
                result[symbol] = (
                    ltp,
                    previous_close,
                    volume,
                )

        return result

    def intraday_5m(
        self,
        instrument_key: str,
    ) -> pd.DataFrame:

        url = (
            f"{UPSTOX_API_BASE}/v3/historical-candle/"
            f"intraday/{quote(instrument_key, safe='')}/minutes/5"
        )

        response = self.http.get_json(url)

        candles = response.get("data", {}).get("candles", [])

        if not candles:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "oi",
                ]
            )

        rows = []

        for candle in candles:
            if len(candle) < 6:
                continue

            rows.append(
                {
                    "timestamp": pd.Timestamp(candle[0]),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                    "oi": float(candle[6]) if len(candle) > 6 else 0.0,
                }
            )

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            utc=True,
        ).dt.tz_convert(IST)

        df = (
            df.sort_values("timestamp")
            .drop_duplicates("timestamp")
            .set_index("timestamp")
        )

        # Do not use a partially formed candle.
        current = now_ist()

        current_bucket = current.replace(
            minute=(current.minute // 5) * 5,
            second=0,
            microsecond=0,
        )

        if current < current_bucket + timedelta(minutes=5):
            if len(df) and df.index[-1] >= current_bucket:
                df = df.iloc[:-1]

        return df.tail(MAX_ROWS)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

class IndicatorEngine:
    @staticmethod
    def calculate(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        out = df.copy()

        close = out["close"]
        volume = out["volume"]

        # EMA
        out["ema9"] = close.ewm(
            span=EMA_FAST,
            adjust=False,
            min_periods=EMA_FAST,
        ).mean()

        out["ema21"] = close.ewm(
            span=EMA_SLOW,
            adjust=False,
            min_periods=EMA_SLOW,
        ).mean()

        # Intraday VWAP.
        # Since the DataFrame is session-specific, cumulative VWAP resets
        # naturally when the DataFrame is recreated for a new session.
        typical_price = (
            out["high"]
            + out["low"]
            + out["close"]
        ) / 3.0

        cumulative_volume = volume.cumsum()

        out["vwap"] = (
            typical_price.mul(volume).cumsum()
            / cumulative_volume.replace(0, np.nan)
        )

        # MACD
        ema12 = close.ewm(
            span=MACD_FAST,
            adjust=False,
            min_periods=MACD_FAST,
        ).mean()

        ema26 = close.ewm(
            span=MACD_SLOW,
            adjust=False,
            min_periods=MACD_SLOW,
        ).mean()

        out["macd"] = ema12 - ema26

        out["macd_signal"] = out["macd"].ewm(
            span=MACD_SIGNAL,
            adjust=False,
            min_periods=MACD_SIGNAL,
        ).mean()

        out["macd_hist"] = (
            out["macd"]
            - out["macd_signal"]
        )

        # RVOL.
        #
        # IMPORTANT:
        # shift(1) prevents the current candle from contaminating its own
        # average-volume benchmark.
        average_volume = (
            volume.rolling(
                RVOL_PERIOD,
                min_periods=RVOL_PERIOD,
            )
            .mean()
            .shift(1)
        )

        out["rvol"] = (
            volume
            / average_volume.replace(0, np.nan)
        )

        # Crossover flags.
        out["cross_up"] = (
            (out["ema9"] > out["ema21"])
            & (out["ema9"].shift(1) <= out["ema21"].shift(1))
        )

        out["cross_down"] = (
            (out["ema9"] < out["ema21"])
            & (out["ema9"].shift(1) >= out["ema21"].shift(1))
        )

        return out

    @staticmethod
    def signal(df: pd.DataFrame) -> Optional[str]:
        if len(df) < 30:
            return None

        row = df.iloc[-1]

        required = [
            "ema9",
            "ema21",
            "vwap",
            "macd_hist",
            "rvol",
        ]

        if row[required].isna().any():
            return None

        if bool(row["cross_up"]):
            if (
                row["close"] > row["vwap"]
                and row["macd_hist"] > 0
                and row["rvol"] > RVOL_THRESHOLD
            ):
                return "LONG"

        if bool(row["cross_down"]):
            if (
                row["close"] < row["vwap"]
                and row["macd_hist"] < 0
                and row["rvol"] > RVOL_THRESHOLD
            ):
                return "SHORT"

        return None


# ---------------------------------------------------------------------------
# Pre-market selector
# ---------------------------------------------------------------------------

class PremarketSelector:
    """
    Selects the top N liquid/momentum stocks.

    Pre-open gap:
        (reference_price / previous_close) - 1

    Liquidity proxy:
        log(1 + previous traded value)

    Momentum score:
        absolute gap

    Combined score:
        70% absolute gap
        30% normalized liquidity

    This deliberately does NOT claim to predict future volume.
    """

    def __init__(
        self,
        market_data: MarketDataService,
        instruments: dict[str, Instrument],
    ):
        self.market_data = market_data
        self.instruments = instruments

    def select(self, top_n: int = TOP_N) -> list[Candidate]:
        logger.info(
            "Running pre-market selection over %d stocks",
            len(self.instruments),
        )

        reference = self.market_data.reference_prices(
            self.instruments
        )

        candidates: list[Candidate] = []

        for symbol, values in reference.items():
            reference_price, previous_close, current_volume = values

            if previous_close <= 0 or reference_price <= 0:
                continue

            gap_pct = (
                reference_price / previous_close
            ) - 1.0

            # At 09:00-09:15 the exchange snapshot may not provide a
            # meaningful "current traded value" in every API mode.
            # Therefore the deterministic ranking is based primarily on gap.
            #
            # Use current quote volume as the liquidity component when it is
            # available.
            previous_day_value = (
                reference_price * current_volume
            )

            previous_day_volume = current_volume

            candidates.append(
                Candidate(
                    symbol=symbol,
                    instrument_key=self.instruments[symbol].instrument_key,
                    previous_close=previous_close,
                    reference_price=reference_price,
                    gap_pct=gap_pct,
                    previous_day_volume=previous_day_volume,
                    previous_day_value=previous_day_value,
                    score=abs(gap_pct),
                )
            )

        if not candidates:
            raise RuntimeError(
                "No valid pre-market candidates were returned."
            )

        df = pd.DataFrame(
            [
                {
                    "symbol": c.symbol,
                    "instrument_key": c.instrument_key,
                    "gap_pct": c.gap_pct,
                    "previous_day_volume": c.previous_day_volume,
                    "previous_day_value": c.previous_day_value,
                    "score": c.score,
                }
                for c in candidates
            ]
        )

        # Liquidity percentile.
        liquidity = np.log1p(
            df["previous_day_value"].clip(lower=0)
        )

        if liquidity.nunique() > 1:
            liquidity_rank = liquidity.rank(pct=True)
        else:
            liquidity_rank = pd.Series(
                0.5,
                index=df.index,
            )

        gap_rank = df["gap_pct"].abs().rank(pct=True)

        df["score"] = (
            0.70 * gap_rank
            + 0.30 * liquidity_rank
        )

        selected_df = (
            df.sort_values(
                ["score", "previous_day_value"],
                ascending=False,
            )
            .head(top_n)
        )

        selected = []

        for _, row in selected_df.iterrows():
            selected.append(
                Candidate(
                    symbol=row["symbol"],
                    instrument_key=row["instrument_key"],
                    previous_close=float(row["gap_pct"] * 0 + 1),
                    reference_price=0,
                    gap_pct=float(row["gap_pct"]),
                    previous_day_volume=float(
                        row["previous_day_volume"]
                    ),
                    previous_day_value=float(
                        row["previous_day_value"]
                    ),
                    score=float(row["score"]),
                )
            )

        logger.info("Pre-market selected stocks:")

        for candidate in selected:
            logger.info(
                "  %-15s gap=%+.3f%% score=%.4f",
                candidate.symbol,
                candidate.gap_pct * 100,
                candidate.score,
            )

        return selected


# ---------------------------------------------------------------------------
# Order manager
# ---------------------------------------------------------------------------

class OrderManager:
    """
    Encapsulates all order submission.

    Upstox order placement uses:
        product = I
        order_type = MARKET
        validity = DAY
    """

    def __init__(
        self,
        access_token: str,
        paper_trading: bool,
    ):
        self.paper_trading = paper_trading

        configuration = upstox_client.Configuration()
        configuration.access_token = access_token

        self.api_client = upstox_client.ApiClient(configuration)

        self.order_api = upstox_client.OrderApi(
            self.api_client
        )

        self.lock = threading.RLock()

    @staticmethod
    def _response_order_id(response: Any) -> Optional[str]:
        if response is None:
            return None

        # SDK models may expose .data.order_id
        try:
            data = response.data

            if hasattr(data, "order_id"):
                return str(data.order_id)

            if isinstance(data, dict):
                value = data.get("order_id")
                if value:
                    return str(value)
        except Exception:
            pass

        # Fallback to dict representation.
        try:
            payload = response.to_dict()

            if isinstance(payload, dict):
                data = payload.get("data", {})
                if isinstance(data, dict):
                    value = data.get("order_id")
                    if value:
                        return str(value)
        except Exception:
            pass

        return None

    def place_market(
        self,
        instrument_key: str,
        quantity: int,
        transaction_type: str,
        tag: str,
    ) -> Optional[str]:

        transaction_type = transaction_type.upper()

        if transaction_type not in {"BUY", "SELL"}:
            raise ValueError(
                f"Invalid transaction_type={transaction_type}"
            )

        if quantity <= 0:
            raise ValueError("quantity must be > 0")

        if self.paper_trading:
            order_id = f"PAPER-{uuid.uuid4().hex[:12]}"

            logger.warning(
                "PAPER ORDER: %s %d %s tag=%s order_id=%s",
                transaction_type,
                quantity,
                instrument_key,
                tag,
                order_id,
            )

            return order_id

        body = upstox_client.PlaceOrderRequest(
            quantity=quantity,
            product="I",
            validity="DAY",
            price=0,
            tag=tag[:40],
            instrument_token=instrument_key,
            order_type="MARKET",
            transaction_type=transaction_type,
            disclosed_quantity=0,
            trigger_price=0,
            is_amo=False,
            market_protection=-1,
        )

        with self.lock:
            try:
                response = self.order_api.place_order(
                    body,
                    "2.0",
                )

                order_id = self._response_order_id(response)

                logger.info(
                    "LIVE ORDER PLACED: %s %d %s order_id=%s",
                    transaction_type,
                    quantity,
                    instrument_key,
                    order_id,
                )

                return order_id

            except ApiException as exc:
                logger.exception(
                    "Upstox order API exception: %s",
                    exc,
                )
                raise

            except Exception:
                logger.exception(
                    "Unexpected order placement exception"
                )
                raise


# ---------------------------------------------------------------------------
# Position manager
# ---------------------------------------------------------------------------

class PositionManager:
    def __init__(
        self,
        order_manager: OrderManager,
    ):
        self.order_manager = order_manager
        self.positions: dict[str, Position] = {}
        self.lock = threading.RLock()

    def has_position(self, symbol: str) -> bool:
        with self.lock:
            return symbol in self.positions

    def count(self) -> int:
        with self.lock:
            return len(self.positions)

    def get(self, symbol: str) -> Optional[Position]:
        with self.lock:
            return self.positions.get(symbol)

    def open(
        self,
        symbol: str,
        instrument_key: str,
        side: str,
        quantity: int,
        execution_price: float,
    ) -> Optional[Position]:

        with self.lock:
            if symbol in self.positions:
                logger.warning(
                    "Duplicate entry prevented for %s",
                    symbol,
                )
                return None

            if self.count() >= MAX_OPEN_POSITIONS:
                logger.info(
                    "Maximum open positions reached: %d",
                    MAX_OPEN_POSITIONS,
                )
                return None

            if side == "LONG":
                transaction = "BUY"

                target = execution_price * (
                    1.0 + TARGET_PCT
                )

                stop = execution_price * (
                    1.0 - STOP_PCT
                )

            elif side == "SHORT":
                transaction = "SELL"

                target = execution_price * (
                    1.0 - TARGET_PCT
                )

                stop = execution_price * (
                    1.0 + STOP_PCT
                )

            else:
                raise ValueError(side)

            order_id = self.order_manager.place_market(
                instrument_key=instrument_key,
                quantity=quantity,
                transaction_type=transaction,
                tag=f"EMA_{side}",
            )

            if not order_id:
                logger.error(
                    "Entry order produced no order ID for %s",
                    symbol,
                )

            position = Position(
                symbol=symbol,
                instrument_key=instrument_key,
                side=side,
                quantity=quantity,
                entry_price=execution_price,
                target_price=target,
                stop_price=stop,
                entry_order_id=order_id,
                opened_at=now_ist(),
            )

            self.positions[symbol] = position

            logger.info(
                "POSITION OPENED | %s | %s | qty=%d | "
                "entry=%.4f target=%.4f stop=%.4f",
                symbol,
                side,
                quantity,
                execution_price,
                target,
                stop,
            )

            return position

    def close(
        self,
        symbol: str,
        reason: str,
        market_price: float,
    ) -> bool:

        with self.lock:
            position = self.positions.get(symbol)

            if not position:
                return False

            transaction = (
                "SELL"
                if position.side == "LONG"
                else "BUY"
            )

            logger.warning(
                "EXIT REQUEST | %s | reason=%s | market=%.4f",
                symbol,
                reason,
                market_price,
            )

            try:
                order_id = self.order_manager.place_market(
                    instrument_key=position.instrument_key,
                    quantity=position.quantity,
                    transaction_type=transaction,
                    tag=f"EXIT_{reason}"[:40],
                )

            except Exception:
                logger.exception(
                    "EXIT ORDER FAILED for %s",
                    symbol,
                )

                # Keep the position in memory because the broker order
                # did not successfully submit.
                return False

            position.exit_order_id = order_id
            position.closed_at = now_ist()

            if position.side == "LONG":
                pnl = (
                    market_price - position.entry_price
                ) * position.quantity
            else:
                pnl = (
                    position.entry_price - market_price
                ) * position.quantity

            logger.warning(
                "POSITION CLOSED | %s | side=%s | reason=%s | "
                "entry=%.4f exit=%.4f approximate_pnl=%.2f",
                symbol,
                position.side,
                reason,
                position.entry_price,
                market_price,
                pnl,
            )

            del self.positions[symbol]

            return True

    def close_all(
        self,
        reason: str,
        price_map: dict[str, float],
    ) -> None:

        for symbol in list(self.positions.keys()):
            price = price_map.get(symbol)

            if price is None:
                logger.error(
                    "Cannot close %s: no current price",
                    symbol,
                )
                continue

            self.close(
                symbol=symbol,
                reason=reason,
                market_price=price,
            )


# ---------------------------------------------------------------------------
# Live WebSocket
# ---------------------------------------------------------------------------

class LiveMarketStreamer:
    """
    Wrapper around the official Upstox MarketDataStreamerV3.

    We use:
        full

    because the full mode contains real-time LTP data and market
    information required for exit monitoring.
    """

    def __init__(
        self,
        access_token: str,
        instrument_keys: list[str],
        on_price: callable,
    ):
        configuration = upstox_client.Configuration()
        configuration.access_token = access_token

        self.api_client = upstox_client.ApiClient(
            configuration
        )

        self.instrument_keys = instrument_keys
        self.on_price = on_price

        self.streamer = upstox_client.MarketDataStreamerV3(
            self.api_client,
            instrument_keys,
            "full",
        )

        self.connected = threading.Event()
        self.stop_event = threading.Event()

        self.streamer.on(
            "open",
            self._on_open,
        )

        self.streamer.on(
            "message",
            self._on_message,
        )

        self.streamer.on(
            "error",
            self._on_error,
        )

        self.streamer.on(
            "close",
            self._on_close,
        )

        # Let SDK manage websocket reconnection.
        try:
            self.streamer.auto_reconnect(
                True,
                5,
                100,
            )
        except Exception:
            logger.exception(
                "Unable to configure streamer auto-reconnect"
            )

    def _on_open(self) -> None:
        logger.info(
            "Upstox market WebSocket connected"
        )

        self.connected.set()

    def _on_close(self, *args: Any) -> None:
        logger.warning(
            "Upstox market WebSocket closed: %s",
            args,
        )

        self.connected.clear()

    def _on_error(self, error: Any) -> None:
        logger.error(
            "Upstox market WebSocket error: %s",
            error,
        )

        self.connected.clear()

    @staticmethod
    def _recursive_find(
        obj: Any,
        keys: tuple[str, ...],
    ) -> Optional[Any]:

        if isinstance(obj, dict):
            for key in keys:
                if key in obj:
                    return obj[key]

            for value in obj.values():
                result = LiveMarketStreamer._recursive_find(
                    value,
                    keys,
                )

                if result is not None:
                    return result

        elif isinstance(obj, list):
            for value in obj:
                result = LiveMarketStreamer._recursive_find(
                    value,
                    keys,
                )

                if result is not None:
                    return result

        return None

    @staticmethod
    def _extract_prices(
        message: dict[str, Any],
    ) -> list[tuple[str, float]]:
        """
        Handle common V3 decoded message structures.

        The SDK converts protobuf to a Python dictionary, but the exact
        nesting can evolve. Therefore this parser intentionally searches
        defensively rather than depending on one private protobuf layout.
        """

        results: list[tuple[str, float]] = []

        feeds = message.get("feeds")

        if not isinstance(feeds, dict):
            return results

        for instrument_key, feed in feeds.items():
            if not isinstance(feed, dict):
                continue

            price = LiveMarketStreamer._recursive_find(
                feed,
                (
                    "ltp",
                    "lastTradedPrice",
                    "last_traded_price",
                ),
            )

            if price is None:
                continue

            try:
                price = float(price)
            except (TypeError, ValueError):
                continue

            if not math.isfinite(price) or price <= 0:
                continue

            results.append(
                (
                    str(instrument_key),
                    price,
                )
            )

        return results

    def _on_message(
        self,
        message: dict[str, Any],
    ) -> None:

        try:
            prices = self._extract_prices(message)

            for instrument_key, price in prices:
                self.on_price(
                    instrument_key,
                    price,
                )

        except Exception:
            logger.exception(
                "Exception processing market-data message"
            )

    def start(self) -> None:
        logger.info(
            "Starting Upstox market streamer for %d instruments",
            len(self.instrument_keys),
        )

        self.streamer.connect()

    def stop(self) -> None:
        self.stop_event.set()

        try:
            self.streamer.disconnect()
        except Exception:
            logger.exception(
                "Error disconnecting market streamer"
            )


# ---------------------------------------------------------------------------
# Main trading bot
# ---------------------------------------------------------------------------

class MorningEMABot:
    def __init__(
        self,
        access_token: str,
    ):
        self.access_token = access_token

        self.http = HTTPClient(
            access_token=access_token,
        )

        self.universe_manager = UniverseManager(
            self.http
        )

        self.market_data = MarketDataService(
            self.http
        )

        self.order_manager = OrderManager(
            access_token=access_token,
            paper_trading=PAPER_TRADING,
        )

        self.position_manager = PositionManager(
            self.order_manager
        )

        self.instruments: dict[str, Instrument] = {}

        self.selected: dict[str, Instrument] = {}

        self.frames: dict[str, pd.DataFrame] = {}

        self.last_processed_candle: dict[str, pd.Timestamp] = {}

        self.latest_prices: dict[str, float] = {}

        self.latest_price_lock = threading.RLock()

        self.streamer: Optional[LiveMarketStreamer] = None

        self.stop_event = threading.Event()

        self.strategy_lock = threading.RLock()

    # ---------------------------------------------------------------------
    # Startup
    # ---------------------------------------------------------------------

    def validate(self) -> None:
        if not self.access_token:
            raise RuntimeError(
                "UPSTOX_ACCESS_TOKEN is not set."
            )

        if ORDER_QUANTITY <= 0:
            raise RuntimeError(
                "ORDER_QUANTITY must be > 0"
            )

        logger.info(
            "Configuration | paper=%s quantity=%d max_positions=%d",
            PAPER_TRADING,
            ORDER_QUANTITY,
            MAX_OPEN_POSITIONS,
        )

        if PAPER_TRADING:
            logger.warning(
                "PAPER_TRADING=true: NO LIVE ORDERS WILL BE SENT."
            )
        else:
            logger.warning(
                "PAPER_TRADING=false: LIVE ORDERS ENABLED."
            )

    def load_universe(self) -> None:
        self.instruments = (
            self.universe_manager.build_instrument_map()
        )

        if not self.instruments:
            raise RuntimeError(
                "Failed to build NSE 500 instrument universe."
            )

    # ---------------------------------------------------------------------
    # Premarket
    # ---------------------------------------------------------------------

    def run_premarket_selection(self) -> None:
        selector = PremarketSelector(
            market_data=self.market_data,
            instruments=self.instruments,
        )

        candidates = selector.select(
            top_n=TOP_N,
        )

        if not candidates:
            raise RuntimeError(
                "Premarket selection returned zero stocks."
            )

        self.selected = {
            candidate.symbol: self.instruments[candidate.symbol]
            for candidate in candidates
        }

        logger.info(
            "Selected %d instruments for live trading: %s",
            len(self.selected),
            ", ".join(self.selected.keys()),
        )

    # ---------------------------------------------------------------------
    # Candle state
    # ---------------------------------------------------------------------

    def initialize_candles(self) -> None:
        logger.info(
            "Loading initial 5-minute candles"
        )

        for symbol, instrument in self.selected.items():
            try:
                df = self.market_data.intraday_5m(
                    instrument.instrument_key
                )

                if df.empty:
                    logger.warning(
                        "%s: no intraday candles available yet",
                        symbol,
                    )
                    continue

                df = IndicatorEngine.calculate(df)

                self.frames[symbol] = df.tail(
                    MAX_ROWS
                )

                logger.info(
                    "%s: loaded %d candles",
                    symbol,
                    len(df),
                )

            except Exception:
                logger.exception(
                    "%s: failed loading initial candles",
                    symbol,
                )

    def refresh_candles(self) -> None:
        for symbol, instrument in self.selected.items():
            if self.stop_event.is_set():
                return

            try:
                df = self.market_data.intraday_5m(
                    instrument.instrument_key
                )

                if df.empty:
                    continue

                df = IndicatorEngine.calculate(
                    df.tail(MAX_ROWS)
                )

                with self.strategy_lock:
                    self.frames[symbol] = df

                self.process_completed_candle(
                    symbol
                )

            except Exception:
                logger.exception(
                    "%s: candle refresh failed",
                    symbol,
                )

    # ---------------------------------------------------------------------
    # Strategy
    # ---------------------------------------------------------------------

    def process_completed_candle(
        self,
        symbol: str,
    ) -> None:

        with self.strategy_lock:
            df = self.frames.get(symbol)

            if df is None or len(df) < 30:
                return

            candle_timestamp = df.index[-1]

            last_processed = self.last_processed_candle.get(
                symbol
            )

            if (
                last_processed is not None
                and candle_timestamp <= last_processed
            ):
                return

            self.last_processed_candle[
                symbol
            ] = candle_timestamp

            signal_name = IndicatorEngine.signal(df)

            row = df.iloc[-1]

        logger.info(
            "%s | candle=%s close=%.4f ema9=%.4f ema21=%.4f "
            "vwap=%.4f macd_hist=%.6f rvol=%.2f signal=%s",
            symbol,
            candle_timestamp,
            row["close"],
            row["ema9"],
            row["ema21"],
            row["vwap"],
            row["macd_hist"],
            row["rvol"],
            signal_name,
        )

        if signal_name is None:
            return

        # Never duplicate a position.
        if self.position_manager.has_position(symbol):
            logger.info(
                "%s: signal ignored because position is already active",
                symbol,
            )
            return

        # Portfolio-level position limit.
        if (
            self.position_manager.count()
            >= MAX_OPEN_POSITIONS
        ):
            logger.info(
                "%s: signal ignored because max positions reached",
                symbol,
            )
            return

        # The crossover is evaluated on the completed candle.
        # Entry is assumed at the current executable price, not the
        # historical candle close.
        with self.latest_price_lock:
            execution_price = self.latest_prices.get(
                symbol
            )

        if execution_price is None:
            execution_price = float(row["close"])

        logger.warning(
            "CONFIRMED SIGNAL | %s | %s | "
            "price=%.4f VWAP=%.4f MACD_hist=%.6f RVOL=%.2f",
            symbol,
            signal_name,
            execution_price,
            row["vwap"],
            row["macd_hist"],
            row["rvol"],
        )

        try:
            self.position_manager.open(
                symbol=symbol,
                instrument_key=self.selected[
                    symbol
                ].instrument_key,
                side=signal_name,
                quantity=ORDER_QUANTITY,
                execution_price=execution_price,
            )

        except Exception:
            logger.exception(
                "%s: entry failed",
                symbol,
            )

    # ---------------------------------------------------------------------
    # Live prices / exits
    # ---------------------------------------------------------------------

    def on_live_price(
        self,
        instrument_key: str,
        price: float,
    ) -> None:

        symbol = None

        for candidate_symbol, instrument in self.selected.items():
            if instrument.instrument_key == instrument_key:
                symbol = candidate_symbol
                break

        if symbol is None:
            return

        with self.latest_price_lock:
            self.latest_prices[symbol] = price

        position = self.position_manager.get(
            symbol
        )

        if position is None:
            return

        # LONG:
        #   target = +1%
        #   stop   = -0.5%
        #
        # SHORT:
        #   target = -1%
        #   stop   = +0.5%

        if position.side == "LONG":
            if price >= position.target_price:
                self.position_manager.close(
                    symbol=symbol,
                    reason="TARGET",
                    market_price=price,
                )

            elif price <= position.stop_price:
                self.position_manager.close(
                    symbol=symbol,
                    reason="STOP",
                    market_price=price,
                )

        elif position.side == "SHORT":
            if price <= position.target_price:
                self.position_manager.close(
                    symbol=symbol,
                    reason="TARGET",
                    market_price=price,
                )

            elif price >= position.stop_price:
                self.position_manager.close(
                    symbol=symbol,
                    reason="STOP",
                    market_price=price,
                )

    def start_streamer(self) -> None:
        keys = [
            instrument.instrument_key
            for instrument in self.selected.values()
        ]

        self.streamer = LiveMarketStreamer(
            access_token=self.access_token,
            instrument_keys=keys,
            on_price=self.on_live_price,
        )

        self.streamer.start()

    # ---------------------------------------------------------------------
    # Forced exit
    # ---------------------------------------------------------------------

    def force_exit_if_required(self) -> None:
        current = current_ist_time()

        if current >= FORCE_EXIT_TIME:
            with self.latest_price_lock:
                price_map = dict(
                    self.latest_prices
                )

            self.position_manager.close_all(
                reason="FORCE_EXIT",
                price_map=price_map,
            )

    # ---------------------------------------------------------------------
    # Main session
    # ---------------------------------------------------------------------

    def run_session(self) -> None:
        logger.info(
            "Waiting for market open / signal window"
        )

        sleep_until(SIGNAL_START)

        logger.info(
            "Signal processing started at %s IST",
            now_ist().strftime("%H:%M:%S"),
        )

        next_refresh = 0.0

        while not self.stop_event.is_set():
            current = current_ist_time()

            if current >= FORCE_EXIT_TIME:
                self.force_exit_if_required()
                break

            if current >= MARKET_CLOSE:
                break

            now_monotonic = time.monotonic()

            if now_monotonic >= next_refresh:
                self.refresh_candles()

                next_refresh = (
                    now_monotonic
                    + CANDLE_REFRESH_SECONDS
                )

            self.force_exit_if_required()

            time.sleep(1)

        logger.info(
            "Trading session loop terminated"
        )

    # ---------------------------------------------------------------------
    # Shutdown
    # ---------------------------------------------------------------------

    def shutdown(self) -> None:
        logger.info("Shutting down bot")

        self.stop_event.set()

        with self.latest_price_lock:
            prices = dict(self.latest_prices)

        if self.position_manager.count() > 0:
            logger.warning(
                "Shutdown with %d open positions",
                self.position_manager.count(),
            )

            self.position_manager.close_all(
                reason="SHUTDOWN",
                price_map=prices,
            )

        if self.streamer:
            self.streamer.stop()

        logger.info("Shutdown complete")

    # ---------------------------------------------------------------------
    # Complete lifecycle
    # ---------------------------------------------------------------------

    def run(self) -> None:
        self.validate()

        # Universe can be loaded before the pre-open window.
        self.load_universe()

        # If started before 09:00, wait.
        if current_ist_time() < PREMARKET_START:
            sleep_until(PREMARKET_START)

        # Premarket selection.
        self.run_premarket_selection()

        # Wait for NSE continuous market.
        sleep_until(MARKET_OPEN)

        # Initial 5-minute candle data.
        self.initialize_candles()

        # Start live stream.
        self.start_streamer()

        # Give WebSocket a moment to establish.
        time.sleep(2)

        # Main strategy loop.
        self.run_session()


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

BOT: Optional[MorningEMABot] = None


def handle_shutdown(
    signum: int,
    frame: Any,
) -> None:

    logger.warning(
        "Received signal %s",
        signum,
    )

    if BOT:
        BOT.shutdown()

    sys.exit(0)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    global BOT

    load_dotenv()

    if not ACCESS_TOKEN:
        logger.error(
            "UPSTOX_ACCESS_TOKEN environment variable is missing."
        )

        logger.error(
            "Create a .env file containing:"
        )

        logger.error(
            "UPSTOX_ACCESS_TOKEN=your_token"
        )

        sys.exit(1)

    BOT = MorningEMABot(
        access_token=ACCESS_TOKEN,
    )

    signal.signal(
        signal.SIGINT,
        handle_shutdown,
    )

    signal.signal(
        signal.SIGTERM,
        handle_shutdown,
    )

    try:
        BOT.run()

    except KeyboardInterrupt:
        logger.warning(
            "Keyboard interrupt"
        )

    except Exception:
        logger.exception(
            "Fatal bot exception"
        )

        raise

    finally:
        if BOT:
            BOT.shutdown()


if __name__ == "__main__":
    main()

