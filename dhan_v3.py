
"""
INTRADAY MOMENTUM SCANNER V2
============================

Research-first / signal-only quantitative trading engine.

MODES
-----
1. Current Day
   Scan today's 09:45 setup.

2. Historical Date
   Reproduce the exact 09:45 decision for a historical trading day.

3. Single Stock
   Analyze one NSE stock on a selected date and explain the complete
   decision path.

4. Backtest
   Run the same strategy across a historical date range.

SELECTION
---------
Maximum:
    5 qualifying SHORT candidates
    5 qualifying LONG candidates

IMPORTANT:
    The engine NEVER manufactures candidates to reach five.
    If only 2 stocks qualify for SHORT, it returns 2/5.

NO TELEGRAM
-----------
Telegram is deliberately excluded from V2.

NO ORDER EXECUTION
------------------
This application generates research/signals only.

INSTALL
-------
pip install pandas numpy requests dhanhq python-dotenv

ENVIRONMENT
-----------
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_access_token

OPTIONAL
--------
MAX_WORKERS=16
REQUEST_TIMEOUT=20
LOG_LEVEL=INFO
CACHE_DIR=./data
OUTPUT_DIR=./output
"""

from __future__ import annotations

import io
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from dhanhq import DhanContext, dhanhq


# =============================================================================
# ENVIRONMENT
# =============================================================================

load_dotenv()


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

IST = ZoneInfo("Asia/Kolkata")

TIMEFRAME = "15"

MARKET_OPEN = dt_time(9, 15)
CANDLE_2_START = dt_time(9, 30)
DECISION_TIME = dt_time(9, 45)

WARMUP_DAYS = 30

MAX_WORKERS = int(
    os.getenv("MAX_WORKERS", "16")
)

REQUEST_TIMEOUT = int(
    os.getenv("REQUEST_TIMEOUT", "20")
)

CACHE_DIR = Path(
    os.getenv("CACHE_DIR", "./data")
)

OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", "./output")
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# LOGGING
# =============================================================================

LOGGER = logging.getLogger(
    "intraday_momentum_scanner"
)


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(
            logging,
            os.getenv(
                "LOG_LEVEL",
                "INFO",
            ).upper(),
            logging.INFO,
        ),
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(threadName)s | "
            "%(name)s | "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class EquityInstrument:
    symbol: str
    security_id: str
    exchange_segment: str = "NSE_EQ"
    instrument_type: str = "EQUITY"


@dataclass(frozen=True)
class IndexInstrument:
    name: str
    security_id: str
    exchange_segment: str = "IDX_I"
    instrument_type: str = "INDEX"


@dataclass
class ScoringConfig:
    """
    Strategy scoring parameters.

    Base score:
        50.0

    The original specification defines continuous scoring but does not
    specify exact transfer coefficients. Therefore all coefficients are
    centralized here and can be calibrated without touching the strategy.

    The scoring model is deliberately deterministic.
    """

    # Directional momentum.
    directional_max_points: float = 18.0
    directional_full_move_pct: float = 1.00

    # Index alpha.
    index_alpha_max_points: float = 12.0
    index_alpha_full_move_pct: float = 1.00

    # VWAP.
    vwap_points: float = 7.0

    # EMA structure.
    ema_points: float = 8.0

    # RSI.
    rsi_sustainable_bonus: float = 5.0
    rsi_exhaustion_penalty: float = 8.0

    # Gap fading.
    gap_fade_bonus: float = 15.0

    # Sector relative strength.
    sector_weakness_bonus: float = 10.0
    sector_strength_bonus: float = 10.0


@dataclass
class SignalResult:
    symbol: str
    signal: str
    conviction: str

    structural_score: float
    short_score: float
    long_score: float

    premarket_gap_pct: float

    sector: str
    sector_total_pct: Optional[float]
    sector_relative_strength: float

    opening_pct: float
    confirmation_pct: float
    total_pct: float

    nifty_total_pct: float
    index_alpha_pct: float

    turnover_crore: float
    execution_price: float

    vwap: float
    ema9: float
    ema20: float
    rsi14: float
    atr_pct: float

    liquidity_pass: bool
    short_criteria_pass: bool
    long_criteria_pass: bool

    decision_timestamp: str


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clamp(
    value: float,
    lower: float,
    upper: float,
) -> float:

    return float(
        max(
            lower,
            min(
                upper,
                value,
            ),
        )
    )


def continuous_points(
    magnitude_pct: float,
    full_move_pct: float,
    max_points: float,
) -> float:

    if not np.isfinite(magnitude_pct):
        return 0.0

    if full_move_pct <= 0:
        return 0.0

    return clamp(
        abs(magnitude_pct)
        / full_move_pct
        * max_points,
        0.0,
        max_points,
    )


def clean_symbol(value: Any) -> str:

    symbol = str(value).strip().upper()

    if symbol.endswith(".NS"):
        symbol = symbol[:-3]

    return symbol


def is_valid_number(value: Any) -> bool:

    try:
        return bool(
            np.isfinite(
                float(value)
            )
        )
    except Exception:
        return False


def epoch_to_ist(value: Any) -> pd.Timestamp:

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        value_float = float(value)

        unit = (
            "ms"
            if abs(value_float)
            > 10_000_000_000
            else "s"
        )

        return pd.to_datetime(
            value_float,
            unit=unit,
            utc=True,
        ).tz_convert(IST)

    return pd.to_datetime(
        value,
        utc=True,
    ).tz_convert(IST)


# =============================================================================
# NSE UNIVERSE
# =============================================================================

class NSEUniverse:

    NIFTY500_URL = (
        "https://nsearchives.nseindia.com/"
        "content/indices/ind_nifty500list.csv"
    )

    NIFTY50_URL = (
        "https://nsearchives.nseindia.com/"
        "content/indices/ind_nifty50list.csv"
    )

    NSE_URL = "https://www.nseindia.com"

    def __init__(
        self,
        session: requests.Session,
    ):
        self.session = session

    def _download(
        self,
        url: str,
    ) -> pd.DataFrame:

        response = self.session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={
                "Referer": self.NSE_URL,
                "Accept": (
                    "text/csv,"
                    "application/octet-stream,"
                    "*/*"
                ),
            },
        )

        response.raise_for_status()

        return pd.read_csv(
            io.BytesIO(
                response.content
            )
        )

    @staticmethod
    def _symbols(
        df: pd.DataFrame,
    ) -> set[str]:

        lookup = {
            str(column)
            .strip()
            .lower(): column
            for column in df.columns
        }

        symbol_column = None

        for candidate in (
            "symbol",
            "symbol name",
        ):
            if candidate in lookup:
                symbol_column = lookup[
                    candidate
                ]
                break

        if symbol_column is None:
            raise RuntimeError(
                "Could not locate symbol column in NSE file"
            )

        return {
            clean_symbol(symbol)
            for symbol in df[
                symbol_column
            ].dropna()
        }

    def load_universe(
        self,
    ) -> list[str]:

        nifty500 = self._download(
            self.NIFTY500_URL
        )

        nifty50 = self._download(
            self.NIFTY50_URL
        )

        universe = (
            self._symbols(nifty500)
        )

        excluded = (
            self._symbols(nifty50)
        )

        result = sorted(
            universe - excluded
        )

        LOGGER.info(
            "NIFTY universe: "
            "500=%d | 50=%d | "
            "scanner=%d",
            len(universe),
            len(excluded),
            len(result),
        )

        return result


# =============================================================================
# DHAN INSTRUMENT MASTER
# =============================================================================

class DhanInstrumentMaster:

    MASTER_URL = (
        "https://images.dhan.co/api-data/"
        "api-scrip-master-detailed.csv"
    )

    def __init__(
        self,
        session: requests.Session,
    ):
        self.session = session
        self.df: Optional[
            pd.DataFrame
        ] = None

    def load(self) -> None:

        LOGGER.info(
            "Loading Dhan instrument master..."
        )

        response = self.session.get(
            self.MASTER_URL,
            timeout=60,
        )

        response.raise_for_status()

        self.df = pd.read_csv(
            io.BytesIO(
                response.content
            ),
            low_memory=False,
        )

        self.df.columns = [
            str(c).strip().upper()
            for c in self.df.columns
        ]

        LOGGER.info(
            "Dhan master rows: %d",
            len(self.df),
        )

    def _get_df(self) -> pd.DataFrame:

        if self.df is None:
            raise RuntimeError(
                "Dhan instrument master not loaded"
            )

        return self.df

    def _id_column(
        self,
        df: pd.DataFrame,
    ) -> str:

        for candidate in (
            "SECURITY_ID",
            "SEM_SMST_SECURITY_ID",
            "SEM_SECURITY_ID",
        ):
            if candidate in df.columns:
                return candidate

        raise RuntimeError(
            "Security ID column not found"
        )

    def _symbol_columns(
        self,
        df: pd.DataFrame,
    ) -> list[str]:

        columns = []

        for candidate in (
            "SYMBOL_NAME",
            "SEM_TRADING_SYMBOL",
            "DISPLAY_NAME",
            "SEM_CUSTOM_SYMBOL",
        ):
            if candidate in df.columns:
                columns.append(candidate)

        if not columns:
            raise RuntimeError(
                "No symbol column found in Dhan master"
            )

        return columns

    def _resolve(
        self,
        symbol: str,
        exchange_segment: str,
        instrument_type: str,
    ) -> str:

        df = self._get_df()

        id_column = self._id_column(df)

        symbol_columns = (
            self._symbol_columns(df)
        )

        candidates = df.copy()

        # --------------------------------------------------------------
        # Exchange filtering.
        # --------------------------------------------------------------

        exchange_columns = [
            column
            for column in (
                "SEGMENT",
                "SEM_SEGMENT",
                "EXCH_ID",
            )
            if column in candidates.columns
        ]

        if exchange_columns:

            exchange_column = (
                exchange_columns[0]
            )

            values = (
                candidates[
                    exchange_column
                ]
                .astype(str)
                .str.upper()
            )

            if exchange_segment == "NSE_EQ":

                candidates = candidates[
                    values.str.contains(
                        "NSE"
                    )
                    | values.isin(
                        {"E"}
                    )
                ]

            elif exchange_segment == "IDX_I":

                candidates = candidates[
                    values.str.contains(
                        "IDX"
                    )
                    | values.isin(
                        {"I"}
                    )
                ]

        # --------------------------------------------------------------
        # Instrument filtering.
        # --------------------------------------------------------------

        instrument_columns = [
            column
            for column in (
                "INSTRUMENT",
                "SEM_INSTRUMENT_NAME",
            )
            if column in candidates.columns
        ]

        if instrument_columns:

            instrument_column = (
                instrument_columns[0]
            )

            wanted = (
                instrument_type.upper()
            )

            values = (
                candidates[
                    instrument_column
                ]
                .astype(str)
                .str.upper()
            )

            candidates = candidates[
                values.eq(wanted)
            ]

        # --------------------------------------------------------------
        # Symbol matching.
        # --------------------------------------------------------------

        target = clean_symbol(symbol)

        mask = pd.Series(
            False,
            index=candidates.index,
        )

        for column in symbol_columns:

            values = (
                candidates[column]
                .astype(str)
                .map(clean_symbol)
            )

            mask |= values.eq(target)

        matches = candidates[
            mask
        ]

        if matches.empty:
            raise KeyError(
                f"Instrument not found: "
                f"{symbol} / "
                f"{exchange_segment} / "
                f"{instrument_type}"
            )

        return str(
            matches.iloc[0][
                id_column
            ]
        )

    def resolve_equity(
        self,
        symbol: str,
    ) -> EquityInstrument:

        return EquityInstrument(
            symbol=clean_symbol(symbol),
            security_id=self._resolve(
                symbol,
                "NSE_EQ",
                "EQUITY",
            ),
        )

    def resolve_index(
        self,
        name: str,
    ) -> IndexInstrument:

        return IndexInstrument(
            name=name,
            security_id=self._resolve(
                name,
                "IDX_I",
                "INDEX",
            ),
        )


# =============================================================================
# DHAN DATA ENGINE
# =============================================================================

class DhanDataEngine:

    def __init__(
        self,
        client_id: str,
        access_token: str,
    ):

        context = DhanContext(
            client_id,
            access_token,
        )

        self.client = dhanhq(
            context
        )

        # Protect the shared SDK client.
        self._lock = threading.RLock()

    def fetch_intraday(
        self,
        instrument: (
            EquityInstrument
            | IndexInstrument
        ),
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:

        raw = self._request_with_retry(
            instrument,
            start_date,
            end_date,
        )

        return clean_ticker_data(
            raw
        )

    def _request_with_retry(
        self,
        instrument: (
            EquityInstrument
            | IndexInstrument
        ),
        start_date: date,
        end_date: date,
    ) -> dict[str, Any]:

        attempts = 3

        for attempt in range(
            1,
            attempts + 1,
        ):

            try:

                with self._lock:

                    response = (
                        self.client
                        .intraday_minute_data(
                            security_id=(
                                instrument.security_id
                            ),
                            exchange_segment=(
                                instrument.exchange_segment
                            ),
                            instrument_type=(
                                instrument.instrument_type
                            ),
                            from_date=(
                                start_date.isoformat()
                            ),
                            to_date=(
                                end_date.isoformat()
                            ),
                            interval=15,
                        )
                    )

                if not response:
                    raise RuntimeError(
                        "Empty Dhan response"
                    )

                return response

            except Exception as exc:

                if (
                    is_transient_error(
                        exc
                    )
                    and attempt < attempts
                ):

                    delay = (
                        0.75
                        * (
                            2
                            ** (
                                attempt - 1
                            )
                        )
                    )

                    LOGGER.warning(
                        "%s transient API "
                        "error attempt %d/%d: %s",
                        getattr(
                            instrument,
                            "symbol",
                            getattr(
                                instrument,
                                "name",
                                "INDEX",
                            ),
                        ),
                        attempt,
                        attempts,
                        exc,
                    )

                    time.sleep(
                        delay
                    )

                    continue

                raise


def is_transient_error(
    exc: Exception,
) -> bool:

    message = str(
        exc
    ).lower()

    indicators = (
        "429",
        "rate limit",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "502",
        "503",
        "504",
    )

    return any(
        item in message
        for item in indicators
    )


# =============================================================================
# DATA NORMALIZATION
# =============================================================================

def clean_ticker_data(
    raw: dict[str, Any],
) -> pd.DataFrame:

    if not isinstance(
        raw,
        dict,
    ):
        raise TypeError(
            "Dhan response must be a dictionary"
        )

    def get(
        *names: str,
    ) -> Any:

        for name in names:

            if name in raw:
                return raw[name]

        return None

    opens = get(
        "open",
        "Open",
    )

    highs = get(
        "high",
        "High",
    )

    lows = get(
        "low",
        "Low",
    )

    closes = get(
        "close",
        "Close",
    )

    volumes = get(
        "volume",
        "Volume",
    )

    timestamps = get(
        "start_time",
        "start_Time",
        "startTime",
        "timestamp",
        "Timestamp",
    )

    fields = (
        opens,
        highs,
        lows,
        closes,
        volumes,
        timestamps,
    )

    if any(
        field is None
        for field in fields
    ):
        raise ValueError(
            "Incomplete Dhan response. "
            f"Keys={list(raw.keys())}"
        )

    lengths = {
        len(field)
        for field in fields
    }

    if len(lengths) != 1:
        raise ValueError(
            "Inconsistent Dhan array lengths"
        )

    df = pd.DataFrame(
        {
            "Open": pd.to_numeric(
                opens,
                errors="coerce",
            ),
            "High": pd.to_numeric(
                highs,
                errors="coerce",
            ),
            "Low": pd.to_numeric(
                lows,
                errors="coerce",
            ),
            "Close": pd.to_numeric(
                closes,
                errors="coerce",
            ),
            "Volume": pd.to_numeric(
                volumes,
                errors="coerce",
            ),
            "Timestamp": [
                epoch_to_ist(
                    timestamp
                )
                for timestamp in timestamps
            ],
        }
    )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Timestamp",
        ]
    )

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    df = df[
        (
            df[
                price_columns
            ] > 0
        ).all(axis=1)
        & (
            df["Volume"] >= 0
        )
    ]

    df = (
        df
        .drop_duplicates(
            subset=["Timestamp"]
        )
        .set_index("Timestamp")
        .sort_index()
    )

    return df


# =============================================================================
# INDICATORS
# =============================================================================

def add_indicators(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    close = result[
        "Close"
    ]

    high = result[
        "High"
    ]

    low = result[
        "Low"
    ]

    # -------------------------------------------------------------------------
    # EMA 9
    # -------------------------------------------------------------------------

    result["EMA9"] = close.ewm(
        alpha=2.0 / 10.0,
        adjust=False,
        min_periods=1,
    ).mean()

    # -------------------------------------------------------------------------
    # EMA 20
    # -------------------------------------------------------------------------

    result["EMA20"] = close.ewm(
        alpha=2.0 / 21.0,
        adjust=False,
        min_periods=1,
    ).mean()

    # -------------------------------------------------------------------------
    # RSI 14 — Wilder smoothing.
    # -------------------------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=1,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=1,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan,
        )
    )

    result["RSI14"] = (
        100.0
        - (
            100.0
            / (
                1.0 + rs
            )
        )
    )

    # Explicit division-by-zero handling.
    result.loc[
        avg_loss == 0,
        "RSI14",
    ] = 100.0

    # -------------------------------------------------------------------------
    # ATR 14 — Wilder smoothing.
    # -------------------------------------------------------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (
                high
                - previous_close
            ).abs(),
            (
                low
                - previous_close
            ).abs(),
        ],
        axis=1,
    ).max(
        axis=1
    )

    result["TR"] = (
        true_range
    )

    result["ATR14"] = (
        true_range
        .ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=1,
        )
        .mean()
    )

    result["ATR_Pct"] = (
        result["ATR14"]
        / close
        * 100.0
    )

    # -------------------------------------------------------------------------
    # Session VWAP.
    #
    # Reset on every IST trading date.
    # -------------------------------------------------------------------------

    typical_price = (
        high
        + low
        + close
    ) / 3.0

    session_key = pd.Series(
        result.index.date,
        index=result.index,
    )

    tpv = (
        typical_price
        * result["Volume"]
    )

    cumulative_tpv = (
        tpv
        .groupby(session_key)
        .cumsum()
    )

    cumulative_volume = (
        result["Volume"]
        .groupby(session_key)
        .cumsum()
    )

    result["VWAP"] = (
        cumulative_tpv
        / cumulative_volume.replace(
            0,
            np.nan,
        )
    )

    return result


# =============================================================================
# SESSION HELPERS
# =============================================================================

def get_session(
    df: pd.DataFrame,
    target_date: date,
) -> pd.DataFrame:

    mask = (
        (df.index.date == target_date)
        & (
            df.index.time
            >= MARKET_OPEN
        )
        & (
            df.index.time
            < DECISION_TIME
        )
    )

    return df.loc[
        mask
    ].copy()


def get_first_two_candles(
    df: pd.DataFrame,
    target_date: date,
) -> Optional[pd.DataFrame]:

    session = get_session(
        df,
        target_date,
    )

    if len(session) < 2:
        return None

    return session.iloc[
        :2
    ].copy()


def get_previous_close(
    df: pd.DataFrame,
    target_date: date,
) -> Optional[float]:

    historical = df[
        df.index.date
        < target_date
    ]

    if historical.empty:
        return None

    return float(
        historical.iloc[-1][
            "Close"
        ]
    )


def calculate_gap(
    df: pd.DataFrame,
    target_date: date,
) -> Optional[float]:

    candles = get_first_two_candles(
        df,
        target_date,
    )

    if candles is None:
        return None

    today_open = float(
        candles.iloc[0][
            "Open"
        ]
    )

    previous_close = (
        get_previous_close(
            df,
            target_date,
        )
    )

    if (
        previous_close is None
        or previous_close <= 0
    ):
        return None

    return (
        (
            today_open
            - previous_close
        )
        / previous_close
        * 100.0
    )


# =============================================================================
# SECTOR MAP
# =============================================================================

SECTOR_MAP: dict[str, str] = {

    # IT
    "TCS": "NIFTY IT",
    "INFY": "NIFTY IT",
    "HCLTECH": "NIFTY IT",
    "WIPRO": "NIFTY IT",
    "TECHM": "NIFTY IT",
    "LTIM": "NIFTY IT",
    "MPHASIS": "NIFTY IT",
    "PERSISTENT": "NIFTY IT",
    "COFORGE": "NIFTY IT",

    # BANK
    "SBIN": "NIFTY BANK",
    "HDFCBANK": "NIFTY BANK",
    "ICICIBANK": "NIFTY BANK",
    "AXISBANK": "NIFTY BANK",
    "KOTAKBANK": "NIFTY BANK",
    "INDUSINDBK": "NIFTY BANK",
    "BANKBARODA": "NIFTY BANK",
    "PNB": "NIFTY BANK",

    # ENERGY
    "RELIANCE": "NIFTY ENERGY",
    "ONGC": "NIFTY ENERGY",
    "IOC": "NIFTY ENERGY",
    "BPCL": "NIFTY ENERGY",
    "GAIL": "NIFTY ENERGY",

    # AUTO
    "MARUTI": "NIFTY AUTO",
    "TATAMOTORS": "NIFTY AUTO",
    "M&M": "NIFTY AUTO",
    "EICHERMOT": "NIFTY AUTO",
    "HEROMOTOCO": "NIFTY AUTO",
    "BAJAJ-AUTO": "NIFTY AUTO",
    "TVSMOTOR": "NIFTY AUTO",
    "ASHOKLEY": "NIFTY AUTO",

    # PHARMA
    "SUNPHARMA": "NIFTY PHARMA",
    "DRREDDY": "NIFTY PHARMA",
    "CIPLA": "NIFTY PHARMA",
    "DIVISLAB": "NIFTY PHARMA",
    "APOLLOHOSP": "NIFTY PHARMA",
    "LUPIN": "NIFTY PHARMA",

    # FMCG
    "HINDUNILVR": "NIFTY FMCG",
    "ITC": "NIFTY FMCG",
    "NESTLEIND": "NIFTY FMCG",
    "BRITANNIA": "NIFTY FMCG",
    "DABUR": "NIFTY FMCG",
    "MARICO": "NIFTY FMCG",

    # METAL
    "TATASTEEL": "NIFTY METAL",
    "HINDALCO": "NIFTY METAL",
    "JSWSTEEL": "NIFTY METAL",
    "VEDL": "NIFTY METAL",
    "JINDALSTEL": "NIFTY METAL",
}


# =============================================================================
# SECTOR CALCULATION
# =============================================================================

def calculate_sector_total(
    df: pd.DataFrame,
    target_date: date,
) -> Optional[float]:

    session = get_session(
        df,
        target_date,
    )

    if len(session) < 2:
        return None

    first_open = float(
        session.iloc[0][
            "Open"
        ]
    )

    second_close = float(
        session.iloc[1][
            "Close"
        ]
    )

    if first_open <= 0:
        return None

    return (
        (
            second_close
            - first_open
        )
        / first_open
        * 100.0
    )


# =============================================================================
# NIFTY 50 MOVEMENT
# =============================================================================

def calculate_nifty_total(
    df: pd.DataFrame,
    target_date: date,
) -> Optional[float]:

    session = get_first_two_candles(
        df,
        target_date,
    )

    if session is None:
        return None

    first_open = float(
        session.iloc[0][
            "Open"
        ]
    )

    second_close = float(
        session.iloc[1][
            "Close"
        ]
    )

    if first_open <= 0:
        return None

    return (
        (
            second_close
            - first_open
        )
        / first_open
        * 100.0
    )


# =============================================================================
# SCORING ENGINE
# =============================================================================

def calculate_scores(
    *,
    opening_pct: float,
    confirmation_pct: float,
    total_pct: float,
    nifty_total_pct: float,
    close: float,
    vwap: float,
    ema9: float,
    ema20: float,
    rsi: float,
    gap_pct: float,
    sector_relative_strength: float,
    config: ScoringConfig,
) -> tuple[float, float]:

    short_score = 50.0
    long_score = 50.0

    # -------------------------------------------------------------------------
    # Directional momentum.
    # -------------------------------------------------------------------------

    opening_points = (
        continuous_points(
            opening_pct,
            config.directional_full_move_pct,
            config.directional_max_points / 2.0,
        )
    )

    confirmation_points = (
        continuous_points(
            confirmation_pct,
            config.directional_full_move_pct,
            config.directional_max_points / 2.0,
        )
    )

    if opening_pct < 0:
        short_score += (
            opening_points
        )

    elif opening_pct > 0:
        long_score += (
            opening_points
        )

    if confirmation_pct < 0:
        short_score += (
            confirmation_points
        )

    elif confirmation_pct > 0:
        long_score += (
            confirmation_points
        )

    # -------------------------------------------------------------------------
    # Index alpha.
    # -------------------------------------------------------------------------

    index_alpha = (
        total_pct
        - nifty_total_pct
    )

    alpha_points = (
        continuous_points(
            index_alpha,
            config.index_alpha_full_move_pct,
            config.index_alpha_max_points,
        )
    )

    if index_alpha < 0:
        short_score += (
            alpha_points
        )

    elif index_alpha > 0:
        long_score += (
            alpha_points
        )

    # -------------------------------------------------------------------------
    # VWAP.
    # -------------------------------------------------------------------------

    if close < vwap:
        short_score += (
            config.vwap_points
        )

    elif close > vwap:
        long_score += (
            config.vwap_points
        )

    # -------------------------------------------------------------------------
    # EMA structure.
    # -------------------------------------------------------------------------

    bearish_ema = (
        close < ema20
        and ema9 < ema20
    )

    bullish_ema = (
        close > ema20
        and ema9 > ema20
    )

    if bearish_ema:
        short_score += (
            config.ema_points
        )

    if bullish_ema:
        long_score += (
            config.ema_points
        )

    # -------------------------------------------------------------------------
    # RSI fatigue.
    # -------------------------------------------------------------------------

    if rsi < 22:
        short_score -= (
            config.rsi_exhaustion_penalty
        )

    elif 32 <= rsi <= 55:
        short_score += (
            config.rsi_sustainable_bonus
        )

    if rsi > 78:
        long_score -= (
            config.rsi_exhaustion_penalty
        )

    elif 45 <= rsi <= 68:
        long_score += (
            config.rsi_sustainable_bonus
        )

    # -------------------------------------------------------------------------
    # Pre-market gap fading.
    # -------------------------------------------------------------------------

    if (
        gap_pct > 2.0
        and total_pct < 0
    ):
        short_score += (
            config.gap_fade_bonus
        )

    if (
        gap_pct < -2.0
        and total_pct > 0
    ):
        long_score += (
            config.gap_fade_bonus
        )

    # -------------------------------------------------------------------------
    # Sector relative strength.
    # -------------------------------------------------------------------------

    if (
        sector_relative_strength
        < -0.5
    ):
        short_score += (
            config.sector_weakness_bonus
        )

    if (
        sector_relative_strength
        > 0.5
    ):
        long_score += (
            config.sector_strength_bonus
        )

    return (
        clamp(
            short_score,
            0,
            100,
        ),
        clamp(
            long_score,
            0,
            100,
        ),
    )


# =============================================================================
# SINGLE STOCK ANALYSIS
# =============================================================================

class StrategyEngine:

    def __init__(
        self,
        config: Optional[
            ScoringConfig
        ] = None,
    ):

        self.config = (
            config
            or ScoringConfig()
        )

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        target_date: date,
        nifty_total_pct: float,
        sector_total_pct: Optional[float],
    ) -> Optional[SignalResult]:

        # Calculate all indicators over historical warmup + target date.
        enriched = add_indicators(
            df
        )

        candles = (
            get_first_two_candles(
                enriched,
                target_date,
            )
        )

        if candles is None:
            return None

        candle1 = candles.iloc[0]
        candle2 = candles.iloc[1]

        open1 = float(
            candle1["Open"]
        )

        close1 = float(
            candle1["Close"]
        )

        open2 = float(
            candle2["Open"]
        )

        close2 = float(
            candle2["Close"]
        )

        volume2 = float(
            candle2["Volume"]
        )

        if open1 <= 0 or open2 <= 0:
            return None

        # ---------------------------------------------------------------------
        # Movement.
        # ---------------------------------------------------------------------

        opening_pct = (
            (
                close1
                - open1
            )
            / open1
            * 100.0
        )

        confirmation_pct = (
            (
                close2
                - open2
            )
            / open2
            * 100.0
        )

        total_pct = (
            (
                close2
                - open1
            )
            / open1
            * 100.0
        )

        # ---------------------------------------------------------------------
        # Liquidity.
        # ---------------------------------------------------------------------

        turnover_crore = (
            volume2
            * close2
            / 10_000_000.0
        )

        liquidity_pass = (
            turnover_crore >= 2.0
        )

        if not liquidity_pass:
            return SignalResult(
                symbol=clean_symbol(symbol),
                signal="NO_TRADE",
                conviction="NONE",
                structural_score=0.0,
                short_score=0.0,
                long_score=0.0,
                premarket_gap_pct=(
                    calculate_gap(
                        enriched,
                        target_date,
                    )
                    or 0.0
                ),
                sector=(
                    next(
                        (
                            sector
                            for stock, sector
                            in SECTOR_MAP.items()
                            if stock
                            == clean_symbol(
                                symbol
                            )
                        ),
                        "UNMAPPED",
                    )
                ),
                sector_total_pct=(
                    sector_total_pct
                ),
                sector_relative_strength=(
                    0.0
                ),
                opening_pct=(
                    opening_pct
                ),
                confirmation_pct=(
                    confirmation_pct
                ),
                total_pct=total_pct,
                nifty_total_pct=(
                    nifty_total_pct
                ),
                index_alpha_pct=(
                    total_pct
                    - nifty_total_pct
                ),
                turnover_crore=(
                    turnover_crore
                ),
                execution_price=(
                    close2
                ),
                vwap=float(
                    enriched.loc[
                        candles.index[-1],
                        "VWAP",
                    ]
                ),
                ema9=float(
                    enriched.loc[
                        candles.index[-1],
                        "EMA9",
                    ]
                ),
                ema20=float(
                    enriched.loc[
                        candles.index[-1],
                        "EMA20",
                    ]
                ),
                rsi14=float(
                    enriched.loc[
                        candles.index[-1],
                        "RSI14",
                    ]
                ),
                atr_pct=float(
                    enriched.loc[
                        candles.index[-1],
                        "ATR_Pct",
                    ]
                ),
                liquidity_pass=False,
                short_criteria_pass=False,
                long_criteria_pass=False,
                decision_timestamp=(
                    candles.index[-1]
                    .isoformat()
                ),
            )

        # ---------------------------------------------------------------------
        # Decision-point indicator values.
        # ---------------------------------------------------------------------

        decision_row = enriched.loc[
            candles.index[-1]
        ]

        close = float(
            decision_row["Close"]
        )

        vwap = float(
            decision_row["VWAP"]
        )

        ema9 = float(
            decision_row["EMA9"]
        )

        ema20 = float(
            decision_row["EMA20"]
        )

        rsi = float(
            decision_row["RSI14"]
        )

        atr_pct = float(
            decision_row["ATR_Pct"]
        )

        # ---------------------------------------------------------------------
        # Gap.
        # ---------------------------------------------------------------------

        gap_pct = calculate_gap(
            enriched,
            target_date,
        )

        if gap_pct is None:
            return None

        # ---------------------------------------------------------------------
        # Sector.
        # ---------------------------------------------------------------------

        clean = clean_symbol(
            symbol
        )

        sector = SECTOR_MAP.get(
            clean,
            "UNMAPPED",
        )

        if sector_total_pct is None:
            sector_relative_strength = 0.0

        else:
            sector_relative_strength = (
                total_pct
                - sector_total_pct
            )

        # ---------------------------------------------------------------------
        # Score.
        # ---------------------------------------------------------------------

        short_score, long_score = (
            calculate_scores(
                opening_pct=opening_pct,
                confirmation_pct=confirmation_pct,
                total_pct=total_pct,
                nifty_total_pct=nifty_total_pct,
                close=close,
                vwap=vwap,
                ema9=ema9,
                ema20=ema20,
                rsi=rsi,
                gap_pct=gap_pct,
                sector_relative_strength=(
                    sector_relative_strength
                ),
                config=self.config,
            )
        )

        # ---------------------------------------------------------------------
        # Signal criteria.
        # ---------------------------------------------------------------------

        short_pass = (
            short_score >= 60.0
            and short_score >= long_score
            and total_pct <= -0.30
            and opening_pct <= -0.25
        )

        long_pass = (
            long_score >= 70.0
            and long_score > short_score
            and total_pct >= 0.50
            and opening_pct >= 0.35
        )

        signal = "NO_TRADE"
        conviction = "NONE"

        if short_pass:

            signal = "SHORT"

            conviction = (
                "HIGH"
                if short_score >= 75
                else "TRADEABLE"
            )

        elif long_pass:

            signal = "LONG"

            conviction = (
                "HIGH"
                if long_score >= 80
                else "TRADEABLE"
            )

        if signal == "SHORT":
            structural_score = (
                short_score
            )

        elif signal == "LONG":
            structural_score = (
                long_score
            )

        else:
            structural_score = max(
                short_score,
                long_score,
            )

        return SignalResult(
            symbol=clean,
            signal=signal,
            conviction=conviction,
            structural_score=round(
                structural_score,
                4,
            ),
            short_score=round(
                short_score,
                4,
            ),
            long_score=round(
                long_score,
                4,
            ),
            premarket_gap_pct=round(
                gap_pct,
                4,
            ),
            sector=sector,
            sector_total_pct=(
                round(
                    sector_total_pct,
                    4,
                )
                if sector_total_pct
                is not None
                else None
            ),
            sector_relative_strength=round(
                sector_relative_strength,
                4,
            ),
            opening_pct=round(
                opening_pct,
                4,
            ),
            confirmation_pct=round(
                confirmation_pct,
                4,
            ),
            total_pct=round(
                total_pct,
                4,
            ),
            nifty_total_pct=round(
                nifty_total_pct,
                4,
            ),
            index_alpha_pct=round(
                total_pct
                - nifty_total_pct,
                4,
            ),
            turnover_crore=round(
                turnover_crore,
                4,
            ),
            execution_price=round(
                close,
                4,
            ),
            vwap=round(
                vwap,
                4,
            ),
            ema9=round(
                ema9,
                4,
            ),
            ema20=round(
                ema20,
                4,
            ),
            rsi14=round(
                rsi,
                4,
            ),
            atr_pct=round(
                atr_pct,
                4,
            ),
            liquidity_pass=True,
            short_criteria_pass=(
                short_pass
            ),
            long_criteria_pass=(
                long_pass
            ),
            decision_timestamp=(
                candles.index[-1]
                .isoformat()
            ),
        )


# =============================================================================
# DATA CACHE
# =============================================================================

class HistoricalDataCache:

    def __init__(
        self,
        data_engine: DhanDataEngine,
    ):

        self.engine = data_engine

    @staticmethod
    def _path(
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> Path:

        safe = (
            clean_symbol(symbol)
            .replace("/", "_")
        )

        return (
            CACHE_DIR
            / (
                f"{safe}_"
                f"{start_date.isoformat()}_"
                f"{end_date.isoformat()}.parquet"
            )
        )

    def get_equity(
        self,
        instrument: EquityInstrument,
        start_date: date,
        end_date: date,
        force_refresh: bool = False,
    ) -> pd.DataFrame:

        path = self._path(
            instrument.symbol,
            start_date,
            end_date,
        )

        if (
            path.exists()
            and not force_refresh
        ):

            try:

                return pd.read_parquet(
                    path
                )

            except Exception as exc:

                LOGGER.warning(
                    "Cache read failed %s: %s",
                    path,
                    exc,
                )

        df = self.engine.fetch_intraday(
            instrument,
            start_date,
            end_date,
        )

        df.to_parquet(
            path
        )

        return df


# =============================================================================
# CONCURRENT UNIVERSE DOWNLOAD
# =============================================================================

def download_batch(
    engine: DhanDataEngine,
    instruments: list[EquityInstrument],
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:

    results: dict[
        str,
        pd.DataFrame,
    ] = {}

    def worker(
        instrument: EquityInstrument,
    ) -> tuple[
        str,
        Optional[pd.DataFrame],
    ]:

        try:

            df = engine.fetch_intraday(
                instrument,
                start_date,
                end_date,
            )

            if df.empty:
                return (
                    instrument.symbol,
                    None,
                )

            return (
                instrument.symbol,
                df,
            )

        except Exception as exc:

            LOGGER.error(
                "%s skipped: %s",
                instrument.symbol,
                exc,
            )

            return (
                instrument.symbol,
                None,
            )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS,
        thread_name_prefix="dhan",
    ) as executor:

        futures = {
            executor.submit(
                worker,
                instrument,
            ): instrument
            for instrument in instruments
        }

        for future in as_completed(
            futures
        ):

            instrument = futures[
                future
            ]

            try:

                symbol, df = (
                    future.result()
                )

                if (
                    df is not None
                    and not df.empty
                ):
                    results[
                        symbol
                    ] = df

            except Exception:

                LOGGER.exception(
                    "Worker failure: %s",
                    instrument.symbol,
                )

    LOGGER.info(
        "Downloaded %d/%d instruments",
        len(results),
        len(instruments),
    )

    return results


# =============================================================================
# RESULT TABLE
# =============================================================================

def results_to_dataframe(
    results: list[SignalResult],
) -> pd.DataFrame:

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            asdict(result)
            for result in results
        ]
    )


# =============================================================================
# TOP 5 + TOP 5 SELECTION
# =============================================================================

def select_candidates(
    results: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if results.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    shorts = (
        results[
            results["signal"]
            == "SHORT"
        ]
        .sort_values(
            [
                "structural_score",
                "short_score",
            ],
            ascending=False,
        )
        .head(5)
        .reset_index(drop=True)
    )

    longs = (
        results[
            results["signal"]
            == "LONG"
        ]
        .sort_values(
            [
                "structural_score",
                "long_score",
            ],
            ascending=False,
        )
        .head(5)
        .reset_index(drop=True)
    )

    return (
        shorts,
        longs,
    )


# =============================================================================
# CONSOLE PRESENTATION
# =============================================================================

def print_candidate_table(
    title: str,
    candidates: pd.DataFrame,
) -> None:

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    if candidates.empty:

        print(
            "No qualifying candidates."
        )

        return

    print(
        f"QUALIFIED: {len(candidates)}/5"
    )

    columns = [
        "symbol",
        "structural_score",
        "conviction",
        "premarket_gap_pct",
        "total_pct",
        "sector",
        "sector_relative_strength",
        "execution_price",
    ]

    display = candidates[
        columns
    ].copy()

    display.columns = [
        "SYMBOL",
        "SCORE",
        "CONVICTION",
        "GAP%",
        "TOTAL%",
        "SECTOR",
        "SECTOR RS%",
        "PRICE",
    ]

    print(
        display.to_string(
            index=False
        )
    )


def print_single_analysis(
    result: Optional[SignalResult],
) -> None:

    if result is None:

        print(
            "\nNo valid 09:45 setup "
            "could be constructed for "
            "this stock/date."
        )

        return

    print()
    print("=" * 100)
    print(
        f"{result.symbol} | 09:45 ANALYSIS"
    )
    print("=" * 100)

    print(
        f"Signal             : {result.signal}"
    )

    print(
        f"Conviction         : {result.conviction}"
    )

    print(
        f"Structural Score   : "
        f"{result.structural_score:.2f}"
    )

    print()

    print("MOVEMENT")
    print("-" * 100)

    print(
        f"Opening %          : "
        f"{result.opening_pct:.3f}%"
    )

    print(
        f"Confirmation %     : "
        f"{result.confirmation_pct:.3f}%"
    )

    print(
        f"Total %            : "
        f"{result.total_pct:.3f}%"
    )

    print()

    print("LIQUIDITY")
    print("-" * 100)

    print(
        f"Turnover           : "
        f"₹{result.turnover_crore:.2f} Cr"
    )

    print(
        f"Liquidity hurdle   : "
        f"{'PASS' if result.liquidity_pass else 'FAIL'}"
    )

    print()

    print("TECHNICAL STRUCTURE")
    print("-" * 100)

    print(
        f"Price              : "
        f"₹{result.execution_price:.2f}"
    )

    print(
        f"VWAP               : "
        f"₹{result.vwap:.2f}"
    )

    print(
        f"EMA9               : "
        f"{result.ema9:.2f}"
    )

    print(
        f"EMA20              : "
        f"{result.ema20:.2f}"
    )

    print(
        f"RSI14              : "
        f"{result.rsi14:.2f}"
    )

    print(
        f"ATR %              : "
        f"{result.atr_pct:.3f}%"
    )

    print()

    print("RELATIVE STRENGTH")
    print("-" * 100)

    print(
        f"NIFTY 50 Total %   : "
        f"{result.nifty_total_pct:.3f}%"
    )

    print(
        f"Index Alpha         : "
        f"{result.index_alpha_pct:.3f}%"
    )

    print(
        f"Sector              : "
        f"{result.sector}"
    )

    if result.sector_total_pct is None:

        print(
            "Sector Total %     : UNAVAILABLE"
        )

    else:

        print(
            f"Sector Total %     : "
            f"{result.sector_total_pct:.3f}%"
        )

    print(
        f"Sector Relative %   : "
        f"{result.sector_relative_strength:.3f}%"
    )

    print()

    print("GAP")
    print("-" * 100)

    print(
        f"Pre-market Gap      : "
        f"{result.premarket_gap_pct:.3f}%"
    )

    print()

    print("SCORES")
    print("-" * 100)

    print(
        f"SHORT Score         : "
        f"{result.short_score:.2f}"
    )

    print(
        f"LONG Score          : "
        f"{result.long_score:.2f}"
    )

    print()

    print("DECISION")
    print("-" * 100)

    print(
        f"SHORT criteria      : "
        f"{'PASS' if result.short_criteria_pass else 'FAIL'}"
    )

    print(
        f"LONG criteria       : "
        f"{'PASS' if result.long_criteria_pass else 'FAIL'}"
    )

    print(
        f"FINAL SIGNAL        : "
        f"{result.signal}"
    )

    print("=" * 100)


# =============================================================================
# RESEARCH APPLICATION
# =============================================================================

class ResearchApplication:

    def __init__(self):

        self.session = (
            requests.Session()
        )

        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "IntradayMomentumResearch/2.0"
                )
            }
        )

        client_id = os.getenv(
            "DHAN_CLIENT_ID"
        )

        access_token = os.getenv(
            "DHAN_ACCESS_TOKEN"
        )

        if not client_id:
            raise RuntimeError(
                "DHAN_CLIENT_ID missing"
            )

        if not access_token:
            raise RuntimeError(
                "DHAN_ACCESS_TOKEN missing"
            )

        self.data_engine = (
            DhanDataEngine(
                client_id,
                access_token,
            )
        )

        self.master = (
            DhanInstrumentMaster(
                self.session
            )
        )

        self.master.load()

        self.universe_loader = (
            NSEUniverse(
                self.session
            )
        )

        self.strategy = (
            StrategyEngine()
        )

        self._instrument_cache: dict[
            str,
            EquityInstrument,
        ] = {}

        self._index_cache: dict[
            str,
            IndexInstrument,
        ] = {}

    # =========================================================================
    # INSTRUMENT RESOLUTION
    # =========================================================================

    def resolve_equity(
        self,
        symbol: str,
    ) -> EquityInstrument:

        symbol = clean_symbol(
            symbol
        )

        if symbol not in (
            self._instrument_cache
        ):

            self._instrument_cache[
                symbol
            ] = (
                self.master.resolve_equity(
                    symbol
                )
            )

        return self._instrument_cache[
            symbol
        ]

    def resolve_index(
        self,
        name: str,
    ) -> IndexInstrument:

        if name not in (
            self._index_cache
        ):

            self._index_cache[
                name
            ] = (
                self.master.resolve_index(
                    name
                )
            )

        return self._index_cache[
            name
        ]

    # =========================================================================
    # INDEX DATA
    # =========================================================================

    def load_nifty(
        self,
        target_date: date,
    ) -> pd.DataFrame:

        instrument = (
            self.resolve_index(
                "NIFTY 50"
            )
        )

        start = (
            target_date
            - timedelta(
                days=WARMUP_DAYS
            )
        )

        end = (
            target_date
            + timedelta(days=1)
        )

        return self.data_engine.fetch_intraday(
            instrument,
            start,
            end,
        )

    def load_sector_data(
        self,
        target_date: date,
    ) -> dict[
        str,
        pd.DataFrame,
    ]:

        sector_names = sorted(
            set(
                SECTOR_MAP.values()
            )
        )

        result = {}

        start = (
            target_date
            - timedelta(
                days=WARMUP_DAYS
            )
        )

        end = (
            target_date
            + timedelta(days=1)
        )

        for sector_name in (
            sector_names
        ):

            try:

                instrument = (
                    self.resolve_index(
                        sector_name
                    )
                )

                result[
                    sector_name
                ] = (
                    self.data_engine
                    .fetch_intraday(
                        instrument,
                        start,
                        end,
                    )
                )

            except Exception as exc:

                LOGGER.warning(
                    "Sector %s unavailable: %s",
                    sector_name,
                    exc,
                )

        return result

    # =========================================================================
    # DATE CONTEXT
    # =========================================================================

    def build_date_context(
        self,
        target_date: date,
    ) -> tuple[
        float,
        dict[str, Optional[float]],
    ]:

        nifty_df = (
            self.load_nifty(
                target_date
            )
        )

        nifty_total = (
            calculate_nifty_total(
                nifty_df,
                target_date,
            )
        )

        if nifty_total is None:
            raise RuntimeError(
                f"NIFTY 50 has insufficient "
                f"data for {target_date}"
            )

        sector_data = (
            self.load_sector_data(
                target_date
            )
        )

        sector_totals = {
            sector: (
                calculate_sector_total(
                    df,
                    target_date,
                )
            )
            for sector, df
            in sector_data.items()
        }

        return (
            nifty_total,
            sector_totals,
        )

    # =========================================================================
    # SINGLE STOCK
    # =========================================================================

    def analyze_stock(
        self,
        symbol: str,
        target_date: date,
    ) -> Optional[SignalResult]:

        symbol = clean_symbol(
            symbol
        )

        LOGGER.info(
            "Analyzing %s for %s",
            symbol,
            target_date,
        )

        instrument = (
            self.resolve_equity(
                symbol
            )
        )

        start = (
            target_date
            - timedelta(
                days=WARMUP_DAYS
            )
        )

        end = (
            target_date
            + timedelta(days=1)
        )

        df = (
            self.data_engine
            .fetch_intraday(
                instrument,
                start,
                end,
            )
        )

        nifty_total, sector_totals = (
            self.build_date_context(
                target_date
            )
        )

        sector = SECTOR_MAP.get(
            symbol
        )

        sector_total = (
            sector_totals.get(
                sector
            )
            if sector
            else None
        )

        return self.strategy.analyze(
            symbol=symbol,
            df=df,
            target_date=target_date,
            nifty_total_pct=nifty_total,
            sector_total_pct=sector_total,
        )

    # =========================================================================
    # FULL DATE SCAN
    # =========================================================================

    def scan_date(
        self,
        target_date: date,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:

        LOGGER.info(
            "Starting full scan: %s",
            target_date,
        )

        universe = (
            self.universe_loader
            .load_universe()
        )

        instruments = []

        for symbol in universe:

            try:

                instruments.append(
                    self.resolve_equity(
                        symbol
                    )
                )

            except Exception as exc:

                LOGGER.warning(
                    "Unable to resolve %s: %s",
                    symbol,
                    exc,
                )

        nifty_total, sector_totals = (
            self.build_date_context(
                target_date
            )
        )

        start = (
            target_date
            - timedelta(
                days=WARMUP_DAYS
            )
        )

        end = (
            target_date
            + timedelta(days=1)
        )

        download_start = (
            time.perf_counter()
        )

        raw_data = (
            download_batch(
                self.data_engine,
                instruments,
                start,
                end,
            )
        )

        LOGGER.info(
            "Market data download: %.3fs",
            time.perf_counter()
            - download_start,
        )

        analyzed: list[
            SignalResult
        ] = []

        for instrument in (
            instruments
        ):

            df = raw_data.get(
                instrument.symbol
            )

            if df is None:
                continue

            sector = SECTOR_MAP.get(
                instrument.symbol
            )

            sector_total = (
                sector_totals.get(
                    sector
                )
                if sector
                else None
            )

            try:

                result = (
                    self.strategy.analyze(
                        symbol=(
                            instrument.symbol
                        ),
                        df=df,
                        target_date=(
                            target_date
                        ),
                        nifty_total_pct=(
                            nifty_total
                        ),
                        sector_total_pct=(
                            sector_total
                        ),
                    )
                )

                if result is not None:
                    analyzed.append(
                        result
                    )

            except Exception:

                LOGGER.exception(
                    "Analysis failed: %s",
                    instrument.symbol,
                )

        all_results = (
            results_to_dataframe(
                analyzed
            )
        )

        shorts, longs = (
            select_candidates(
                all_results
            )
        )

        return (
            all_results,
            shorts,
            longs,
        )


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:

    def __init__(
        self,
        app: ResearchApplication,
    ):

        self.app = app

    @staticmethod
    def trading_days(
        start_date: date,
        end_date: date,
    ) -> list[date]:

        dates = pd.bdate_range(
            start=start_date,
            end=end_date,
        )

        return [
            timestamp.date()
            for timestamp in dates
        ]

    def run(
        self,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:

        all_rows = []

        days = self.trading_days(
            start_date,
            end_date,
        )

        LOGGER.info(
            "Backtest sessions: %d",
            len(days),
        )

        for index, target_date in (
            enumerate(days, start=1)
        ):

            LOGGER.info(
                "BACKTEST %d/%d: %s",
                index,
                len(days),
                target_date,
            )

            try:

                _, shorts, longs = (
                    self.app.scan_date(
                        target_date
                    )
                )

                if not shorts.empty:

                    short_rows = (
                        shorts.copy()
                    )

                    short_rows[
                        "backtest_date"
                    ] = target_date

                    all_rows.append(
                        short_rows
                    )

                if not longs.empty:

                    long_rows = (
                        longs.copy()
                    )

                    long_rows[
                        "backtest_date"
                    ] = target_date

                    all_rows.append(
                        long_rows
                    )

            except Exception as exc:

                LOGGER.error(
                    "Backtest date %s failed: %s",
                    target_date,
                    exc,
                )

        if not all_rows:
            return pd.DataFrame()

        result = pd.concat(
            all_rows,
            ignore_index=True,
        )

        output = (
            OUTPUT_DIR
            / (
                "backtest_"
                f"{start_date}_"
                f"{end_date}.csv"
            )
        )

        result.to_csv(
            output,
            index=False,
        )

        LOGGER.info(
            "Backtest saved: %s",
            output,
        )

        return result


# =============================================================================
# DATE PARSING
# =============================================================================

def parse_date(
    value: str,
) -> date:

    return date.fromisoformat(
        value.strip()
    )


def current_ist_date() -> date:

    return datetime.now(
        IST
    ).date()


# =============================================================================
# CLI
# =============================================================================

def print_menu() -> None:

    print()
    print("=" * 70)
    print(
        "INTRADAY MOMENTUM SCANNER V2"
    )
    print("=" * 70)
    print(
        "1. Current Day"
    )
    print(
        "2. Historical Date"
    )
    print(
        "3. Analyze Single Stock"
    )
    print(
        "4. Backtest"
    )
    print(
        "5. Exit"
    )
    print("=" * 70)


def run_current_day(
    app: ResearchApplication,
) -> None:

    target_date = (
        current_ist_date()
    )

    print(
        f"\nScanning current date: "
        f"{target_date}"
    )

    all_results, shorts, longs = (
        app.scan_date(
            target_date
        )
    )

    print_candidate_table(
        "TOP 5 SHORT CANDIDATES",
        shorts,
    )

    print_candidate_table(
        "TOP 5 LONG CANDIDATES",
        longs,
    )

    output = (
        OUTPUT_DIR
        / (
            f"scan_{target_date}.csv"
        )
    )

    all_results.to_csv(
        output,
        index=False,
    )

    print(
        f"\nFull result set saved to: "
        f"{output}"
    )


def run_historical(
    app: ResearchApplication,
) -> None:

    value = input(
        "\nEnter historical date "
        "(YYYY-MM-DD): "
    )

    target_date = parse_date(
        value
    )

    all_results, shorts, longs = (
        app.scan_date(
            target_date
        )
    )

    print_candidate_table(
        "TOP 5 HISTORICAL SHORTS",
        shorts,
    )

    print_candidate_table(
        "TOP 5 HISTORICAL LONGS",
        longs,
    )

    output = (
        OUTPUT_DIR
        / (
            f"historical_"
            f"{target_date}.csv"
        )
    )

    all_results.to_csv(
        output,
        index=False,
    )

    print(
        f"\nSaved: {output}"
    )


def run_single_stock(
    app: ResearchApplication,
) -> None:

    symbol = input(
        "\nEnter NSE stock symbol: "
    ).strip()

    value = input(
        "Enter analysis date "
        "(YYYY-MM-DD): "
    )

    target_date = parse_date(
        value
    )

    result = (
        app.analyze_stock(
            symbol,
            target_date,
        )
    )

    print_single_analysis(
        result
    )


def run_backtest(
    app: ResearchApplication,
) -> None:

    start_value = input(
        "\nBacktest start "
        "(YYYY-MM-DD): "
    )

    end_value = input(
        "Backtest end "
        "(YYYY-MM-DD): "
    )

    start_date = parse_date(
        start_value
    )

    end_date = parse_date(
        end_value
    )

    if start_date > end_date:

        raise ValueError(
            "Start date must be <= end date"
        )

    engine = (
        BacktestEngine(
            app
        )
    )

    result = engine.run(
        start_date,
        end_date,
    )

    if result.empty:

        print(
            "\nNo qualifying signals."
        )

        return

    print()
    print("=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    print(
        f"Trading dates      : "
        f"{result['backtest_date'].nunique()}"
    )

    print(
        f"Total signals      : "
        f"{len(result)}"
    )

    print(
        f"Short signals      : "
        f"{(result['signal'] == 'SHORT').sum()}"
    )

    print(
        f"Long signals       : "
        f"{(result['signal'] == 'LONG').sum()}"
    )

    print()

    print(
        result[
            [
                "backtest_date",
                "symbol",
                "signal",
                "structural_score",
                "conviction",
                "execution_price",
                "total_pct",
            ]
        ].to_string(
            index=False
        )
    )


# =============================================================================
# APPLICATION ENTRY
# =============================================================================

def main() -> int:

    configure_logging()

    try:

        app = (
            ResearchApplication()
        )

        while True:

            print_menu()

            choice = input(
                "\nSelect option: "
            ).strip()

            try:

                if choice == "1":

                    run_current_day(
                        app
                    )

                elif choice == "2":

                    run_historical(
                        app
                    )

                elif choice == "3":

                    run_single_stock(
                        app
                    )

                elif choice == "4":

                    run_backtest(
                        app
                    )

                elif choice == "5":

                    print(
                        "\nExiting."
                    )

                    return 0

                else:

                    print(
                        "\nInvalid option."
                    )

            except Exception as exc:

                LOGGER.exception(
                    "Operation failed"
                )

                print(
                    f"\nERROR: {exc}"
                )

    except KeyboardInterrupt:

        print(
            "\nInterrupted."
        )

        return 130

    except Exception as exc:

        LOGGER.exception(
            "Fatal application error"
        )

        print(
            f"\nFATAL ERROR: {exc}"
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )


