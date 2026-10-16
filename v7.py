from __future__ import annotations

from datetime import datetime
from getpass import getpass
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import io
import json
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests


# ================================================================
# MORE12 — UPSTOX API VERSION
# ================================================================
# Original scanner logic preserved.
# Yahoo Finance/yfinance has been removed.
# Market data is now supplied by Upstox V3.
#
# Modes:
#   1 = Opening 15m candle (09:15-09:30) — current day
#   2 = Opening 15m candle (09:15-09:30) — historical date
#   3 = Latest completed 15m candle — current day
#
# IMPORTANT:
# Upstox returns current-day candles through the V3 intraday endpoint.
# Historical dates use the V3 historical-candle endpoint.
# ================================================================

MARKET_TZ = "Asia/Kolkata"
MARKET_ZONE = ZoneInfo(MARKET_TZ)

UPSTOX_INTERVAL_MINUTES = 15
UPSTOX_INSTRUMENT_URL = (
    "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
)
UPSTOX_INSTRUMENT_CACHE = "upstox_nse_instruments.json"

UPSTOX_MAX_WORKERS = 20
UPSTOX_REQUESTS_PER_SECOND = 45
UPSTOX_REQUESTS_PER_MINUTE = 480
UPSTOX_REQUEST_TIMEOUT = 20
UPSTOX_MAX_RETRIES = 3
UPSTOX_DEBUG_ERRORS = False
UPSTOX_MAX_ERROR_LINES = 20

UPSTOX_ACCESS_TOKEN_ENV = "UPSTOX_ACCESS_TOKEN"

_UPSTOX_TOKEN: Optional[str] = None
_UPSTOX_SESSION_LOCAL = threading.local()
_UPSTOX_LIMITER = None
_UPSTOX_INSTRUMENT_MAP: Optional[Dict[str, str]] = None


class UpstoxRateLimiter:
    """Thread-safe rate limiter used by the scanner's parallel requests."""

    def __init__(self, per_second: int, per_minute: int):
        self.per_second = per_second
        self.per_minute = per_minute
        self.lock = threading.Lock()
        self.starts = deque()

    def acquire(self) -> None:
        while True:
            sleep_for = 0.0

            with self.lock:
                now = time.monotonic()

                while self.starts and now - self.starts[0] >= 60.0:
                    self.starts.popleft()

                if len(self.starts) >= self.per_minute:
                    sleep_for = 60.0 - (now - self.starts[0]) + 0.001
                else:
                    recent_second = [
                        ts for ts in self.starts
                        if now - ts < 1.0
                    ]

                    if len(recent_second) >= self.per_second:
                        sleep_for = 1.0 - (now - recent_second[0]) + 0.001
                    else:
                        self.starts.append(now)
                        return

            time.sleep(max(sleep_for, 0.001))


def get_upstox_token() -> str:
    global _UPSTOX_TOKEN, _UPSTOX_LIMITER

    if _UPSTOX_TOKEN:
        return _UPSTOX_TOKEN

    token = os.getenv(UPSTOX_ACCESS_TOKEN_ENV, "").strip()

    if not token:
        print()
        print("Upstox access token is required.")
        print(
            "Set UPSTOX_ACCESS_TOKEN in your environment, "
            "or enter the token below."
        )
        token = getpass("UPSTOX ACCESS TOKEN: ").strip()

    if not token:
        raise RuntimeError(
            "No Upstox access token supplied."
        )

    _UPSTOX_TOKEN = token

    _UPSTOX_LIMITER = UpstoxRateLimiter(
        UPSTOX_REQUESTS_PER_SECOND,
        UPSTOX_REQUESTS_PER_MINUTE,
    )

    return _UPSTOX_TOKEN


def get_upstox_session() -> requests.Session:
    session = getattr(_UPSTOX_SESSION_LOCAL, "session", None)

    if session is None:
        session = requests.Session()
        session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        _UPSTOX_SESSION_LOCAL.session = session

    return session


def upstox_request(url: str) -> dict:
    """Authenticated GET with retry handling for transient API failures."""
    token = get_upstox_token()

    if _UPSTOX_LIMITER is None:
        raise RuntimeError("Upstox rate limiter is not initialized.")

    last_error = None

    for attempt in range(UPSTOX_MAX_RETRIES):
        _UPSTOX_LIMITER.acquire()

        try:
            response = get_upstox_session().get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                },
                timeout=UPSTOX_REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                payload = response.json()

                if payload.get("status") != "success":
                    raise RuntimeError(
                        f"Upstox returned non-success response: {payload}"
                    )

                return payload

            if response.status_code in {429, 500, 502, 503, 504}:
                last_error = RuntimeError(
                    f"Upstox HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )
                if attempt < UPSTOX_MAX_RETRIES - 1:
                    time.sleep(1.0 * (attempt + 1))
                    continue

            try:
                detail = response.json()
            except Exception:
                detail = response.text[:500]

            raise RuntimeError(
                f"Upstox HTTP {response.status_code}: {detail}"
            )

        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < UPSTOX_MAX_RETRIES - 1:
                time.sleep(1.0 * (attempt + 1))
                continue

    raise RuntimeError(f"Upstox request failed: {last_error}")


def download_upstox_instruments() -> Dict[str, str]:
    """Build trading-symbol -> NSE equity instrument-key mapping."""
    global _UPSTOX_INSTRUMENT_MAP

    if _UPSTOX_INSTRUMENT_MAP is not None:
        return _UPSTOX_INSTRUMENT_MAP

    cache_path = UPSTOX_INSTRUMENT_CACHE
    records = None

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
        except Exception:
            records = None

    if records is None:
        response = requests.get(
            UPSTOX_INSTRUMENT_URL,
            timeout=30,
        )
        response.raise_for_status()

        with gzip.GzipFile(
            fileobj=io.BytesIO(response.content)
        ) as gz:
            records = json.loads(
                gz.read().decode("utf-8")
            )

        try:
            with open(cache_path, "w", encoding="utf-8") as handle:
                json.dump(records, handle)
        except Exception:
            pass

    mapping: Dict[str, str] = {}

    for item in records:
        if not isinstance(item, dict):
            continue

        if item.get("segment") != "NSE_EQ":
            continue

        symbol = str(
            item.get("trading_symbol", "")
        ).strip().upper()

        instrument_key = str(
            item.get("instrument_key", "")
        ).strip()

        if not symbol or not instrument_key:
            continue

        instrument_type = str(
            item.get("instrument_type", "")
        ).strip().upper()

        if symbol not in mapping:
            mapping[symbol] = instrument_key
        elif instrument_type == "EQ":
            mapping[symbol] = instrument_key

    if not mapping:
        raise RuntimeError(
            "Upstox NSE instrument master returned no NSE_EQ instruments."
        )

    _UPSTOX_INSTRUMENT_MAP = mapping

    print(
        f"✅ Upstox instrument map loaded: "
        f"{len(mapping)} NSE equities"
    )

    return mapping


def ticker_to_instrument_key(ticker: str) -> Optional[str]:
    symbol = (
        ticker
        .replace(".NS", "")
        .strip()
        .upper()
    )
    return download_upstox_instruments().get(symbol)


def parse_upstox_candles(payload: dict) -> pd.DataFrame:
    candles = (
        payload
        .get("data", {})
        .get("candles", [])
    )

    if not candles:
        return pd.DataFrame()

    rows = []

    for candle in candles:
        if len(candle) < 6:
            continue

        rows.append({
            "Timestamp": candle[0],
            "Open": candle[1],
            "High": candle[2],
            "Low": candle[3],
            "Close": candle[4],
            "Volume": candle[5],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
    )

    df = df.dropna(
        subset=["Timestamp"]
    ).set_index("Timestamp")

    if getattr(df.index, "tz", None) is None:
        df.index = df.index.tz_localize(MARKET_TZ)
    else:
        df.index = df.index.tz_convert(MARKET_TZ)

    for column in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    df = df[
        (df["Open"] > 0)
        & (df["High"] > 0)
        & (df["Low"] > 0)
        & (df["Close"] > 0)
        & (df["High"] >= df["Low"])
    ]

    return df.sort_index()


def download_upstox_current_day(
    instrument_key: str,
) -> pd.DataFrame:
    """Get current-session 15-minute candles from Upstox V3."""
    url = (
        "https://api.upstox.com/v3/"
        "historical-candle/intraday/"
        f"{requests.utils.quote(instrument_key, safe='')}/"
        f"minutes/{UPSTOX_INTERVAL_MINUTES}"
    )

    payload = upstox_request(url)
    return parse_upstox_candles(payload)


def download_upstox_historical_day(
    instrument_key: str,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Get 15-minute candles for one historical trading date."""
    to_date = target_date.normalize()
    from_date = to_date - pd.Timedelta(days=1)

    url = (
        "https://api.upstox.com/v3/"
        "historical-candle/"
        f"{requests.utils.quote(instrument_key, safe='')}/"
        f"minutes/{UPSTOX_INTERVAL_MINUTES}/"
        f"{to_date:%Y-%m-%d}/{from_date:%Y-%m-%d}"
    )

    payload = upstox_request(url)
    return parse_upstox_candles(payload)


def is_current_market_date(target_date: pd.Timestamp) -> bool:
    now = datetime.now(MARKET_ZONE)
    return target_date.date() == now.date()


def download_one_ticker(
    ticker: str,
    mode: str,
    target_date: pd.Timestamp,
) -> Tuple[str, pd.DataFrame, Optional[str]]:
    """Download exactly the dataset required for one scanner symbol."""
    try:
        instrument_key = ticker_to_instrument_key(ticker)

        if not instrument_key:
            return ticker, pd.DataFrame(), "instrument_key not found"

        if mode in {"1", "3"}:
            df = download_upstox_current_day(
                instrument_key
            )
        else:
            df = download_upstox_historical_day(
                instrument_key,
                target_date,
            )

        if df.empty:
            return ticker, df, "Upstox returned no candles"

        return ticker, df, None

    except Exception as exc:
        return ticker, pd.DataFrame(), str(exc)


def download_scanner_data(
    tickers: List[str],
    mode: str,
    target_date: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """Parallel Upstox data download with shared rate limiting."""
    results: Dict[str, pd.DataFrame] = {}
    errors: List[Tuple[str, str]] = []

    print(
        f"⏳ Downloading 15m Upstox data for "
        f"{len(tickers)} equities..."
    )

    with ThreadPoolExecutor(
        max_workers=UPSTOX_MAX_WORKERS
    ) as executor:
        futures = [
            executor.submit(
                download_one_ticker,
                ticker,
                mode,
                target_date,
            )
            for ticker in tickers
        ]

        completed = 0

        for future in as_completed(futures):
            completed += 1

            try:
                ticker, df, error = future.result()
            except Exception as exc:
                ticker, df, error = "UNKNOWN", pd.DataFrame(), str(exc)

            if error:
                errors.append((ticker, error))

            if not df.empty:
                results[ticker] = df

            if completed % 25 == 0 or completed == len(futures):
                sys.stdout.write(
                    f"\r   Progress: "
                    f"{completed}/{len(futures)}"
                )
                sys.stdout.flush()

    if UPSTOX_DEBUG_ERRORS and errors:
        from collections import Counter
        summary = Counter(err for _, err in errors)
        print(f"⚠️ Symbols without usable data: {len(errors)}")
        print("   Common causes:")
        for reason, count in summary.most_common(5):
            print(f"   - {count}x {reason}")
        print("   Sample errors:")
        for ticker, error in errors[:UPSTOX_MAX_ERROR_LINES]:
            print(f"   - {ticker}: {error}")

    return results


def normalize_current_day_candles(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
) -> pd.DataFrame:
    """Keep only target-day candles and normalize timestamps."""
    if df.empty:
        return df

    out = df.copy()

    out = out[
        out.index.date == target_date.date()
    ].sort_index()

    return out


def get_latest_yahoo_compatible_candle(
    df: pd.DataFrame,
) -> Optional[pd.Series]:
    """
    Preserve the original more12.py Mode 3 behavior exactly.

    The Yahoo implementation selected the final row returned for today
    with ``day_candles.iloc[[-1]]``. It did NOT remove a currently forming
    15-minute candle. We intentionally preserve that behavior here so that
    changing the data vendor does not silently change the strategy logic.
    """
    if df.empty:
        return None

    return df.iloc[-1]


def get_opening_15m_candle(
    df: pd.DataFrame,
) -> Optional[pd.Series]:
    """Return the completed 09:15-09:30 candle only."""
    if df.empty:
        return None

    opening = df[
        (df.index.hour == 9)
        & (df.index.minute == 15)
    ]

    if opening.empty:
        return None

    return opening.iloc[0]




# Official NSE Indices constituent list. The downloaded CSV is checked to contain
# exactly 50 unique symbols before it is accepted as the scan universe.
NIFTY50_CONSTITUENTS_URL = (
    "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
)

# Fallback membership list based on the NIFTY 50 constituents published as of
# 2026-10-14. This fallback is used only if the official CSV cannot be loaded.
# Update it if NSE changes the index and the official endpoint is unavailable.
NIFTY50_FALLBACK_SYMBOLS = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BEL", "BHARTIARTL",
    "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HINDALCO",
    "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC",
    "JSWSTEEL", "JIOFIN", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC",
    "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SHRIRAMFIN",
    "SUNPHARMA", "TATACONSUM", "TMPV", "TATASTEEL", "TCS",
    "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO",
]


def get_nifty50_universe() -> List[str]:
    """Return only the 50 NIFTY 50 constituents (never NIFTY 500/all NSE)."""
    try:
        response = requests.get(
            NIFTY50_CONSTITUENTS_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
                ),
                "Accept": "text/csv,application/json,*/*",
            },
            timeout=10,
        )
        response.raise_for_status()

        constituents = pd.read_csv(io.BytesIO(response.content))
        symbol_column = next(
            (column for column in constituents.columns
             if str(column).strip().lower() == "symbol"),
            None,
        )
        if symbol_column is not None:
            symbols = [
                str(symbol).strip().upper().removesuffix(".NS")
                for symbol in constituents[symbol_column].dropna().tolist()
            ]
            # Require exactly 50 distinct, non-empty symbols before trusting it.
            symbols = list(dict.fromkeys(symbol for symbol in symbols if symbol))
            if len(symbols) == 50:
                print("✅ Loaded current 50-stock universe from NSE Indices.")
                return [f"{symbol}.NS" for symbol in symbols]

        print(
            "⚠️ NSE constituent CSV did not validate as 50 unique symbols; "
            "using the built-in NIFTY 50 fallback list."
        )
    except Exception as exc:
        print(
            "⚠️ Could not load the official NIFTY 50 list "
            f"({exc}); using the built-in NIFTY 50 fallback list."
        )

    if len(set(NIFTY50_FALLBACK_SYMBOLS)) != 50:
        raise RuntimeError("Built-in NIFTY 50 fallback must contain 50 unique symbols.")
    return [f"{symbol}.NS" for symbol in NIFTY50_FALLBACK_SYMBOLS]



def scan_momentum_stocks(
    mode: str = "1",
    target_date_str: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scans the configured NSE universe using Upstox 15-minute candles.

    Mode 1:
        Current-day opening 15m candle.
    Mode 2:
        Historical opening 15m candle for supplied YYYY-MM-DD.
    Mode 3:
        Current-day latest completed 15m candle.
    """
    if mode in {"1", "3"}:
        current_date = datetime.now(
            MARKET_ZONE
        ).strftime("%Y-%m-%d")

    else:
        if not target_date_str:
            raise ValueError(
                "Historical mode requires a target date."
            )

        current_date = target_date_str

        datetime.strptime(
            current_date,
            "%Y-%m-%d",
        )

    target_date = pd.Timestamp(
        current_date
    ).normalize()

    now = datetime.now(MARKET_ZONE)

    # A live Upstox intraday request represents the current trading session.
    # Before 09:15 there is no current-session candle for today.  For Mode 1
    # there is additionally no completed opening candle before 09:30.
    if mode in {"1", "3"} and target_date.date() == now.date():
        if now < now.replace(hour=9, minute=15, second=0, microsecond=0):
            print(
                "⚠️ NSE market has not opened yet (09:15 IST). "
                "Live modes 1 and 3 have no current-day candle to scan."
            )
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )

    print(
        f"📅 Scan Date: {current_date} | "
        f"Mode: "
        f"{'Latest Intraday Rolling' if mode == '3' else 'Opening Candle'}"
    )

    tickers = get_nifty50_universe()

    print(
        f"📊 Universe: {len(tickers)} NIFTY 50 constituents"
    )

    # Load instrument master once before starting worker threads.
    download_upstox_instruments()

    data_by_ticker = download_scanner_data(
        tickers,
        mode,
        target_date,
    )

    all_losers = []
    shortlisted_losers = []
    all_gainers = []
    shortlisted_gainers = []

    for ticker in tickers:
        try:
            ticker_df = data_by_ticker.get(
                ticker,
                pd.DataFrame(),
            )

            if ticker_df.empty:
                continue

            ticker_df = normalize_current_day_candles(
                ticker_df,
                target_date,
            )

            if ticker_df.empty:
                continue

            # --------------------------------------------------------
            # Select the same logical candle as the original scanner.
            # --------------------------------------------------------
            if mode == "3":
                latest_candle = (
                    get_latest_yahoo_compatible_candle(
                        ticker_df,
                    )
                )

                candle_label = "Latest_15m_Close"

            else:
                latest_candle = (
                    get_opening_15m_candle(
                        ticker_df,
                    )
                )

                candle_label = "15m_Close_Entry"

                # Never use a still-forming opening candle in live mode.
                if mode == "1":
                    opening_end = now.replace(
                        hour=9,
                        minute=30,
                        second=0,
                        microsecond=0,
                    )

                    if (
                        target_date.date() == now.date()
                        and now < opening_end
                    ):
                        latest_candle = None

            if latest_candle is None:
                continue

            open_p = float(
                latest_candle["Open"]
            )
            close_p = float(
                latest_candle["Close"]
            )
            high_p = float(
                latest_candle["High"]
            )
            low_p = float(
                latest_candle["Low"]
            )
            vol = int(
                latest_candle["Volume"]
            )

            if open_p == 0:
                continue

            pct_change = (
                (close_p - open_p)
                / open_p
            ) * 100

            turnover_cr = (
                vol * close_p
            ) / 10_000_000

            candle_range = (
                high_p - low_p
            )

            close_location = (
                (close_p - low_p)
                / candle_range
                if candle_range > 0
                else 1.0
            )

            # --------------------------------------------------------
            # SHORT (SELLING) LOGIC — PRESERVED
            # --------------------------------------------------------
            if pct_change < 0:
                row_data_short = {
                    "Ticker": ticker.replace(
                        ".NS",
                        "",
                    ),
                    "Date": current_date,
                    "15m_Open": round(
                        open_p,
                        2,
                    ),
                    candle_label: round(
                        close_p,
                        2,
                    ),
                    "Candle_Drop_%": round(
                        pct_change,
                        2,
                    ),
                    "Turnover_Cr": round(
                        turnover_cr,
                        2,
                    ),
                    "1%_Profit_Target": round(
                        close_p * 0.99,
                        2,
                    ),
                    "Hard_Stop_Loss": round(
                        high_p,
                        2,
                    ),
                }

                # Tier 1: General Losers
                if (
                    pct_change < -0.5
                    and turnover_cr >= 4.0
                    and close_location <= 0.25
                ):
                    all_losers.append(
                        row_data_short
                    )

                # Tier 2: Shortlisted Most Falling
                if (
                    pct_change < -1.5
                    and turnover_cr >= 15.0
                    and close_location <= 0.15
                ):
                    shortlisted_losers.append(
                        row_data_short
                    )

            # --------------------------------------------------------
            # LONG (BUYING) LOGIC — PRESERVED
            # --------------------------------------------------------
            elif pct_change > 0:
                row_data_long = {
                    "Ticker": ticker.replace(
                        ".NS",
                        "",
                    ),
                    "Date": current_date,
                    "15m_Open": round(
                        open_p,
                        2,
                    ),
                    candle_label: round(
                        close_p,
                        2,
                    ),
                    "Candle_Gain_%": round(
                        pct_change,
                        2,
                    ),
                    "Turnover_Cr": round(
                        turnover_cr,
                        2,
                    ),
                    "1%_Profit_Target": round(
                        close_p * 1.01,
                        2,
                    ),
                    "Hard_Stop_Loss": round(
                        low_p,
                        2,
                    ),
                }

                # Tier 1: General Gainers
                if (
                    pct_change > 0.5
                    and turnover_cr >= 4.0
                    and close_location >= 0.75
                ):
                    all_gainers.append(
                        row_data_long
                    )

                # Tier 2: Shortlisted Most Rising
                if (
                    pct_change > 1.5
                    and turnover_cr >= 15.0
                    and close_location >= 0.85
                ):
                    shortlisted_gainers.append(
                        row_data_long
                    )

        except Exception:
            continue

    df_all_losers = pd.DataFrame(
        all_losers
    )

    if not df_all_losers.empty:
        df_all_losers = (
            df_all_losers
            .sort_values(
                by="Candle_Drop_%",
                ascending=True,
            )
            .reset_index(drop=True)
        )

    df_short_losers = pd.DataFrame(
        shortlisted_losers
    )

    if not df_short_losers.empty:
        df_short_losers = (
            df_short_losers
            .sort_values(
                by="Candle_Drop_%",
                ascending=True,
            )
            .reset_index(drop=True)
        )

    df_all_gainers = pd.DataFrame(
        all_gainers
    )

    if not df_all_gainers.empty:
        df_all_gainers = (
            df_all_gainers
            .sort_values(
                by="Candle_Gain_%",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    df_short_gainers = pd.DataFrame(
        shortlisted_gainers
    )

    if not df_short_gainers.empty:
        df_short_gainers = (
            df_short_gainers
            .sort_values(
                by="Candle_Gain_%",
                ascending=False,
            )
            .reset_index(drop=True)
        )

    return (
        df_all_losers,
        df_short_losers,
        df_all_gainers,
        df_short_gainers,
    )


if __name__ == "__main__":
    print("Select Market Analysis Mode:")
    print("1. Today")
    print("2. Historical")
    print("3. Latest Completed 15m ")
    choice = input("Enter your choice (1, 2 or 3): ").strip()

    target_date = None
    match choice:
        case "1" | "3":
            pass
        case "2":
            target_date = input("Enter the target date in format YYYY-MM-DD (e.g., 2026-08-15): ").strip()
            try:
                datetime.strptime(target_date, "%Y-%m-%d")
            except ValueError:
                print("❌ Invalid date format provided. Please use YYYY-MM-DD.")
                sys.exit(1)
        case _:
            print("⚠️ Invalid selection. Defaulting to Mode 1 (Live Opening Data).")
            choice = "1"

    all_losers, shortlisted_losers, all_gainers, shortlisted_gainers = scan_momentum_stocks(choice, target_date)

    # Output Gainers (Long Positions)
    print("\n" + "=" * 80)
    print("🟢 REAL-TIME MOMENTUM BUYERS (GAINERS):")
    if not all_gainers.empty:
        print(all_gainers.to_string(index=False))
    else:
        print("No stocks matched the general breakout criteria.")

    print("\n" + "-" * 80)
    print("🎯 HIGH-CONVICTION SHORTLISTED (MOST RISING) BUYERS:")
    if not shortlisted_gainers.empty:
        print(shortlisted_gainers.to_string(index=False))
    else:
        print("No stocks matched the high-conviction breakout criteria.")

    # Output Losers (Short Positions)
    print("\n" + "=" * 80)
    print("🔴 REAL-TIME MOMENTUM SELLERS (LOSERS):")
    if not all_losers.empty:
        print(all_losers.to_string(index=False))
    else:
        print("No stocks matched the general breakdown criteria.")

    print("\n" + "-" * 80)
    print("🎯 HIGH-CONVICTION SHORTLISTED (MOST FALLING) SELLERS:")
    if not shortlisted_losers.empty:
        print(shortlisted_losers.to_string(index=False))
    else:
        print("No stocks matched the high-conviction breakdown criteria.")
