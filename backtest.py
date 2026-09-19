import yfinance as yf
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CAPITAL = 2000
MAX_TRADES = 2
LEVERAGE = 5.0

RVOL_LIMIT = 2.0

TARGET_PCT = 0.0070       # +0.70%
STOP_PCT = 0.0035         # -0.35%

SIGNAL_TIME = "09:40"

# Yahoo NSE symbols
SYMBOLS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "LT.NS",
    "ITC.NS",
    "BHARTIARTL.NS",
]


# ============================================================
# SWITCH CASE
# ============================================================

MODE = "SINGLE_DAY"

"""
Available modes:

"SINGLE_DAY"
    Test one historical date.

"MULTI_DAY"
    Test every available trading day in the selected range.

"SINGLE_STOCK"
    Test one stock across the selected range.

"UNIVERSE"
    Test the entire SYMBOLS list.
"""


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data(symbol):

    print(f"Downloading {symbol}...")

    df = yf.download(
        symbol,
        period="60d",
        interval="5m",
        auto_adjust=False,
        prepost=False,
        progress=False,
    )

    if df.empty:
        return pd.DataFrame()

    # yfinance can return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    if "Datetime" in df.columns:
        df.rename(
            columns={"Datetime": "datetime"},
            inplace=True,
        )

    # Convert timezone to India
    if df["datetime"].dt.tz is None:
        df["datetime"] = (
            df["datetime"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("Asia/Kolkata")
        )
    else:
        df["datetime"] = (
            df["datetime"]
            .dt.tz_convert("Asia/Kolkata")
        )

    df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        },
        inplace=True,
    )

    df = df[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    return df.dropna().reset_index(drop=True)


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA 9
    df["EMA9"] = (
        df["close"]
        .ewm(
            span=9,
            adjust=False,
        )
        .mean()
    )

    # EMA 21
    df["EMA21"] = (
        df["close"]
        .ewm(
            span=21,
            adjust=False,
        )
        .mean()
    )

    # Typical Price
    df["TypicalPrice"] = (
        df["high"]
        + df["low"]
        + df["close"]
    ) / 3

    # --------------------------------------------------------
    # SESSION VWAP
    #
    # IMPORTANT:
    # VWAP resets at the beginning of every trading day.
    # --------------------------------------------------------

    df["date"] = df["datetime"].dt.date

    df["TPVolume"] = (
        df["TypicalPrice"]
        * df["volume"]
    )

    df["CumTPVolume"] = (
        df.groupby("date")["TPVolume"]
        .cumsum()
    )

    df["CumVolume"] = (
        df.groupby("date")["volume"]
        .cumsum()
    )

    df["VWAP"] = (
        df["CumTPVolume"]
        / df["CumVolume"]
    )

    # --------------------------------------------------------
    # EMA CROSS
    # --------------------------------------------------------

    previous_ema9 = df["EMA9"].shift(1)
    previous_ema21 = df["EMA21"].shift(1)

    df["Cross_Up"] = (
        (previous_ema9 <= previous_ema21)
        &
        (df["EMA9"] > df["EMA21"])
    )

    df["Cross_Down"] = (
        (previous_ema9 >= previous_ema21)
        &
        (df["EMA9"] < df["EMA21"])
    )

    # --------------------------------------------------------
    # Intraday velocity
    # --------------------------------------------------------

    df["Velocity"] = (
        df["close"].pct_change()
        * 100
    )

    return df


# ============================================================
# RVOL
# ============================================================

def calculate_rvol(df, target_date):

    """
    Calculate RVOL for the 09:40 candle.

    Current volume
    -----------------------
    Mean volume of the 09:40 candle
    across previous 10 sessions
    """

    data = df.copy()

    data["date"] = data["datetime"].dt.date
    data["time"] = data["datetime"].dt.strftime("%H:%M")

    # 09:40 candle only
    same_time = data[
        data["time"] == SIGNAL_TIME
    ]

    # Current day
    current = same_time[
        same_time["date"] == target_date
    ]

    if current.empty:
        return np.nan

    current_volume = float(
        current.iloc[-1]["volume"]
    )

    # Previous sessions
    previous = same_time[
        same_time["date"] < target_date
    ]

    previous_days = sorted(
        previous["date"].unique(),
        reverse=True,
    )[:10]

    if len(previous_days) < 10:
        return np.nan

    previous_volume = previous[
        previous["date"].isin(previous_days)
    ]["volume"]

    average_volume = (
        previous_volume.mean()
    )

    if average_volume <= 0:
        return np.nan

    return (
        current_volume
        / average_volume
    )


# ============================================================
# PREVIOUS DAY CLOSE
# ============================================================

def previous_day_close(
    df,
    target_date,
):

    data = df.copy()

    data["date"] = (
        data["datetime"].dt.date
    )

    previous = data[
        data["date"] < target_date
    ]

    if previous.empty:
        return np.nan

    previous_dates = sorted(
        previous["date"].unique(),
        reverse=True,
    )

    previous_date = previous_dates[0]

    previous_day = previous[
        previous["date"]
        == previous_date
    ]

    if previous_day.empty:
        return np.nan

    return float(
        previous_day.iloc[-1]["close"]
    )


# ============================================================
# POSITION SIZE
# ============================================================

def calculate_quantity(
    entry_price,
):

    allocated_capital = (
        CAPITAL
        / MAX_TRADES
    )

    quantity = int(
        (
            allocated_capital
            * LEVERAGE
        )
        // entry_price
    )

    return quantity


# ============================================================
# SIGNAL
# ============================================================

def generate_signal(
    symbol,
    df,
    target_date,
):

    target_timestamp = pd.Timestamp(
        f"{target_date} {SIGNAL_TIME}",
        tz="Asia/Kolkata",
    )

    # Only information available at 09:40
    history = df[
        df["datetime"]
        <= target_timestamp
    ].copy()

    if history.empty:
        return None

    history = add_indicators(
        history
    )

    candle = history[
        history["datetime"]
        == target_timestamp
    ]

    if candle.empty:
        return None

    candle = candle.iloc[-1]

    # --------------------------------------------------------
    # Previous Day Close
    # --------------------------------------------------------

    prev_close = previous_day_close(
        df,
        target_date,
    )

    if np.isnan(prev_close):
        return None

    # --------------------------------------------------------
    # RVOL
    # --------------------------------------------------------

    rvol = calculate_rvol(
        df,
        target_date,
    )

    if np.isnan(rvol):
        return None

    close = float(
        candle["close"]
    )

    vwap = float(
        candle["VWAP"]
    )

    velocity = float(
        candle["Velocity"]
    )

    quantity = calculate_quantity(
        close
    )

    if quantity <= 0:
        return None

    # ========================================================
    # LONG
    # ========================================================

    if (
        close > vwap
        and close > prev_close
        and rvol > RVOL_LIMIT
        and bool(candle["Cross_Up"])
    ):

        return {
            "symbol": symbol,
            "side": "LONG",
            "entry": close,
            "quantity": quantity,
            "velocity": velocity,
            "rvol": rvol,
            "vwap": vwap,
            "previous_close": prev_close,
        }

    # ========================================================
    # SHORT
    # ========================================================

    if (
        close < vwap
        and close < prev_close
        and rvol > RVOL_LIMIT
        and bool(candle["Cross_Down"])
    ):

        return {
            "symbol": symbol,
            "side": "SHORT",
            "entry": close,
            "quantity": quantity,
            "velocity": velocity,
            "rvol": rvol,
            "vwap": vwap,
            "previous_close": prev_close,
        }

    return None


# ============================================================
# TRADE SIMULATOR
# ============================================================

def simulate_trade(
    df,
    signal,
):

    entry = signal["entry"]
    side = signal["side"]
    quantity = signal["quantity"]

    signal_time = pd.Timestamp(
        f"{TARGET_DATE} {SIGNAL_TIME}",
        tz="Asia/Kolkata",
    )

    future = df[
        df["datetime"]
        > signal_time
    ].copy()

    # --------------------------------------------------------
    # LONG
    # --------------------------------------------------------

    if side == "LONG":

        target = (
            entry
            * (1 + TARGET_PCT)
        )

        stop = (
            entry
            * (1 - STOP_PCT)
        )

    # --------------------------------------------------------
    # SHORT
    # --------------------------------------------------------

    else:

        target = (
            entry
            * (1 - TARGET_PCT)
        )

        stop = (
            entry
            * (1 + STOP_PCT)
        )

    # --------------------------------------------------------
    # Candle-by-candle simulation
    # --------------------------------------------------------

    for _, candle in future.iterrows():

        high = float(
            candle["high"]
        )

        low = float(
            candle["low"]
        )

        timestamp = candle[
            "datetime"
        ]

        # ====================================================
        # LONG
        # ====================================================

        if side == "LONG":

            # Conservative assumption:
            # if target and stop occur in same candle,
            # stop is considered first.

            if low <= stop:

                exit_price = stop
                reason = "STOP"

                pnl = (
                    exit_price
                    - entry
                ) * quantity

                return {
                    **signal,
                    "exit": exit_price,
                    "pnl": pnl,
                    "reason": reason,
                    "exit_time": timestamp,
                }

            if high >= target:

                exit_price = target
                reason = "TARGET"

                pnl = (
                    exit_price
                    - entry
                ) * quantity

                return {
                    **signal,
                    "exit": exit_price,
                    "pnl": pnl,
                    "reason": reason,
                    "exit_time": timestamp,
                }

        # ====================================================
        # SHORT
        # ====================================================

        else:

            if high >= stop:

                exit_price = stop
                reason = "STOP"

                pnl = (
                    entry
                    - exit_price
                ) * quantity

                return {
                    **signal,
                    "exit": exit_price,
                    "pnl": pnl,
                    "reason": reason,
                    "exit_time": timestamp,
                }

            if low <= target:

                exit_price = target
                reason = "TARGET"

                pnl = (
                    entry
                    - exit_price
                ) * quantity

                return {
                    **signal,
                    "exit": exit_price,
                    "pnl": pnl,
                    "reason": reason,
                    "exit_time": timestamp,
                }

    # --------------------------------------------------------
    # Neither target nor stop hit
    # Exit at last candle
    # --------------------------------------------------------

    if not future.empty:

        last = future.iloc[-1]

        exit_price = float(
            last["close"]
        )

        if side == "LONG":

            pnl = (
                exit_price
                - entry
            ) * quantity

        else:

            pnl = (
                entry
                - exit_price
            ) * quantity

        return {
            **signal,
            "exit": exit_price,
            "pnl": pnl,
            "reason": "EOD",
            "exit_time": last[
                "datetime"
            ],
        }

    return None


# ============================================================
# SINGLE DAY TEST
# ============================================================

def run_single_day():

    print()
    print("=" * 70)
    print(
        f"BACKTEST DATE: {TARGET_DATE}"
    )
    print("=" * 70)

    candidates = []

    for symbol in SYMBOLS:

        signal = generate_signal(
            symbol,
            DATA[symbol],
            TARGET_DATE,
        )

        if signal:

            candidates.append(
                signal
            )

    # ========================================================
    # LONG RANKING
    # Highest velocity first
    # ========================================================

    longs = [
        x for x in candidates
        if x["side"] == "LONG"
    ]

    longs.sort(
        key=lambda x: x["velocity"],
        reverse=True,
    )

    # ========================================================
    # SHORT RANKING
    # Lowest velocity first
    # ========================================================

    shorts = [
        x for x in candidates
        if x["side"] == "SHORT"
    ]

    shorts.sort(
        key=lambda x: x["velocity"]
    )

    # ========================================================
    # SELECT MAX 2
    # ========================================================

    selected = []

    if longs:

        selected.append(
            longs[0]
        )

    if (
        shorts
        and len(selected)
        < MAX_TRADES
    ):

        selected.append(
            shorts[0]
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print()
    print(
        f"Total candidates: "
        f"{len(candidates)}"
    )

    print(
        f"Long candidates: "
        f"{len(longs)}"
    )

    print(
        f"Short candidates: "
        f"{len(shorts)}"
    )

    print()

    if not selected:

        print(
            "NO TRADE"
        )

        return []

    results = []

    for signal in selected:

        print(
            f"{signal['side']:5} "
            f"{signal['symbol']:15} "
            f"Entry={signal['entry']:.2f} "
            f"RVOL={signal['rvol']:.2f} "
            f"Velocity={signal['velocity']:.2f}% "
            f"Qty={signal['quantity']}"
        )

        result = simulate_trade(
            DATA[
                signal["symbol"]
            ],
            signal,
        )

        if result:

            results.append(
                result
            )

            print(
                f"  EXIT={result['exit']:.2f} "
                f"REASON={result['reason']} "
                f"PnL=₹{result['pnl']:.2f}"
            )

    return results


# ============================================================
# LOAD DATA
# ============================================================

def load_all_data():

    data = {}

    for symbol in SYMBOLS:

        df = download_data(
            symbol
        )

        if not df.empty:

            data[symbol] = df

    return data


# ============================================================
# MULTI-DAY MODE
# ============================================================

def run_multi_day():

    all_results = []

    # Find available dates
    dates = set()

    for df in DATA.values():

        dates.update(
            df["datetime"]
            .dt.date
            .unique()
        )

    dates = sorted(dates)

    for day in dates:

        # Need enough history before
        # running the 10-session RVOL.
        if str(day) < START_DATE:
            continue

        global TARGET_DATE

        TARGET_DATE = day

        results = run_single_day()

        all_results.extend(
            results
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL BACKTEST")
    print("=" * 70)

    if not all_results:

        print(
            "No trades generated."
        )

        return

    result_df = pd.DataFrame(
        all_results
    )

    total_pnl = (
        result_df["pnl"]
        .sum()
    )

    wins = result_df[
        result_df["pnl"] > 0
    ]

    losses = result_df[
        result_df["pnl"] <= 0
    ]

    print(
        f"Trades       : "
        f"{len(result_df)}"
    )

    print(
        f"Winners      : "
        f"{len(wins)}"
    )

    print(
        f"Losers       : "
        f"{len(losses)}"
    )

    print(
        f"Win Rate     : "
        f"{len(wins) / len(result_df) * 100:.2f}%"
    )

    print(
        f"Total PnL    : "
        f"₹{total_pnl:.2f}"
    )

    print(
        f"Final Capital: "
        f"₹{CAPITAL + total_pnl:.2f}"
    )

    print()

    print(
        result_df[
            [
                "symbol",
                "side",
                "entry",
                "exit",
                "quantity",
                "pnl",
                "reason",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# MAIN SWITCH
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHANGE THIS DATE
    # --------------------------------------------------------

    TARGET_DATE = "2026-09-15"

    # --------------------------------------------------------
    # Change this if using MULTI_DAY
    # --------------------------------------------------------

    START_DATE = "2026-08-01"

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    DATA = load_all_data()

    # --------------------------------------------------------
    # SWITCH CASE
    # --------------------------------------------------------

    match MODE:

        case "SINGLE_DAY":

            run_single_day()

        case "MULTI_DAY":

            run_multi_day()

        case "SINGLE_STOCK":

            # Automatically reduce universe to one stock.
            # Change SYMBOLS above to whichever stock you want.
            run_multi_day()

        case "UNIVERSE":

            run_multi_day()

        case _:

            print(
                "Unknown MODE:",
                MODE,
            )

