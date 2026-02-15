#!/usr/bin/env python3
"""
KR market oriented backtester (long-only SMA crossover).

Input format (CSV):
- Required columns: date, open, high, low, close, volume
- date format: YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    commission_bps: float = 1.5  # brokerage fee (example default)
    slippage_bps: float = 2.0    # execution slippage (example default)
    sell_tax_bps: float = 18.0   # transaction tax on sell side (example default)


@dataclass
class BacktestConfig:
    initial_cash: float = 10_000_000
    mfi_period: int = 14
    mfi_min: float = 60.0
    ema_period: int = 200
    vwap_window: int = 21
    liquidity_window: int = 30
    min_trading_value: float = 3_000_000_000
    min_lot: int = 1
    cost: CostModel = field(default_factory=CostModel)


class KRBacktester:
    def __init__(self, config: BacktestConfig):
        self.config = config

    @staticmethod
    def load_csv(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        req_cols = {"date", "open", "high", "low", "close", "volume"}
        missing = req_cols.difference(df.columns.str.lower())

        if missing:
            # case-insensitive fallback by lowering once
            lower_map = {c.lower(): c for c in df.columns}
            if not req_cols.issubset(lower_map.keys()):
                raise ValueError(f"CSV missing columns: {sorted(req_cols - set(lower_map.keys()))}")
            df = df.rename(columns={lower_map[c]: c for c in req_cols})

        # normalize known column names (case-insensitive)
        rename_map = {}
        for c in ["date", "open", "high", "low", "close", "volume"]:
            for orig in df.columns:
                if orig.lower() == c:
                    rename_map[orig] = c
                    break
        df = df.rename(columns=rename_map)

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").drop_duplicates("date")

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date", "close"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("No valid rows after cleaning input data")

        return df

    @staticmethod
    def load_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as e:
            raise RuntimeError("yfinance is not installed. Install it with: pip install yfinance") from e

        data = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if data.empty:
            raise ValueError(f"No data downloaded for ticker: {ticker}")

        # yfinance may return MultiIndex columns for some versions
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0].lower() for c in data.columns]
        else:
            data.columns = [str(c).lower() for c in data.columns]

        data = data.reset_index().rename(columns={"Date": "date", "date": "date"})
        data = data.rename(columns={"adj close": "adj_close"})

        # Ensure required set exists
        for required in ["open", "high", "low", "close", "volume"]:
            if required not in data.columns:
                raise ValueError(f"Downloaded data missing column: {required}")

        out = data[["date", "open", "high", "low", "close", "volume"]].copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values("date").reset_index(drop=True)
        return out

    def run_sma_crossover(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        c = cfg.cost

        out = df.copy()
        out["typical_price"] = (out["high"] + out["low"] + out["close"]) / 3.0
        out["money_flow"] = out["typical_price"] * out["volume"]

        # MFI(14): positive/negative money flow based on typical price change
        tp_diff = out["typical_price"].diff()
        out["pos_money_flow"] = np.where(tp_diff > 0, out["money_flow"], 0.0)
        out["neg_money_flow"] = np.where(tp_diff < 0, out["money_flow"], 0.0)
        pmf = out["pos_money_flow"].rolling(cfg.mfi_period).sum()
        nmf = out["neg_money_flow"].rolling(cfg.mfi_period).sum()
        mfr = pmf / nmf.replace(0, np.nan)
        out["mfi"] = 100 - (100 / (1 + mfr))
        out.loc[(nmf == 0) & (pmf > 0), "mfi"] = 100.0
        out.loc[(nmf == 0) & (pmf == 0), "mfi"] = np.nan

        out["ema_long"] = out["close"].ewm(span=cfg.ema_period, adjust=False).mean()

        # 1-month rolling VWAP (default 21 trading days)
        vwap_num = (out["typical_price"] * out["volume"]).rolling(cfg.vwap_window).sum()
        vwap_den = out["volume"].rolling(cfg.vwap_window).sum()
        out["vwap_month"] = vwap_num / vwap_den.replace(0, np.nan)
        out["vwap_cross_up"] = (
            (out["close"].shift(1) <= out["vwap_month"].shift(1))
            & (out["close"] > out["vwap_month"])
        )

        out["trading_value"] = out["close"] * out["volume"]
        out["trading_value_ma30"] = out["trading_value"].rolling(cfg.liquidity_window).mean()

        out["mfi_ok"] = (out["mfi"] > cfg.mfi_min) & (out["mfi"] > out["mfi"].shift(1))
        out["ema_ok"] = out["close"] > out["ema_long"]
        out["liquidity_ok"] = out["trading_value_ma30"] >= cfg.min_trading_value

        # Entry requires VWAP cross-up (B안) + other filters
        out["entry_signal"] = out["vwap_cross_up"] & out["mfi_ok"] & out["ema_ok"] & out["liquidity_ok"]
        # Hold regime after entry; cross-up is not required every day
        out["hold_regime"] = (out["close"] > out["vwap_month"]) & out["mfi_ok"] & out["ema_ok"] & out["liquidity_ok"]
        out["exit_signal"] = ~out["hold_regime"]

        cash = cfg.initial_cash
        shares = 0

        cash_list = []
        shares_list = []
        equity_list = []
        trades = []

        for i, row in out.iterrows():
            px = float(row["close"])
            enter = bool(row["entry_signal"]) if pd.notna(row["entry_signal"]) else False
            exit_ = bool(row["exit_signal"]) if pd.notna(row["exit_signal"]) else False

            if math.isnan(px) or px <= 0:
                cash_list.append(cash)
                shares_list.append(shares)
                equity_list.append(cash + shares * 0.0)
                trades.append(0)
                continue

            traded = 0

            # Buy on entry signal while flat
            if enter and shares == 0:
                buy_price = px * (1 + c.slippage_bps / 10_000)
                max_affordable = int(cash // (buy_price * (1 + c.commission_bps / 10_000)))
                qty = (max_affordable // cfg.min_lot) * cfg.min_lot

                if qty > 0:
                    gross = qty * buy_price
                    fee = gross * (c.commission_bps / 10_000)
                    total = gross + fee
                    cash -= total
                    shares += qty
                    traded = qty

            # Sell when regime breaks
            elif exit_ and shares > 0:
                sell_price = px * (1 - c.slippage_bps / 10_000)
                qty = shares
                gross = qty * sell_price
                fee = gross * (c.commission_bps / 10_000)
                tax = gross * (c.sell_tax_bps / 10_000)
                net = gross - fee - tax
                cash += net
                shares = 0
                traded = -qty

            equity = cash + shares * px

            cash_list.append(cash)
            shares_list.append(shares)
            equity_list.append(equity)
            trades.append(traded)

        out["cash"] = cash_list
        out["shares"] = shares_list
        out["trade_qty"] = trades
        out["signal"] = (out["shares"] > 0).astype(int)
        out["equity"] = equity_list
        out["daily_return"] = out["equity"].pct_change().fillna(0.0)

        # Buy & hold benchmark (no trading costs to simplify comparison)
        bh_shares = int(cfg.initial_cash // out["close"].iloc[0])
        bh_cash = cfg.initial_cash - bh_shares * out["close"].iloc[0]
        out["bh_equity"] = bh_cash + bh_shares * out["close"]
        out["bh_daily_return"] = out["bh_equity"].pct_change().fillna(0.0)

        return out

    @staticmethod
    def compute_metrics(equity: pd.Series, daily_return: pd.Series, trading_days: int = 252) -> dict:
        if equity.empty:
            return {}

        start_val = float(equity.iloc[0])
        end_val = float(equity.iloc[-1])

        n = len(equity)
        years = max(n / trading_days, 1e-9)

        total_return = (end_val / start_val) - 1 if start_val > 0 else float("nan")
        cagr = (end_val / start_val) ** (1 / years) - 1 if start_val > 0 and end_val > 0 else float("nan")

        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        mdd = float(drawdown.min()) if not drawdown.empty else float("nan")

        vol = float(daily_return.std(ddof=1) * math.sqrt(trading_days))
        mean = float(daily_return.mean() * trading_days)
        sharpe = mean / vol if vol > 0 else float("nan")

        return {
            "start_value": start_val,
            "end_value": end_val,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": mdd,
            "annual_volatility": vol,
            "sharpe": sharpe,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Korean market oriented backtester (MFI/VWAP/EMA/liquidity)")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", type=str, help="Path to OHLCV CSV")
    src.add_argument("--ticker", type=str, help="Ticker for yfinance, e.g., 005930.KS")

    p.add_argument("--start", type=str, default="2018-01-01", help="Start date (for --ticker)")
    p.add_argument("--end", type=str, default="2026-01-01", help="End date (for --ticker)")

    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--mfi-period", type=int, default=14)
    p.add_argument("--mfi-min", type=float, default=60.0)
    p.add_argument("--ema-period", type=int, default=200)
    p.add_argument("--vwap-window", type=int, default=21, help="Trading days for monthly VWAP")
    p.add_argument("--liquidity-window", type=int, default=30)
    p.add_argument("--min-trading-value", type=float, default=3_000_000_000, help="KRW")
    p.add_argument("--min-lot", type=int, default=1)

    p.add_argument("--commission-bps", type=float, default=1.5)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--sell-tax-bps", type=float, default=18.0)

    p.add_argument("--output", type=str, default="price_3b_result.csv", help="Output result CSV")
    return p.parse_args()


def format_pct(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x * 100:.2f}%"


def main() -> None:
    args = parse_args()

    cfg = BacktestConfig(
        initial_cash=args.initial_cash,
        mfi_period=args.mfi_period,
        mfi_min=args.mfi_min,
        ema_period=args.ema_period,
        vwap_window=args.vwap_window,
        liquidity_window=args.liquidity_window,
        min_trading_value=args.min_trading_value,
        min_lot=args.min_lot,
        cost=CostModel(
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            sell_tax_bps=args.sell_tax_bps,
        ),
    )

    bt = KRBacktester(cfg)

    if args.csv:
        df = bt.load_csv(Path(args.csv))
        source_desc = f"CSV: {args.csv}"
    else:
        df = bt.load_yfinance(args.ticker, args.start, args.end)
        source_desc = f"yfinance: {args.ticker} ({args.start} to {args.end})"

    if cfg.mfi_period <= 1:
        raise ValueError("mfi_period must be greater than 1")
    if cfg.vwap_window <= 1:
        raise ValueError("vwap_window must be greater than 1")
    if cfg.liquidity_window <= 1:
        raise ValueError("liquidity_window must be greater than 1")

    result = bt.run_sma_crossover(df)

    strat = bt.compute_metrics(result["equity"], result["daily_return"])
    bench = bt.compute_metrics(result["bh_equity"], result["bh_daily_return"])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)

    print("=" * 64)
    print("KR Backtest Summary")
    print("=" * 64)
    print(f"Source        : {source_desc}")
    print(f"Rows          : {len(result)}")
    print(f"Date range    : {result['date'].iloc[0].date()} ~ {result['date'].iloc[-1].date()}")
    print(f"Initial cash  : {cfg.initial_cash:,.0f} KRW")
    print(
        f"Rule Params   : MFI({cfg.mfi_period})>{cfg.mfi_min}, "
        f"VWAP({cfg.vwap_window}) cross-up entry, EMA({cfg.ema_period}) filter"
    )
    print(
        f"Liquidity     : {cfg.liquidity_window}D avg trading value >= {cfg.min_trading_value:,.0f} KRW"
    )
    print(
        f"Costs (bps)   : commission={cfg.cost.commission_bps}, "
        f"slippage={cfg.cost.slippage_bps}, sell_tax={cfg.cost.sell_tax_bps}"
    )
    print("-" * 64)
    print("[Strategy]")
    print(f"End Value     : {strat['end_value']:,.0f}")
    print(f"Total Return  : {format_pct(strat['total_return'])}")
    print(f"CAGR          : {format_pct(strat['cagr'])}")
    print(f"MDD           : {format_pct(strat['max_drawdown'])}")
    print(f"Volatility    : {format_pct(strat['annual_volatility'])}")
    print(f"Sharpe        : {strat['sharpe']:.3f}" if not np.isnan(strat["sharpe"]) else "Sharpe        : nan")
    print("-" * 64)
    print("[Buy & Hold]")
    print(f"End Value     : {bench['end_value']:,.0f}")
    print(f"Total Return  : {format_pct(bench['total_return'])}")
    print(f"CAGR          : {format_pct(bench['cagr'])}")
    print(f"MDD           : {format_pct(bench['max_drawdown'])}")
    print(f"Volatility    : {format_pct(bench['annual_volatility'])}")
    print(f"Sharpe        : {bench['sharpe']:.3f}" if not np.isnan(bench["sharpe"]) else "Sharpe        : nan")
    print("=" * 64)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
