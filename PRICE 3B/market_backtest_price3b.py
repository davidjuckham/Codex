#!/usr/bin/env python3
"""CSV-driven market-wide PRICE 3B portfolio backtest.

Expected input: a directory with one or more CSV files.
Each CSV must include at least:
- date, open, high, low, close, volume
Optional:
- ticker (if missing, ticker is derived from filename stem)
- trading_value (if missing, computed as close * volume)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    commission_bps: float = 1.5
    slippage_bps: float = 2.0
    sell_tax_bps: float = 18.0


@dataclass
class StrategyConfig:
    mfi_period: int = 14
    mfi_min: float = 60.0
    ema_period: int = 200
    vwap_window: int = 21
    liquidity_window: int = 30
    min_trading_value: float = 3_000_000_000


@dataclass
class PortfolioConfig:
    initial_cash: float = 10_000_000
    max_positions: int = 10
    min_lot: int = 1
    costs: CostModel = field(default_factory=CostModel)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV-based KOSPI+KOSDAQ PRICE 3B portfolio backtest")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", type=str, help="Directory containing per-ticker CSV files")
    src.add_argument("--data-file", type=str, help="Single CSV file for whole market (must include ticker column)")
    p.add_argument("--file-glob", type=str, default="*.csv", help="CSV file pattern in data-dir")

    p.add_argument("--start", type=str, default="2026-01-01", help="Start date YYYY-MM-DD")
    p.add_argument("--end", type=str, default="2026-01-31", help="End date YYYY-MM-DD")

    p.add_argument("--initial-cash", type=float, default=10_000_000)
    p.add_argument("--max-positions", type=int, default=10)
    p.add_argument("--min-lot", type=int, default=1)

    p.add_argument("--mfi-period", type=int, default=14)
    p.add_argument("--mfi-min", type=float, default=60.0)
    p.add_argument("--ema-period", type=int, default=200)
    p.add_argument("--vwap-window", type=int, default=21)
    p.add_argument("--liquidity-window", type=int, default=30)
    p.add_argument("--min-trading-value", type=float, default=3_000_000_000)

    p.add_argument("--commission-bps", type=float, default=1.5)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--sell-tax-bps", type=float, default=18.0)

    p.add_argument("--output-equity", type=str, default="price_3b_market_equity.csv")
    p.add_argument("--output-trades", type=str, default="price_3b_market_trades.csv")
    p.add_argument("--output-realized", type=str, default="price_3b_market_realized.csv")
    return p.parse_args()


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def load_single_csv(path: Path, require_ticker: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    lower_map = {c.lower(): c for c in df.columns}
    required = ["date", "open", "high", "low", "close", "volume"]

    for col in required:
        if col not in lower_map:
            raise ValueError(f"{path.name}: missing required column '{col}'")

    rename_map = {lower_map[c]: c for c in required}
    has_ticker = "ticker" in lower_map
    if has_ticker:
        rename_map[lower_map["ticker"]] = "ticker"
    if "trading_value" in lower_map:
        rename_map[lower_map["trading_value"]] = "trading_value"

    df = df.rename(columns=rename_map)
    df = df[list(rename_map.values())]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "trading_value" in df.columns:
        df["trading_value"] = pd.to_numeric(df["trading_value"], errors="coerce")
    else:
        df["trading_value"] = df["close"] * df["volume"]

    if "ticker" not in df.columns:
        if require_ticker:
            raise ValueError(f"{path.name}: missing required column 'ticker'")
        # Example: 005930.csv -> ticker 005930
        df["ticker"] = path.stem
    else:
        df["ticker"] = df["ticker"].astype(str)

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).copy()
    df = df.sort_values("date").drop_duplicates(["date", "ticker"]).reset_index(drop=True)
    return df[["date", "ticker", "open", "high", "low", "close", "volume", "trading_value"]]


def load_market_panel(data_dir: Path, file_glob: str, lookback_start: date, end: date) -> pd.DataFrame:
    files = sorted(data_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No files matched '{file_glob}' in {data_dir}")

    parts: list[pd.DataFrame] = []
    for fp in files:
        if not fp.is_file():
            continue
        try:
            d = load_single_csv(fp)
        except Exception as e:
            print(f"[WARN] Skip {fp.name}: {e}")
            continue
        d = d[(d["date"].dt.date >= lookback_start) & (d["date"].dt.date <= end)]
        if not d.empty:
            parts.append(d)

    if not parts:
        raise RuntimeError("No usable rows loaded from CSV files")

    panel = pd.concat(parts, ignore_index=True)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def load_market_panel_from_file(data_file: Path, lookback_start: date, end: date) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"data-file not found: {data_file}")
    d = load_single_csv(data_file, require_ticker=True)
    d = d[(d["date"].dt.date >= lookback_start) & (d["date"].dt.date <= end)]
    if d.empty:
        raise RuntimeError("No usable rows loaded from data-file in the requested window")
    return d.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_indicators(panel: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = panel.copy()

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["money_flow"] = df["typical_price"] * df["volume"]

    grp = df.groupby("ticker", group_keys=False)
    tp_diff = grp["typical_price"].diff()

    df["pos_money_flow"] = np.where(tp_diff > 0, df["money_flow"], 0.0)
    df["neg_money_flow"] = np.where(tp_diff < 0, df["money_flow"], 0.0)

    pmf = grp["pos_money_flow"].rolling(cfg.mfi_period).sum().reset_index(level=0, drop=True)
    nmf = grp["neg_money_flow"].rolling(cfg.mfi_period).sum().reset_index(level=0, drop=True)

    mfr = pmf / nmf.replace(0, np.nan)
    df["mfi"] = 100 - (100 / (1 + mfr))
    df.loc[(nmf == 0) & (pmf > 0), "mfi"] = 100.0
    df.loc[(nmf == 0) & (pmf == 0), "mfi"] = np.nan

    df["ema_long"] = grp["close"].transform(lambda s: s.ewm(span=cfg.ema_period, adjust=False).mean())

    tpv = df["typical_price"] * df["volume"]
    vwap_num = tpv.groupby(df["ticker"]).rolling(cfg.vwap_window).sum().reset_index(level=0, drop=True)
    vwap_den = grp["volume"].rolling(cfg.vwap_window).sum().reset_index(level=0, drop=True)
    df["vwap_month"] = vwap_num / vwap_den.replace(0, np.nan)

    prev_close = grp["close"].shift(1)
    prev_vwap = grp["vwap_month"].shift(1)
    df["vwap_cross_up"] = (prev_close <= prev_vwap) & (df["close"] > df["vwap_month"])

    df["trading_value_ma30"] = grp["trading_value"].rolling(cfg.liquidity_window).mean().reset_index(level=0, drop=True)

    prev_mfi = grp["mfi"].shift(1)
    df["mfi_ok"] = (df["mfi"] > cfg.mfi_min) & (df["mfi"] > prev_mfi)
    df["ema_ok"] = df["close"] > df["ema_long"]
    df["liquidity_ok"] = df["trading_value_ma30"] >= cfg.min_trading_value

    df["entry_signal"] = df["vwap_cross_up"] & df["mfi_ok"] & df["ema_ok"] & df["liquidity_ok"]
    df["hold_regime"] = (df["close"] > df["vwap_month"]) & df["mfi_ok"] & df["ema_ok"] & df["liquidity_ok"]
    df["exit_signal"] = ~df["hold_regime"]
    return df


def compute_metrics(equity: pd.Series, daily_return: pd.Series, trading_days: int = 252) -> dict:
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


def format_pct(x: float) -> str:
    if np.isnan(x):
        return "nan"
    return f"{x * 100:.2f}%"


def run_portfolio(df: pd.DataFrame, start: date, end: date, pcfg: PortfolioConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = pcfg.costs
    period_df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()
    dates = sorted(period_df["date"].dropna().unique())
    fixed_budget_per_position = pcfg.initial_cash / max(pcfg.max_positions, 1)

    cash = pcfg.initial_cash
    positions: dict[str, int] = {}
    last_close: dict[str, float] = {}

    equity_rows: list[dict] = []
    trade_rows: list[dict] = []

    daily = {d: g.set_index("ticker") for d, g in period_df.groupby("date")}

    for d in dates:
        day = daily[d]

        for t in list(positions.keys()):
            if t not in day.index:
                continue
            row = day.loc[t]
            px = float(row["close"])
            last_close[t] = px
            exit_sig = bool(row["exit_signal"]) if pd.notna(row["exit_signal"]) else True
            if exit_sig:
                qty = positions[t]
                sell_price = px * (1 - c.slippage_bps / 10_000)
                gross = qty * sell_price
                fee = gross * (c.commission_bps / 10_000)
                tax = gross * (c.sell_tax_bps / 10_000)
                cash += gross - fee - tax
                trade_rows.append(
                    {
                        "date": d,
                        "ticker": t,
                        "side": "SELL",
                        "qty": qty,
                        "price": sell_price,
                        "gross": gross,
                        "fee": fee,
                        "tax": tax,
                        "cash_after": cash,
                    }
                )
                del positions[t]

        slots = max(pcfg.max_positions - len(positions), 0)
        if slots > 0:
            candidates = day[(day["entry_signal"] == True) & (day["close"] > 0)].copy()
            if not candidates.empty:
                candidates = candidates[~candidates.index.isin(positions.keys())]
                candidates = candidates.sort_values("trading_value", ascending=False)
                selected = candidates.head(slots)
                if not selected.empty:
                    for t, row in selected.iterrows():
                        px = float(row["close"])
                        buy_price = px * (1 + c.slippage_bps / 10_000)
                        unit_cost = buy_price * (1 + c.commission_bps / 10_000)
                        budget_each = min(fixed_budget_per_position, cash)
                        qty = int((budget_each // unit_cost) // pcfg.min_lot * pcfg.min_lot)
                        if qty <= 0:
                            continue
                        gross = qty * buy_price
                        fee = gross * (c.commission_bps / 10_000)
                        total = gross + fee
                        if total > cash:
                            continue
                        cash -= total
                        positions[t] = positions.get(t, 0) + qty
                        last_close[t] = px
                        trade_rows.append(
                            {
                                "date": d,
                                "ticker": t,
                                "side": "BUY",
                                "qty": qty,
                                "price": buy_price,
                                "gross": gross,
                                "fee": fee,
                                "tax": 0.0,
                                "cash_after": cash,
                            }
                        )

        mtm = 0.0
        for t, qty in positions.items():
            if t in day.index:
                px = float(day.loc[t, "close"])
                last_close[t] = px
            else:
                px = last_close.get(t, 0.0)
            mtm += qty * px

        equity_rows.append(
            {
                "date": d,
                "cash": cash,
                "positions": len(positions),
                "equity": cash + mtm,
            }
        )

    eq = pd.DataFrame(equity_rows)
    tr = pd.DataFrame(trade_rows)
    if not eq.empty:
        eq["daily_return"] = eq["equity"].pct_change().fillna(0.0)
    return eq, tr


def build_realized_trades(tr: pd.DataFrame) -> pd.DataFrame:
    """Pair BUY/SELL rows per ticker and compute realized PnL."""
    if tr.empty:
        return pd.DataFrame(columns=["ticker", "buy_price", "sell_price", "profit_krw", "return_pct"])

    tmp = tr.copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.sort_values(["ticker", "date"]).reset_index(drop=True)

    rows: list[dict] = []
    for ticker, g in tmp.groupby("ticker"):
        open_buy: dict | None = None
        for _, r in g.iterrows():
            side = str(r["side"]).upper()
            if side == "BUY":
                open_buy = {
                    "ticker": ticker,
                    "qty": float(r["qty"]),
                    "buy_price": float(r["price"]),
                    "buy_gross": float(r["gross"]),
                    "buy_fee": float(r["fee"]),
                }
            elif side == "SELL" and open_buy is not None:
                sell_gross = float(r["gross"])
                sell_fee = float(r["fee"])
                sell_tax = float(r["tax"])

                cost_total = open_buy["buy_gross"] + open_buy["buy_fee"]
                proceeds_total = sell_gross - sell_fee - sell_tax
                profit = proceeds_total - cost_total
                ret = profit / cost_total if cost_total > 0 else float("nan")

                rows.append(
                    {
                        "ticker": ticker,
                        "buy_price": open_buy["buy_price"],
                        "sell_price": float(r["price"]),
                        "profit_krw": profit,
                        "return_pct": ret * 100 if not np.isnan(ret) else float("nan"),
                    }
                )
                open_buy = None

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    if start >= end:
        raise ValueError("start must be earlier than end")

    scfg = StrategyConfig(
        mfi_period=args.mfi_period,
        mfi_min=args.mfi_min,
        ema_period=args.ema_period,
        vwap_window=args.vwap_window,
        liquidity_window=args.liquidity_window,
        min_trading_value=args.min_trading_value,
    )
    pcfg = PortfolioConfig(
        initial_cash=args.initial_cash,
        max_positions=args.max_positions,
        min_lot=args.min_lot,
        costs=CostModel(
            commission_bps=args.commission_bps,
            slippage_bps=args.slippage_bps,
            sell_tax_bps=args.sell_tax_bps,
        ),
    )

    if scfg.ema_period <= 1:
        raise ValueError("ema_period must be greater than 1")

    lookback_days = max(scfg.ema_period + 30, scfg.vwap_window + 30, scfg.liquidity_window + 30)
    lookback_start = start - timedelta(days=int(lookback_days * 1.8))

    print(f"Backtest period: {start} ~ {end}")
    print(f"Lookback start : {lookback_start}")
    if args.data_file:
        data_file = Path(args.data_file)
        print(f"Loading CSV    : {data_file}")
        panel = load_market_panel_from_file(data_file, lookback_start, end)
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"data-dir not found: {data_dir}")
        print(f"Loading CSVs   : {data_dir} ({args.file_glob})")
        panel = load_market_panel(data_dir, args.file_glob, lookback_start, end)
    print(f"Universe size  : {panel['ticker'].nunique()} tickers")
    print(f"Rows loaded    : {len(panel)}")

    print("Calculating PRICE 3B indicators/signals...")
    sig = add_indicators(panel, scfg)

    print("Running portfolio simulation...")
    eq, tr = run_portfolio(sig, start, end, pcfg)
    realized = build_realized_trades(tr)

    if eq.empty:
        raise RuntimeError("No equity rows generated. Check date range and data availability.")

    metrics = compute_metrics(eq["equity"], eq["daily_return"])

    Path(args.output_equity).parent.mkdir(parents=True, exist_ok=True)
    eq.to_csv(args.output_equity, index=False)
    tr.to_csv(args.output_trades, index=False)
    realized.to_csv(args.output_realized, index=False)

    print("=" * 64)
    print("PRICE 3B Market Backtest Summary (CSV Universe)")
    print("=" * 64)
    print(f"Period          : {start} ~ {end}")
    print(f"Initial Cash    : {pcfg.initial_cash:,.0f} KRW")
    print(f"Max Positions   : {pcfg.max_positions}")
    print(f"Trades          : {len(tr)}")
    print(f"Closed Trades   : {len(realized)}")
    print(f"End Value       : {metrics['end_value']:,.0f}")
    print(f"Total Return    : {format_pct(metrics['total_return'])}")
    print(f"CAGR            : {format_pct(metrics['cagr'])}")
    print(f"MDD             : {format_pct(metrics['max_drawdown'])}")
    print(f"Volatility      : {format_pct(metrics['annual_volatility'])}")
    print(f"Sharpe          : {metrics['sharpe']:.3f}" if not np.isnan(metrics["sharpe"]) else "Sharpe          : nan")
    print("=" * 64)
    print(f"Saved equity    : {args.output_equity}")
    print(f"Saved trades    : {args.output_trades}")
    print(f"Saved realized  : {args.output_realized}")


if __name__ == "__main__":
    main()
