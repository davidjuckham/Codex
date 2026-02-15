# 한국 시장 중심 백테스트 프로그램

KRX 종목(예: `005930.KS`)에 맞춰 바로 사용할 수 있는 Python 기반 백테스트 CLI입니다.

기본 전략은 아래 4개 조건 기반의 롱온리 전략입니다.

- `MFI(14) > 60` 이고 전일 대비 상승
- `VWAP(1달)` 위로 가격이 돌파(cross-up, B안)
- `가격 > EMA(200)`
- `30일 평균 거래대금 >= 3B KRW`

한국 시장에서 자주 고려하는 항목(매도세, 수수료, 슬리피지)을 bps 단위로 반영할 수 있습니다.

## 1) 설치

```bash
cd /Users/jusungyun/Documents/Codex/kr_backtester
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) 입력 데이터 포맷 (CSV)

CSV는 아래 컬럼이 필요합니다.

- `date` (YYYY-MM-DD)
- `open`
- `high`
- `low`
- `close`
- `volume`

예시:

```csv
date,open,high,low,close,volume
2024-01-02,78500,79800,78200,79600,12345678
2024-01-03,79800,80100,79000,79200,11000000
```

## 3) 실행 예시

### A. CSV로 실행

```bash
python backtest_kr.py \
  --csv ./sample.csv \
  --initial-cash 10000000 \
  --mfi-period 14 \
  --mfi-min 60 \
  --ema-period 200 \
  --vwap-window 21 \
  --liquidity-window 30 \
  --min-trading-value 3000000000 \
  --commission-bps 1.5 \
  --slippage-bps 2.0 \
  --sell-tax-bps 18.0 \
  --output ./result_samsung.csv
```

### B. yfinance로 KRX 티커 실행

```bash
python backtest_kr.py \
  --ticker 005930.KS \
  --start 2018-01-01 \
  --end 2026-01-01 \
  --initial-cash 10000000 \
  --output ./result_005930.csv
```

## 4) 출력 결과

- 콘솔 요약: 최종자산, 총수익률, CAGR, MDD, 변동성, Sharpe
- 파일 저장: `--output` 경로에 일별 백테스트 결과 CSV

주요 컬럼:

- `mfi`, `ema_long`, `vwap_month`, `vwap_cross_up`
- `mfi_ok`, `ema_ok`, `liquidity_ok`, `entry_signal`, `hold_regime`, `exit_signal`, `signal`
- `trade_qty`, `shares`, `cash`, `equity`
- `daily_return`
- `bh_equity`, `bh_daily_return` (Buy & Hold 비교)

## 5) 주의

- 기본 수수료/세율 값은 예시값입니다. 실제 적용 전 증권사/시장 규정에 맞게 조정하세요.
- 전략 로직은 조건 기반 예시입니다. 필요하면 손절/익절, 리밸런싱, 포지션 사이징으로 확장 가능합니다.
