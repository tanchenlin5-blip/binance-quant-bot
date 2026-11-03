import os, time, json, csv, logging
from datetime import datetime, timezone, date
from pathlib import Path

import requests
import pandas as pd

BASE_URL = "https://fapi.binance.com"
CONFIG_PATH = Path("config.json")
STATE_PATH = Path("paper_state.json")
TRADES_PATH = Path("trades.csv")
SIGNALS_PATH = Path("signals.csv")

DEFAULT_CONFIG = {
    "starting_balance": 1000.0,
    "timeframe": "15m",
    "scan_interval_seconds": 90,
    "min_24h_quote_volume": 10000000,
    "max_symbols_to_analyze": 40,
    "max_open_positions": 3,
    "risk_per_trade_pct": 0.005,
    "max_daily_loss_pct": 0.02,
    "simulated_leverage": 1.0,
    "fee_rate": 0.0005,
    "slippage_rate": 0.0002,
    "atr_stop_multiplier": 1.8,
    "take_profit_r_multiple": 2.0,
    "min_volume_ratio": 1.2,
    "rsi_long_min": 50,
    "rsi_long_max": 68,
    "rsi_short_min": 32,
    "rsi_short_max": 50,
    "dry_run_only": True
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log", encoding="utf-8")]
)
log = logging.getLogger("quant-bot")
session = requests.Session()
session.headers.update({"User-Agent": "BinancePaperQuantBot/1.0"})


def load_config():
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        log.info("已生成 config.json，请按需调整参数。")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if cfg.get("dry_run_only") is not True:
        raise RuntimeError("安全锁：本版本只允许模拟交易，dry_run_only 必须为 true。")
    return cfg


def api_get(path, params=None):
    r = session.get(BASE_URL + path, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_universe(cfg):
    info = api_get("/fapi/v1/exchangeInfo")
    allowed = {
        s["symbol"] for s in info["symbols"]
        if s.get("status") == "TRADING"
        and s.get("contractType") == "PERPETUAL"
        and s.get("quoteAsset") == "USDT"
        and s.get("marginAsset") == "USDT"
    }
    tickers = api_get("/fapi/v1/ticker/24hr")
    ranked = []
    for t in tickers:
        sym = t.get("symbol")
        if sym in allowed:
            qv = float(t.get("quoteVolume", 0))
            if qv >= cfg["min_24h_quote_volume"]:
                ranked.append((sym, qv))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:int(cfg["max_symbols_to_analyze"])]


def get_candles(symbol, interval, limit=100):
    rows = api_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open","high","low","close","volume","quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Exclude the still-forming candle to avoid look-ahead / intrabar signal flicker.
    now_ms = int(time.time() * 1000)
    df = df[df["close_time"].astype("int64") < now_ms].copy()
    return df.reset_index(drop=True)


def add_indicators(df):
    if len(df) < 50:
        return df
    close = df["close"]
    df["ema9"] = close.ewm(span=9, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-12)
    df["rsi"] = 100 - (100 / (1 + rs))
    prev_close = close.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_sma20"].replace(0, 1e-12)
    return df


def make_signal(df, cfg):
    if len(df) < 50:
        return None
    x = df.iloc[-1]
    if pd.isna(x["atr"]) or x["atr"] <= 0 or pd.isna(x["rsi"]) or pd.isna(x["vol_ratio"]):
        return None
    if x["vol_ratio"] < cfg["min_volume_ratio"]:
        return None
    # Trend-following confirmation. Signals are based only on fully closed candles.
    if x["ema9"] > x["ema21"] and x["close"] > x["ema9"] \
       and cfg["rsi_long_min"] <= x["rsi"] <= cfg["rsi_long_max"]:
        return {"side":"LONG", "entry":float(x["close"]), "atr":float(x["atr"]),
                "rsi":float(x["rsi"]), "vol_ratio":float(x["vol_ratio"])}
    if x["ema9"] < x["ema21"] and x["close"] < x["ema9"] \
       and cfg["rsi_short_min"] <= x["rsi"] <= cfg["rsi_short_max"]:
        return {"side":"SHORT", "entry":float(x["close"]), "atr":float(x["atr"]),
                "rsi":float(x["rsi"]), "vol_ratio":float(x["vol_ratio"])}
    return None


def fresh_state(cfg):
    return {
        "balance": float(cfg["starting_balance"]),
        "positions": {},
        "realized_pnl": 0.0,
        "daily_realized_pnl": 0.0,
        "day": str(date.today()),
        "paused": False,
        "last_scan": None
    }


def load_state(cfg):
    if not STATE_PATH.exists():
        return fresh_state(cfg)
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        log.exception("状态文件读取失败，停止以避免状态错乱。")
        raise


def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_csv(path, row, headers):
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not exists:
            w.writeheader()
        w.writerow(row)


def log_signal(symbol, sig):
    append_csv(SIGNALS_PATH, {
        "time": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "side": sig["side"], "entry": sig["entry"],
        "rsi": round(sig["rsi"], 2), "volume_ratio": round(sig["vol_ratio"], 2)
    }, ["time","symbol","side","entry","rsi","volume_ratio"])


def close_position(state, symbol, exit_price, reason, cfg):
    p = state["positions"].pop(symbol)
    direction = 1 if p["side"] == "LONG" else -1
    gross = direction * (exit_price - p["entry"]) * p["qty"]
    fees = (p["entry"] + exit_price) * p["qty"] * cfg["fee_rate"]
    net = gross - fees
    state["balance"] += net
    state["realized_pnl"] += net
    state["daily_realized_pnl"] += net
    append_csv(TRADES_PATH, {
        "close_time": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol, "side": p["side"], "entry": p["entry"],
        "exit": exit_price, "qty": p["qty"], "gross_pnl": round(gross, 6),
        "fees_est": round(fees, 6), "net_pnl": round(net, 6), "reason": reason
    }, ["close_time","symbol","side","entry","exit","qty","gross_pnl","fees_est","net_pnl","reason"])
    log.info("模拟平仓 %s %s | 出场 %.8f | 净盈亏 %.4f USDT | %s",
             symbol, p["side"], exit_price, net, reason)


def manage_positions(state, prices, cfg):
    for symbol in list(state["positions"].keys()):
        p = state["positions"][symbol]
        if symbol not in prices:
            continue
        price = prices[symbol]
        # Conservative bar-based handling: if both stop and target could be touched,
        # stop is assumed first. Live fills may differ due to gaps/slippage.
        candles = get_candles(symbol, cfg["timeframe"], 3)
        if len(candles) < 1:
            continue
        bar = candles.iloc[-1]
        hi, lo = float(bar["high"]), float(bar["low"])
        if p["side"] == "LONG":
            if lo <= p["stop"]:
                close_position(state, symbol, p["stop"] * (1-cfg["slippage_rate"]), "止损", cfg)
            elif hi >= p["target"]:
                close_position(state, symbol, p["target"] * (1-cfg["slippage_rate"]), "止盈", cfg)
        else:
            if hi >= p["stop"]:
                close_position(state, symbol, p["stop"] * (1+cfg["slippage_rate"]), "止损", cfg)
            elif lo <= p["target"]:
                close_position(state, symbol, p["target"] * (1+cfg["slippage_rate"]), "止盈", cfg)


def open_position(state, symbol, sig, cfg):
    if symbol in state["positions"] or len(state["positions"]) >= cfg["max_open_positions"]:
        return
    entry = sig["entry"] * (1 + cfg["slippage_rate"] if sig["side"] == "LONG" else 1 - cfg["slippage_rate"])
    stop_dist = sig["atr"] * cfg["atr_stop_multiplier"]
    if stop_dist <= 0:
        return
    risk_cash = state["balance"] * cfg["risk_per_trade_pct"]
    qty = risk_cash / stop_dist
    max_notional = state["balance"] * cfg["simulated_leverage"]
    qty = min(qty, max_notional / entry)
    if qty <= 0:
        return
    if sig["side"] == "LONG":
        stop, target = entry - stop_dist, entry + stop_dist * cfg["take_profit_r_multiple"]
    else:
        stop, target = entry + stop_dist, entry - stop_dist * cfg["take_profit_r_multiple"]
    state["positions"][symbol] = {
        "side": sig["side"], "entry": entry, "qty": qty,
        "stop": stop, "target": target,
        "opened_at": datetime.now(timezone.utc).isoformat()
    }
    log.info("模拟开仓 %s %s | 入场 %.8f | 数量 %.6f | 止损 %.8f | 止盈 %.8f",
             symbol, sig["side"], entry, qty, stop, target)


def scan_once(cfg, state):
    today = str(date.today())
    if state.get("day") != today:
        state["day"] = today
        state["daily_realized_pnl"] = 0.0
        state["paused"] = False
    daily_limit = cfg["starting_balance"] * cfg["max_daily_loss_pct"]
    if state["daily_realized_pnl"] <= -daily_limit:
        state["paused"] = True
        log.warning("触发每日亏损熔断，停止新开仓。")
    ranked = get_universe(cfg)
    # 已持仓交易对即使跌出成交额排名，也必须继续监控止盈止损。
    ranked_symbols = {symbol for symbol, _ in ranked}
    for held_symbol in state["positions"]:
        if held_symbol not in ranked_symbols:
            ranked.append((held_symbol, 0.0))
    log.info("扫描合约数：%d（含全部已持仓交易对）", len(ranked))
    prices = {}
    for symbol, qv in ranked:
        try:
            candles = add_indicators(get_candles(symbol, cfg["timeframe"], 100))
            if len(candles) < 50:
                continue
            last = float(candles.iloc[-1]["close"])
            prices[symbol] = last
            if symbol in state["positions"]:
                continue
            sig = make_signal(candles, cfg)
            if sig:
                log_signal(symbol, sig)
                log.info("信号 %s %s | RSI %.1f | 量能 %.2fx | 24h成交额 %.0f USDT",
                         symbol, sig["side"], sig["rsi"], sig["vol_ratio"], qv)
                if not state["paused"]:
                    open_position(state, symbol, sig, cfg)
        except requests.RequestException as e:
            log.warning("%s 行情请求失败：%s", symbol, e)
        except Exception:
            log.exception("%s 分析失败", symbol)
        time.sleep(0.08)
    manage_positions(state, prices, cfg)
    state["last_scan"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    log.info("模拟账户余额 %.2f USDT | 累计已实现盈亏 %.2f | 持仓 %d",
             state["balance"], state["realized_pnl"], len(state["positions"]))


def main():
    cfg = load_config()
    state = load_state(cfg)
    log.warning("当前为 PAPER 模拟交易模式：不会发送任何真实订单。")
    log.info("初始资金 %.2f USDT | 周期 %s | 每笔风险 %.2f%%",
             cfg["starting_balance"], cfg["timeframe"], cfg["risk_per_trade_pct"]*100)
    while True:
        start = time.time()
        try:
            scan_once(cfg, state)
        except requests.RequestException as e:
            log.error("币安行情接口异常：%s", e)
        except KeyboardInterrupt:
            log.info("收到停止指令，保存状态。")
            save_state(state)
            break
        except Exception:
            log.exception("本轮扫描异常")
        elapsed = time.time() - start
        time.sleep(max(5, cfg["scan_interval_seconds"] - elapsed))


if __name__ == "__main__":
    main()

{
  "starting_balance": 1000.0,
  "timeframe": "15m",
  "scan_interval_seconds": 90,
  "min_24h_quote_volume": 10000000,
  "max_symbols_to_analyze": 40,
  "max_open_positions": 3,
  "risk_per_trade_pct": 0.005,
  "max_daily_loss_pct": 0.02,
  "simulated_leverage": 1.0,
  "fee_rate": 0.0005,
  "slippage_rate": 0.0002,
  "atr_stop_multiplier": 1.8,
  "take_profit_r_multiple": 2.0,
  "min_volume_ratio": 1.2,
  "rsi_long_min": 50,
  "rsi_long_max": 68,
  "rsi_short_min": 32,
  "rsi_short_max": 50,
  "dry_run_only": true
}
