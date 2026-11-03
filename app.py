
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import requests
import os

app = FastAPI(title="币安量化机器人")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html>
    <html lang="zh-CN">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="utf-8">
    <title>币安量化机器人</title>
    <body style="font-family:Arial;padding:20px;background:#101827;color:white">
    <h2>币安量化机器人</h2>
    <p>运行模式：模拟观察，不进行真实交易</p>
    <button onclick="loadData()" style="padding:12px">刷新行情</button>
    <p id="status">等待获取行情</p>
    <pre id="data" style="white-space:pre-wrap"></pre>
    <script>
    async function loadData(){
      document.getElementById('status').textContent='正在获取行情...';
      try {
        const r=await fetch('/api/market');
        const d=await r.json();
        document.getElementById('data').textContent=
          JSON.stringify(d,null,2);
        document.getElementById('status').textContent='行情获取完成';
      } catch(e) {
        document.getElementById('status').textContent='获取失败，请稍后重试';
      }
    }
    </script>
    </body></html>
    """

@app.get("/api/market")
def market():
    try:
        r = requests.get(
            "https://fapi.binance.com/fapi/v1/ticker/24hr",
            timeout=10
        )
        r.raise_for_status()
        items = [
            {
                "symbol": x["symbol"],
                "price": x["lastPrice"],
                "change24h": x["priceChangePercent"]
            }
            for x in r.json()
            if x["symbol"].endswith("USDT")
        ]
        items.sort(
            key=lambda x: abs(float(x["change24h"])),
            reverse=True
        )
        return {"mode": "模拟观察", "markets": items[:20]}
    except Exception:
        return {"error": "行情暂时无法获取，请稍后重试"}
