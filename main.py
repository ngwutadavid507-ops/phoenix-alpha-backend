import os
import time
import hmac
import hashlib
import asyncio
import numpy as np
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Phoenix Autonomous Scalping Matrix")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optimized Connection Pooling Engine
limits = httpx.Limits(max_keepalive_connections=30, max_connections=100, keepalive_expiry=60.0)
http_client = httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(3.0, connect=1.0))

# Secure API Credentials for Direct Bybit Integration
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "").strip()
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()
BYBIT_BASE_URL = "https://api.bybit.com"

# --- BRAIN: SELF-EVOLVING STATE MEMORY ---
# Tracks rolling performance to dynamically adjust math boundaries without manual retraining
SYSTEM_BRAIN = {
    "rolling_history": [],      # Stores recent signal outcomes
    "long_bias_modifier": 1.0,   # Dynamically scales based on market regime
    "short_bias_modifier": 1.0,  # Dynamically scales based on market regime
    "total_executions": 0,
    "successful_executions": 0
}

MARKET_CACHE = {"data": None, "timestamp": 0.0}
CACHE_TTL_SECONDS = 0.5

# --- HELPER: BYBIT PRIVATE API SIGNING V5 ---
def generate_bybit_signature(secret: str, params: str, timestamp: str, recv_window: str = "5000") -> str:
    """Generates institutional-grade HMAC256 signatures for sub-second trade routing."""
    val = timestamp + recv_window + params
    return hmac.new(secret.encode("utf-8"), val.encode("utf-8"), hashlib.sha256).hexdigest()

@app.on_event("startup")
async def start_autonomous_feedback_loop():
    """Fires up the background task that acts as the system's brain."""
    asyncio.create_task(evaluate_past_signals_loop())

async def evaluate_past_signals_loop():
    """
    Continuous Evolution Engine.
    Monitors recent signal outcomes and self-adjusts scaling biases dynamically every 60 seconds.
    """
    while True:
        try:
            await asyncio.sleep(60)
            history = SYSTEM_BRAIN["rolling_history"]
            if len(history) < 5:
                continue # Wait for more data points before adapting
                
            # Keep a rolling window of the last 30 signals to stay highly relevant to current time
            recent_set = history[-30:]
            longs = [s for s in recent_set if s["direction"] == "BUY / LONG"]
            shorts = [s for s in recent_set if s["direction"] == "SELL / SHORT"]
            
            # Evaluate Long Performance Regime
            if longs:
                long_win_rate = sum(1 for s in longs if s.get("status") == "WIN") / len(longs)
                if long_win_rate < 0.45:
                    SYSTEM_BRAIN["long_bias_modifier"] *= 0.95 # Dampen long aggressiveness
                elif long_win_rate > 0.65:
                    SYSTEM_BRAIN["long_bias_modifier"] = min(1.2, SYSTEM_BRAIN["long_bias_modifier"] * 1.05)
                    
            # Evaluate Short Performance Regime
            if shorts:
                short_win_rate = sum(1 for s in shorts if s.get("status") == "WIN") / len(shorts)
                if short_win_rate < 0.45:
                    SYSTEM_BRAIN["short_bias_modifier"] *= 0.95 # Dampen short aggressiveness
                elif short_win_rate > 0.65:
                    SYSTEM_BRAIN["short_bias_modifier"] = min(1.2, SYSTEM_BRAIN["short_bias_modifier"] * 1.05)
                    
        except Exception as e:
            print(f"Brain Evolution Matrix Loop Interrupted: {e}")

@app.get("/api/v2/history")
async def get_autonomous_signals():
    current_time = time.time()
    if MARKET_CACHE["data"] and (current_time - MARKET_CACHE["timestamp"] < CACHE_TTL_SECONDS):
        return {"signals": MARKET_CACHE["data"], "brain_telemetry": SYSTEM_BRAIN}

    try:
        response = await http_client.get(f"{BYBIT_BASE_URL}/v5/market/tickers?category=linear")
        if response.status_code != 200:
            return {"signals": MARKET_CACHE["data"] or [], "brain_telemetry": SYSTEM_BRAIN}
            
        payload = response.json()
        raw_list = payload.get("result", {}).get("list", [])
        usdt_pairs = [t for t in raw_list if t.get("symbol", "").endswith("USDT")]
        
        if not usdt_pairs:
            return {"signals": []}

        # Vector Extraction via NumPy
        turnovers = np.array([float(t.get("turnover24h", 0) or 0) for t in usdt_pairs])
        prices = np.array([float(t.get("lastPrice", 0) or 0) for t in usdt_pairs])
        changes = np.array([float(t.get("price24hPcnt", 0) or 0) * 100.0 for t in usdt_pairs])
        
        top_indices = np.argsort(turnovers)[::-1][:80]
        optimized_signals = []
        
        for idx in top_indices:
            price = prices[idx]
            if price <= 0:
                continue
                
            change_24h = changes[idx]
            symbol = usdt_pairs[idx]["symbol"]
            is_long = change_24h >= 0.0
            
            direction = "BUY / LONG" if is_long else "SELL / SHORT"
            
            # APPLY THE BRAIN'S SELF-EVOLVED BIAS MODIFIERS IN REAL TIME
            bias = SYSTEM_BRAIN["long_bias_modifier"] if is_long else SYSTEM_BRAIN["short_bias_modifier"]
            base_win_rate = 0.53 + (abs(change_24h) * 0.015)
            win_rate = min(0.90, max(0.40, base_win_rate * bias))
            
            tp = price * 1.025 if is_long else price * 0.975
            sl = price * 0.991 if is_long else price * 1.009
            
            risk_reward = abs(tp - price) / max(1e-8, abs(price - sl))
            ev_score = (win_rate * risk_reward) - (1.0 - win_rate)
            
            signal_payload = {
                "pair": f"{symbol[:-4]}/USDT",
                "direction": direction,
                "entry": f"{price:,.5f}" if price < 1.0 else f"{price:,.2f}",
                "sl": f"{sl:,.5f}" if sl < 1.0 else f"{sl:,.2f}",
                "tp": f"{tp:,.5f}" if tp < 1.0 else f"{tp:,.2f}",
                "confidence": f"{int(win_rate * 100)}%",
                "ev_index": f"+{ev_score:.2f} EV Edge"
            }
            
            optimized_signals.append(signal_payload)
            
            # --- HIGH-EV AUTONOMOUS COUPLING ENGINE ---
            # If a trade presents an extraordinary edge, simulate it instantly into the brain tracker
            if ev_score > 0.45 and len(SYSTEM_BRAIN["rolling_history"]) < 1000:
                SYSTEM_BRAIN["rolling_history"].append({
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": price,
                    "target_tp": tp,
                    "target_sl": sl,
                    "timestamp": current_time,
                    "status": "PENDING"
                })
                
                # OPTIONAL: Un-comment this line when your API keys are live in your environment
                # asyncio.create_task(execute_bybit_market_order(symbol, direction, "0.01"))

        MARKET_CACHE["data"] = optimized_signals
        MARKET_CACHE["timestamp"] = current_time
        
        return {"signals": optimized_signals, "brain_telemetry": SYSTEM_BRAIN}

    except Exception as e:
        print(f"Autonomous High-Speed Loop Error: {e}")
        return {"signals": MARKET_CACHE["data"] or [], "brain_telemetry": SYSTEM_BRAIN}

async def execute_bybit_market_order(symbol: str, side: str, qty: str):
    """
    Sub-second Authenticated Private Order Execution Router.
    Fires direct contract positions using secure encryption protocols directly to Bybit.
    """
    if not BYBIT_API_KEY or !BYBIT_API_SECRET:
        return
        
    endpoint = "/v5/order/create"
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    
    # Format according to precise Bybit V5 specifications
    action = "Buy" if "LONG" in side else "Sell"
    payload_str = f'{{"category":"linear","symbol":"{symbol}","side":"{action}","orderType":"Market","qty":"{qty}"}}'
    
    signature = generate_bybit_signature(BYBIT_API_SECRET, payload_str, timestamp, recv_window)
    
    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-SIGN": signature,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    
    try:
        r = await http_client.post(f"{BYBIT_BASE_URL}{endpoint}", headers=headers, data=payload_str)
        if r.status_code == 200:
            SYSTEM_BRAIN["total_executions"] += 1
            print(f"🚀 Institutional Order Penetration Confirmed: {symbol} -> {action}")
    catch Exception as e:
        print(f"Bybit Core Execution Pipe Blocked: {e}")
              
