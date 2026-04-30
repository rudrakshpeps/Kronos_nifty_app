import sys
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime as dt_obj, timedelta

# --- 1. KRONOS SYSTEM INITIALIZATION ---
# Use relative paths for Cloud compatibility
current_dir = os.getcwd()
KRONOS_PATH = current_dir 
WEIGHTS_PATH = os.path.join(current_dir, 'weights', 'Kronos-base')

sys.path.append(KRONOS_PATH)

@st.cache_resource
def load_kronos_engine():
    """Caches the model in memory to prevent slow reloads[cite: 8]."""
    try:
        from model import Kronos, KronosTokenizer, KronosPredictor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # local_files_only=True requires the weights folder in your repo[cite: 8]
        model = Kronos.from_pretrained(WEIGHTS_PATH, local_files_only=True).to(device)
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        predictor = KronosPredictor(model, tokenizer, max_context=512)
        return predictor, device
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return None, None

# --- 2. LOGIC UTILITIES ---
def add_technicals(df):
    """Calculates indicators used for model context[cite: 8]."""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain.div(loss.replace(0, 0.001))
    df['rsi'] = 100 - (100 / (1 + rs))
    df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
    return df

def get_strategy_recommendation(pred_df):
    """Applies the Strategy Selection Matrix to prediction data[cite: 8]."""
    window = pred_df.iloc[1:min(24, len(pred_df))]
    start_price = window['open'].iloc[0]
    end_price = window['close'].iloc[-1]
    
    expected_ret = ((end_price - start_price) / start_price) * 100
    safe_high = window['high'].max()
    safe_low = window['low'].min()
    vol_width = ((safe_high - safe_low) / start_price) * 100
    
    if abs(expected_ret) <= 0.35 and vol_width <= 1.2:
        return "IRON FLY / IRON CONDOR", "success", "Maximize Theta decay while remaining range-bound.[cite: 8]"
    elif expected_ret > 0.35 and vol_width <= 2.0:
        return "BULL PUT SPREAD", "info", f"Bullish Trend. Sell Put Credit at support ({round(safe_low/50)*50}).[cite: 8]"
    elif expected_ret < -0.35 and vol_width <= 2.0:
        return "BEAR CALL SPREAD", "warning", f"Bearish Trend. Sell Call Credit at resistance ({round(safe_high/50)*50}).[cite: 8]"
    else:
        return "WAIT / STAGNANT", "error", "High Volatility or unclear direction. Avoid selling options.[cite: 8]"

# --- 3. STREAMLIT UI ---
st.set_page_config(page_title="Kronos Pro Dashboard", layout="wide")
st.title("🛡️ Kronos Pro: Strategy Advisor")

predictor, device = load_kronos_engine()

with st.sidebar:
    st.header("Terminal Settings")
    ticker = st.text_input("Ticker Symbol", "^NSEI")
    mc_samples = st.slider("Monte Carlo Samples", 10, 100, 50)
    run_btn = st.button("🚀 Analyze Market")

if run_btn and predictor:
    try:
        with st.spinner("Executing Transformer Inference..."):
            # 1. Main Data Fetching[cite: 8]
            nifty_raw = yf.download(ticker, period="60d", interval="15m", progress=False)
            if nifty_raw.empty:
                st.error("❌ Failed to fetch Nifty data. Check ticker or connection.[cite: 8]")
                st.stop()
            
            if isinstance(nifty_raw.columns, pd.MultiIndex): nifty_raw.columns = [c[0] for c in nifty_raw.columns]
            df = nifty_raw.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df.rename(columns={'datetime': 'date', 'timestamp': 'date'}, inplace=True)
            df['date'] = pd.to_datetime(df['date'])

            # 2. Correlated Data Fetching (Hang Seng)[cite: 8]
            hsi_raw = yf.download("^HSI", period="60d", interval="15m", progress=False)
            if not hsi_raw.empty:
                if isinstance(hsi_raw.columns, pd.MultiIndex): hsi_raw.columns = [c[0] for c in hsi_raw.columns]
                hsi_close = hsi_raw['Close'].rename("hsi_close")
                df = pd.merge(df, hsi_close, left_on='date', right_index=True, how='left').ffill()
            else:
                st.warning("⚠️ Hang Seng data unavailable. Using Nifty fallback.[cite: 8]")
                df['hsi_close'] = df['close']
            
            # 3. Technicals & Cleaning[cite: 8]
            df = add_technicals(df).dropna()
            
            # Critical Validation: Prevent tensor error by ensuring enough context[cite: 8]
            if len(df) < 150:
                st.error(f"❌ Need 150 candles for inference, but found only {len(df)}.[cite: 8]")
                st.stop()
            
            # 4. Market Window Alignment[cite: 8]
            now_dt = dt_obj.now()
            start_ts = now_dt.replace(hour=9, minute=15, second=0, microsecond=0)
            if now_dt.hour >= 16: start_ts += timedelta(days=1)
            if start_ts.weekday() >= 5: start_ts += timedelta(days=(7-start_ts.weekday()))
            y_ts = pd.date_range(start=start_ts, periods=24, freq='15min')

            # 5. Prediction Engine[cite: 8]
            with torch.inference_mode():
                x_df = df.tail(150).copy()
                # Ensure input tensor has data before calling model[cite: 8]
                pred_df = predictor.predict(df=x_df, x_timestamp=x_df['date'], 
                                            y_timestamp=pd.Series(y_ts), pred_len=24, sample_count=mc_samples)
            
            pred_df['date'] = y_ts.values[:len(pred_df)]

            # 6. Dashboard Results[cite: 8]
            strat, level, desc = get_strategy_recommendation(pred_df)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Strategy Type", strat)
            c2.metric("Predicted Volatility", f"{round(((pred_df['high'].max()-pred_df['low'].min())/pred_df['open'].iloc[0])*100, 2)}%")
            c3.metric("Strike Range", f"{round(pred_df['low'].min())} - {round(pred_df['high'].max())}")

            if level == "success": st.success(desc)
            elif level == "warning": st.warning(desc)
            elif level == "error": st.error(desc)
            else: st.info(desc)

            # 7. Visualization[cite: 8]
            fig, ax = plt.subplots(figsize=(12, 6))
            plt.style.use('dark_background')
            h_plt = df.tail(60).reset_index(drop=True)
            up, dn = h_plt[h_plt.close >= h_plt.open], h_plt[h_plt.close < h_plt.open]
            
            ax.bar(up.index, up.close - up.open, 0.6, bottom=up.open, color='#26a69a')
            ax.vlines(up.index, up.low, up.high, color='#26a69a')
            ax.bar(dn.index, dn.close - dn.open, 0.6, bottom=dn.open, color='#ef5350')
            ax.vlines(dn.index, dn.low, dn.high, color='#ef5350')
            
            p_idx = range(len(h_plt), len(h_plt) + len(pred_df))
            ax.plot(p_idx, pred_df['close'], color='#ff7f0e', ls='--', marker='o', ms=4, label='Forecast')
            ax.fill_between(p_idx, pred_df['low'], pred_df['high'], color='#ff7f0e', alpha=0.15, label='Volatility Cloud')
            ax.legend()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Processing Error: {e}[cite: 8]")
