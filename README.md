# 🛡️ Kronos Pro: Strategy Advisor

**Kronos Pro** is a high-precision market forecasting dashboard powered by the **Kronos Transformer** engine. It leverages multi-horizon time-series forecasting to predict **Nifty 50** intraday movements and recommends institutional-grade option selling strategies based on a proprietary volatility-trend matrix.

## 🧠 Core Technology
The system is built on state-of-the-art deep learning architectures for time-series analysis:
* **Transformer-Based Weights**: Utilizes self-attention mechanisms to identify long-range correlations in Nifty patterns.
* **Multi-Variable Correlation**: Integrates **Hang Seng (^HSI)** price action as a global context variable.
* **Uncertainty Quantification**: Uses **Monte Carlo (MC) Sampling** to generate a "Volatility Cloud" for risk assessment.

## 🎯 Strategy Selection Matrix
The engine recommends an execution plan for the **09:30 AM – 03:00 PM** window:

| Forecast Condition | Recommended Strategy | Trading Purpose |
| :--- | :--- | :--- |
| **Low Trend + Narrow Range** | **Iron Fly / Iron Condor** | Maximize Theta decay in range-bound markets. |
| **Bullish Trend + Support** | **Bull Put Spread** | Sell Put Credit at predicted low support. |
| **Bearish Trend + Resistance** | **Bear Call Spread** | Sell Call Credit at predicted high resistance. |
| **High Range / No Direction** | **Wait / Stagnant** | Avoid selling if the volatility cloud is too wide. |

## 🚀 Deployment Features
* **Streamlit Web Interface**: Interactive dashboard with real-time technical indicators (RSI, MACD).
* **Tradable Strike Ranges**: Automatically rounds support/resistance levels to the nearest **50-point strike**.
* **Cloud Optimized**: Configured for Streamlit Community Cloud with cached model loading.

## 🛠️ Installation & Setup
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Launch the app: `streamlit run app.py`.

---
**Disclaimer**: *Kronos Pro is an AI-driven analytical tool. Financial trading involves significant risk. Always validate recommendations against your personal risk management strategy.*
