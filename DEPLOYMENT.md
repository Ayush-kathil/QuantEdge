# QuantEdge v3: Deployment & Operations Guide

This guide covers deployment procedures for **Streamlit Community Cloud**, local development environments, and institutional containerized infrastructures.

---

## 1. Local Environment Setup

### System Prerequisites
- **Python:** 3.11 or 3.12 (64-bit)
- **Git**
- **Operating System:** Windows, macOS, or Linux

### Quick Start
```bash
# 1. Clone repository
git clone https://github.com/shivam/QuantEdge.git
cd QuantEdge

# 2. Create isolated virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Upgrade pip and install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch the quantitative terminal
python -m streamlit run app.py
```

*Note for Windows users:* If `streamlit` is not directly on your system PATH, launching via `python -m streamlit run app.py` ensures the Python interpreter directly invokes the installed Streamlit module without environment path conflicts.

---

## 2. Streamlit Community Cloud Deployment

QuantEdge is configured for zero-configuration, single-click deployment to Streamlit Community Cloud:

1. **Push Repository to GitHub:**
   Ensure all files are committed to your GitHub repository on branch `main`.

2. **Access Streamlit Cloud:**
   - Log in to [share.streamlit.io](https://share.streamlit.io).
   - Click **New app**.

3. **Configure App Settings:**
   - **Repository:** `<your-username>/QuantEdge`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **Python version:** Select `3.11` or `3.12`

4. **Deploy:**
   - Click **Deploy!**.
   - Streamlit Cloud will read `requirements.txt`, install all scientific dependencies (`pandas`, `numpy`, `scipy`, `statsmodels`, `plotly`, `reportlab`, `yfinance`), and launch the terminal.

---

## 3. Offline Mode & Network Resilience

In corporate firewall environments or during financial API maintenance windows:
- QuantEdge includes a deterministic offline test fixture located at `sample_data/sample_sp500.csv`.
- If Yahoo Finance rate-limits or is unreachable, simply switch the sidebar **Source Provider** to **"Pre-Packaged Offline Sample"** or upload any standard OHLCV CSV file.
- The platform never crashes on network failure and handles unavailable tickers gracefully.

---

## 4. Verification & Testing

Verify that all mathematical models, futures margin calculations, and statistical moments pass before deployment:

```bash
# Run the complete test suite (104 tests)
python -m pytest -v

# Run individual domain suites
python -m pytest tests/test_final_quantitative_validation.py -v
python -m pytest tests/test_audit_metrics_v3.py -v
python -m pytest tests/test_futures_accounting.py -v
python -m pytest tests/test_indian_markets.py -v
python -m pytest tests/test_portfolio_research.py -v
python -m pytest tests/test_reporting.py -v
```

---

## 5. Security & Provenance Disclosures

- **Cryptographic Fingerprinting:** Every ingested dataset is hashed using SHA-256 over normalized timestamps and OHLCV float values. The SHA-256 hash guarantees dataset identity, not underlying exchange authenticity.
- **EOD Historical Data:** Financial market data provided via Yahoo Finance is delayed end-of-day (EOD) data and must not be treated as a zero-latency real-time tick stream.
- **Continuous Futures Rollover:** Continuous commodity and index futures (`GC=F`, `CL=F`, `ZN=F`, `NIFTY_FUT`) represent front-month stitched series subject to roll-yield leakage (contango/backwardation) and expiration step-discontinuities.
