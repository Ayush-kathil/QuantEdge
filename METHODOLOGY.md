# QuantEdge v3.1: Quantitative Methodology & Mathematical Specifications

## 1. Executive Summary

This document provides complete mathematical formulations, statistical definitions, boundary conventions, and execution mechanics implemented across the **QuantEdge v3.1** institutional quantitative market research and backtesting engine.

Every equation is implemented in pure, audited Python using standard scientific libraries (`numpy`, `pandas`, `scipy`, `statsmodels`).

---

## 2. Return Calculations & Compounding (Metrics 1–4)

### Metric 1: Arithmetic (Simple) Returns
Given a price series $P_t$ for $t = 0, 1, \dots, T$:
$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}} = \frac{P_t}{P_{t-1}} - 1$$
- **Implementation:** `analytics/returns.py` -> `compute_daily_returns(series, method="arithmetic")`
- **Boundary Handling:** Returns `NaN` for $t=0$. Undefined if $P_{t-1} \le 0$.

### Metric 2: Logarithmic (Continuously Compounded) Returns
$$r_t = \ln\left(\frac{P_t}{P_{t-1}}\right) = \ln(P_t) - \ln(P_{t-1})$$
- **Implementation:** `analytics/returns.py` -> `compute_daily_returns(series, method="log")`
- **Properties:** Additive across time: $\sum_{t=1}^T r_t = \ln(P_T / P_0)$.

### Metric 3: Cumulative Total Returns
For arithmetic returns:
$$C_t = \prod_{i=1}^t (1 + R_i) - 1$$
For logarithmic returns:
$$C_t = \exp\left(\sum_{i=1}^t r_i\right) - 1$$
- **Implementation:** `analytics/returns.py` -> `compute_cumulative_returns(returns, method)`
- **Base Period:** Starts deterministically at $0.0$ ($0\%$).

### Metric 4: Compound Annual Growth Rate (CAGR)
Annualized geometric progression rate of capital:
- **Trading Day Convention (Default for Strategy Backtests):**
  $$\text{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{252}{N}} - 1$$
  where $N$ is the number of trading periods (bars), and $252$ is the standard annual trading day convention.
- **Calendar Day Convention (Alternative for Multi-Year Fund Auditing):**
  $$\text{CAGR} = \left(\frac{V_T}{V_0}\right)^{\frac{365.25}{D_{\text{calendar}}}} - 1$$
- **Boundary & Singularity Protections:**
  - **Capital Wipeout ($V_T \le 0$):** Returns $-1.0$ ($-100\%$).
  - **Invalid Base ($V_0 \le 0$):** Returns $0.0$.
  - **Short Duration Safeguard:** If $N < 15$ bars or elapsed time $< 0.05$ years, returns simple total return $(V_T - V_0)/V_0$ to prevent compounding illusions on high-frequency noise.
- **Implementation:** `analytics/returns.py` -> `compute_cagr()`

---

## 3. Realized Volatility Estimators (Metrics 5–8 + Yang-Zhang)

### Metric 5: Annualized Close-to-Close Realized Volatility
Sample standard deviation of daily arithmetic returns scaled by the square root of annual trading days:
$$\hat{\sigma} = \sqrt{\frac{1}{N-1} \sum_{t=1}^N (R_t - \bar{R})^2}$$
$$\sigma_{\text{ann}} = \hat{\sigma} \times \sqrt{252}$$
- **Degrees of Freedom:** Sample standard deviation uses Bessel's correction ($ddof=1$).
- **Implementation:** `analytics/volatility.py` -> `compute_realized_volatility()`

### Metric 6: Parkinson High-Low Volatility Estimator
An extreme-value volatility estimator utilizing daily High ($H_t$) and Low ($L_t$) prices. Approximately 5 times more statistically efficient than the close-to-close estimator:
$$\sigma_P = \sqrt{\frac{1}{4 \ln(2) \cdot N} \sum_{t=1}^N \left(\ln\frac{H_t}{L_t}\right)^2} \times \sqrt{252}$$
- **Mathematical Limitations & Jump Bias:** Assumes continuous Brownian motion with zero drift and zero opening jumps. If a market exhibits overnight opening gaps ($O_t \neq C_{t-1}$), Parkinson volatility understates total volatility.
- **Implementation:** `analytics/volatility.py` -> `compute_parkinson_volatility()`

### Metric 7: Garman-Klass OHLC Volatility Estimator
Extends Parkinson by incorporating Open ($O_t$) and Close ($C_t$) prices, providing approximately 8 times the efficiency of close-to-close volatility:
$$\sigma_{GK} = \sqrt{\frac{1}{N} \sum_{t=1}^N \left[ \frac{1}{2}\left(\ln\frac{H_t}{L_t}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_t}{O_t}\right)^2 \right]} \times \sqrt{252}$$
- **Limitations:** Does not capture overnight opening gaps ($O_t / C_{t-1}$).
- **Implementation:** `analytics/volatility.py` -> `compute_garman_klass_volatility()`

### Yang-Zhang Jump-Robust Volatility Estimator
A composite minimum-variance unbiased estimator that accounts for overnight opening jumps, continuous intraday open-to-close volatility, and non-zero drift:
$$\sigma_{YZ}^2 = \sigma_{\text{overnight}}^2 + k \cdot \sigma_{\text{open-to-close}}^2 + (1 - k) \cdot \sigma_{RS}^2$$
where:
$$\sigma_{\text{overnight}}^2 = \frac{1}{N-1}\sum_{t=1}^N \left(\ln\frac{O_t}{C_{t-1}} - \bar{o}\right)^2, \quad \sigma_{\text{open-to-close}}^2 = \frac{1}{N-1}\sum_{t=1}^N \left(\ln\frac{C_t}{O_t} - \bar{c}\right)^2$$
$$\sigma_{RS}^2 = \frac{1}{N}\sum_{t=1}^N \left[ \ln\frac{H_t}{C_t}\ln\frac{H_t}{O_t} + \ln\frac{L_t}{C_t}\ln\frac{L_t}{O_t} \right], \quad k = \frac{0.34}{1.34 + \frac{N+1}{N-1}}$$
- **Implementation:** `analytics/volatility.py` -> `compute_yang_zhang_volatility()`

### Metric 8: Exponentially Weighted Moving Average (EWMA) Volatility
RiskMetrics recursive volatility model giving greater statistical weight to recent price innovations:
$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) R_t^2$$
- Standard decay factor: $\lambda = 0.94$ (daily frequency).
- **Implementation:** `analytics/volatility.py` -> `compute_ewma_volatility()`

---

## 4. Technical Indicators (Metrics 9–14)

### Metric 9: Simple Moving Average (SMA)
$$\text{SMA}_t(N) = \frac{1}{N} \sum_{i=0}^{N-1} P_{t-i}$$

### Metric 10: Exponential Moving Average (EMA)
$$\text{EMA}_t(N) = \alpha P_t + (1 - \alpha) \text{EMA}_{t-1}(N), \quad \alpha = \frac{2}{N + 1}$$

### Metric 11: Relative Strength Index (RSI)
J. Welles Wilder's modified exponential smoothing of price changes:
$$\Delta P_t = P_t - P_{t-1}, \quad U_t = \max(\Delta P_t, 0), \quad D_t = \max(-\Delta P_t, 0)$$
$$\bar{U}_t = \frac{1}{N} U_t + \left(1 - \frac{1}{N}\right) \bar{U}_{t-1}, \quad \bar{D}_t = \frac{1}{N} D_t + \left(1 - \frac{1}{N}\right) \bar{D}_{t-1}$$
$$\text{RS}_t = \frac{\bar{U}_t}{\bar{D}_t}, \quad \text{RSI}_t = 100 - \frac{100}{1 + \text{RS}_t}$$
- **Boundary Handling:** If $\bar{D}_t = 0$, $\text{RSI} = 100.0$. If $\bar{U}_t = 0$, $\text{RSI} = 0.0$. If $\bar{U}_t = \bar{D}_t = 0$ (completely flat price action), $\text{RSI} = 50.0$.

### Metric 12: Moving Average Convergence Divergence (MACD)
$$\text{MACD Line}_t = \text{EMA}_t(12) - \text{EMA}_t(26)$$
$$\text{Signal Line}_t = \text{EMA}_t(\text{MACD Line}, 9)$$
$$\text{Histogram}_t = \text{MACD Line}_t - \text{Signal Line}_t$$

### Metric 13: Average True Range (ATR)
$$\text{TR}_t = \max\left( H_t - L_t, \, |H_t - C_{t-1}|, \, |L_t - C_{t-1}| \right)$$
$$\text{ATR}_t = \frac{1}{N} \text{TR}_t + \left(1 - \frac{1}{N}\right) \text{ATR}_{t-1}$$

### Metric 14: Bollinger Bands
$$\text{Middle Band}_t = \text{SMA}_t(N)$$
$$\text{Upper Band}_t = \text{Middle Band}_t + k \cdot \sigma_t(N)$$
$$\text{Lower Band}_t = \text{Middle Band}_t - k \cdot \sigma_t(N)$$
$$\text{Bandwidth}_t = \frac{\text{Upper Band}_t - \text{Lower Band}_t}{\text{Middle Band}_t}$$
Standard parameters: $N = 20, k = 2.0$.

---

## 5. Statistical Moments & Tail Risk (Metrics 15–20)

### Metric 15: Skewness (Fisher-Pearson Unbiased Moment)
Measures the asymmetry of the return distribution around its mean:
$$S = \frac{\frac{1}{n} \sum_{i=1}^n (R_i - \bar{R})^3}{\left[\frac{1}{n} \sum_{i=1}^n (R_i - \bar{R})^2\right]^{3/2}} \times \frac{\sqrt{n(n-1)}}{n-2}$$
- **Interpretation:** $S < 0$ denotes left-skewness (downside crash risk, typical in equity indices).

### Metric 16: Excess Kurtosis (Fisher Definition)
Measures the tail fatness relative to a normal distribution:
$$K_{\text{excess}} = \frac{n(n+1)}{(n-1)(n-2)(n-3)} \sum_{i=1}^n \left(\frac{R_i - \bar{R}}{s}\right)^4 - \frac{3(n-1)^2}{(n-2)(n-3)}$$
- **Interpretation:** $K > 0$ denotes leptokurtosis (fat tails, higher probability of extreme tail risk events than Gaussian curves). Normal distribution has $K = 0.0$.

### Metric 17: Jarque-Bera Test of Normality
Tests whether skewness and excess kurtosis jointly match a normal distribution:
$$JB = \frac{n}{6}\left(S^2 + \frac{K^2}{4}\right) \sim \chi^2(2)$$
- **Asymptotic Caveat in Finite Financial Samples:** While $p < 0.05$ strictly rejects Gaussianity, $p > 0.05$ in finite historical samples fails to reject $H_0$, but does NOT mathematically prove underlying Gaussianity.

### Metric 18: Historical Value-at-Risk (VaR)
Non-parametric empirical quantile of negative returns:
$$\text{VaR}_\alpha = -Q_{1-\alpha}(R)$$
Expressed as a positive fraction (e.g. $0.02 = 2.0\%$ loss threshold at 95% confidence).

### Metric 19: Parametric (Gaussian) Value-at-Risk
$$\text{VaR}_\alpha^{\text{param}} = -(\mu - z_\alpha \cdot \sigma)$$
where $z_{0.95} \approx 1.64485, z_{0.99} \approx 2.32635$.

### Metric 20: Conditional Value-at-Risk (CVaR / Expected Shortfall)
The expected loss given that loss exceeds the Value-at-Risk threshold:
$$\text{CVaR}_\alpha = \mathbb{E}\left[ -R \mid -R \ge \text{VaR}_\alpha \right] = \frac{1}{|T_\alpha|} \sum_{R_t \le -\text{VaR}_\alpha} (-R_t)$$
- **Mathematical Property:** $\text{CVaR}_\alpha \ge \text{VaR}_\alpha$ always holds.

---

## 6. Risk-Adjusted Performance Metrics (Metrics 21–26)

### Metric 21: Sharpe Ratio
$$\text{Sharpe} = \frac{\mathbb{E}[R_s - R_f] \times 252}{\sigma_s \times \sqrt{252}}$$
- **Zero Volatility Safeguard:** If $\sigma_s \le 10^{-6}$, Sharpe is constrained to $0.0$.

### Metric 22: Sortino Ratio
Measures return relative to downside semideviation below the daily Minimum Acceptable Return ($\text{MAR} = R_f / 252$):
$$\sigma_{\text{downside}} = \sqrt{\frac{1}{N} \sum_{t=1}^N \left(\min(R_t - \text{MAR}, \, 0)\right)^2} \times \sqrt{252}$$
$$\text{Sortino} = \frac{(\bar{R}_s - R_f) \times 252}{\sigma_{\text{downside}}}$$
- **Zero Downside Handling:** If $\sigma_{\text{downside}} \le 10^{-6}$ and excess return $> 0$, Sortino is mathematically infinite (`float("inf")`), flagged as `"Inf (Zero Downside Vol)"`.

### Metric 23: Calmar Ratio
$$\text{Calmar} = \frac{\text{CAGR}}{\text{Max Drawdown}}$$

### Metric 24: Maximum Drawdown (MDD)
Given the cumulative equity curve $E_t$:
$$\text{HWM}_t = \max_{0 \le i \le t} E_i, \quad \text{DD}_t = \frac{\text{HWM}_t - E_t}{\text{HWM}_t}$$
$$\text{MDD} = \max_{0 \le t \le T} \text{DD}_t$$

### Metric 25: Profit Factor
Evaluated post-friction on closed round-trip trades:
$$\text{Profit Factor} = \frac{\sum \text{Net Profits}}{\sum |\text{Net Losses}|}$$
- If gross losses $= 0$ and gross profits $> 0$, Profit Factor is `float("inf")`.

### Metric 26: Win Rate
$$\text{Win Rate} = \frac{N_{\text{winning trades}}}{N_{\text{total closed trades}}}$$

---

## 7. Institutional Futures Mechanics & Accounting

### Contract Specifications & Notional Value
- **Notional Contract Value:** $\text{Notional}_t = P_t \times \text{Multiplier}$
- **Tick Value:** $\text{Tick Value} = \text{Multiplier} \times \text{Tick Size}$

### Dynamic Historical Lot Size Schedule (NSE Circular Compliance)
Regulatory lot size revisions alter contract notional value across time. Hardcoding static sizes corrupts backtest accuracy. QuantEdge implements dynamic date-aware lot size tracking:
- **NIFTY 50 Futures (`NIFTY_FUT`):**
  - Pre-2021: **75**
  - July 2021 – April 2024: **50**
  - April 2024 – November 19, 2024: **25**
  - November 20, 2024 onwards (SEBI Index Derivatives Framework): **75**
- **BANK NIFTY Futures (`BANKNIFTY_FUT`):**
  - Pre-July 2023: **25**
  - July 2023 – November 19, 2024: **15**
  - November 20, 2024 onwards: **30**

### Daily Mark-to-Market (MTM) Variation Margin
Positions are settled daily at the official closing price:
$$\Delta \text{MTM}_t = \text{Position}_{t-1} \times \text{Multiplier} \times (P_t - P_{t-1})$$
- Cash balance updates daily by $\Delta \text{MTM}_t$.
- **Margin Call:** Triggered when Account Equity drops below the Maintenance Margin Requirement:
  $$\text{Maintenance Margin}_t = |\text{Position}_t| \times P_t \times \text{Multiplier} \times \text{Maintenance Margin Rate}$$

---

## 8. Portfolio Optimization & Covariance Condition Number Regularization

### Markowitz Mean-Variance Framework
For expected annualized returns vector $\boldsymbol{\mu}$ and annualized covariance matrix $\boldsymbol{\Sigma}$:
$$\max_{\mathbf{w}} \frac{\mathbf{w}^T \boldsymbol{\mu} - R_f}{\sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}} \quad \text{s.t.} \quad \sum_{i=1}^n w_i = 1, \quad 0 \le w_i \le 1$$

### Condition Number Auditing & Shrinkage
The sample covariance matrix $\hat{\boldsymbol{\Sigma}}$ often suffers from estimation noise and near-collinearity.
- **Spectral Condition Number:**
  $$\kappa(\boldsymbol{\Sigma}) = \frac{\lambda_{\max}}{\lambda_{\min}}$$
- If $\kappa(\boldsymbol{\Sigma}) > 10^4$ or $\lambda_{\min} \le 10^{-6}$, the covariance matrix is ill-conditioned, leading to numerical divergence in SLSQP optimizers.
- **Regularization via Convex Shrinkage:**
  $$\boldsymbol{\Sigma}_{\text{reg}} = (1 - \delta) \hat{\boldsymbol{\Sigma}} + \delta \operatorname{diag}(\hat{\boldsymbol{\Sigma}}) + \epsilon \mathbf{I}$$
  where $\delta = 0.05, \epsilon = 10^{-6}$. This guarantees strict positive definiteness ($\lambda_{\min} > 0$) and finite condition numbers.

### Percentage Risk Contribution (%RC)
$$RC_i = w_i \frac{(\boldsymbol{\Sigma} \mathbf{w})_i}{\sigma_p}, \quad \%RC_i = \frac{RC_i}{\sigma_p}$$
Ensures total portfolio risk is completely decomposed across assets: $\sum_{i=1}^n \%RC_i = 100\%$.

---

## 9. Backtesting & Execution Engine

### Execution Timing (Look-Ahead Bias Elimination)
1. **Signal Generation:** Signal for day $t$ is calculated strictly at the close of bar $t$ using information set $\mathcal{F}_t = \{P_0, \dots, P_t\}$.
2. **Trade Execution:** The order executes at the **Open of bar $t+1$** with slippage and transaction commissions applied.
3. No bar $t$ price can ever be filled on bar $t$ signal generation.

### Short Accounting Balance Sheet Conservation
When short $S$ shares at price $P_e$:
$$\text{Cash Balance} = \text{Initial Cash} + S \cdot P_e - \text{Commissions}$$
$$\text{Liability}_t = S \cdot P_t$$
$$\text{Equity}_t = \text{Cash Balance} - \text{Liability}_t$$
Preserves fundamental conservation of value across all market regimes.
