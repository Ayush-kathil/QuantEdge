# Release Notes: QuantEdge v3.1
**Release Date:** November 2026  
**Status:** Verified Institutional Quantitative Release  
**Test Suite:** 104 / 104 Passing Tests (`pytest`)

---

## Key Highlights & Architectural Upgrades

### 1. Complete Quantitative & Mathematical Audit (26 Audited Metrics)
- Added `tests/test_final_quantitative_validation.py` containing independent reference tests against analytical ground truths for all 26 financial, volatility, technical, statistical, and risk metrics.
- Standalone `compute_cagr()` in `analytics/returns.py`: Supports both trading-day ($252/N$) and calendar-day ($365.25/D$) conventions, total capital wipeout handling ($-1.0$), invalid base safeguards ($0.0$), and short duration protections ($N < 15$).
- Implemented **Yang-Zhang Volatility Estimator** (`compute_yang_zhang_volatility()`) in `analytics/volatility.py`: Combines overnight jump variance with continuous open-to-close variance, overcoming the zero-jump limitations of Parkinson and Garman-Klass estimators.
- Refined Sortino Ratio zero downside volatility handling (`float("inf")` flagged with contextual notes).
- Formalized finite-sample interpretation caveats for Jarque-Bera normality tests.

### 2. Institutional Futures Accounting & Regulatory Revisions
- Implemented `HISTORICAL_LOT_SIZE_SCHEDULE` and `get_historical_lot_size(symbol, date)` in `analytics/futures_accounting.py`.
- Encoded SEBI Index Derivatives Framework revisions effective **November 20, 2024**:
  - `NIFTY_FUT`: 75 (current) | 25 (Apr–Nov 2024) | 50 (2021–2024) | 75 (pre-2021)
  - `BANKNIFTY_FUT`: 30 (current) | 15 (Jul 2023–Nov 2024) | 25 (pre-2023)
- Fully dynamic, date-aware lot size tracking prevents historical margin distortion and miscalculated notional exposures.

### 3. Portfolio Optimization & Spectral Health Regularization
- Added `CovarianceAuditResult`, `audit_covariance_matrix()`, and `regularize_covariance_matrix()` in `analytics/portfolio.py`.
- Audits spectral condition number $\kappa(\boldsymbol{\Sigma}) = \lambda_{\max} / \lambda_{\min}$ to diagnose collinearity and singular covariance matrices.
- Automatic convex shrinkage regularization ($\boldsymbol{\Sigma}_{\text{reg}} = (1-\delta)\boldsymbol{\Sigma} + \delta \operatorname{diag}(\boldsymbol{\Sigma}) + \epsilon \mathbf{I}$) ensures strict positive definiteness and reliable SLSQP convergence.

### 4. Indian Market Catalog Taxonomy Expansion
- Refactored `config/instruments.py` to organize Indian financial assets into 12 distinct taxonomy categories matching official exchange classifications.
- Added family alias mappings in `get_instruments_by_family()` for full backward compatibility across legacy test suites and UI pickers.

### 5. Automated Testing & Verification Gate
- Test count expanded from 78 to **104 tests** across 12 test modules.
- Streamlit headless runtime validated with `streamlit.testing.v1.AppTest` with zero exceptions.
