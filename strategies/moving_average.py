"""Moving Average Crossover trend-following quantitative strategy."""
from typing import Dict, Any, Literal
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from analytics.indicators import compute_sma, compute_ema


class MovingAverageCrossover(BaseStrategy):
    """
    Dual Moving Average Crossover Strategy.
    
    Generates bullish trend signals when the short-term moving average crosses above
    the long-term moving average, and bearish/exit signals on downward crossover.
    """

    def __init__(
        self,
        short_period: int = 20,
        long_period: int = 50,
        ma_type: Literal["SMA", "EMA"] = "SMA",
        long_only: bool = True
    ):
        if short_period >= long_period:
            raise ValueError(f"short_period ({short_period}) must be less than long_period ({long_period})")

        parameters = {
            "short_period": short_period,
            "long_period": long_period,
            "ma_type": ma_type,
            "long_only": long_only
        }
        super().__init__(
            name="Moving Average Crossover",
            description=f"Dual {ma_type} crossover ({short_period}/{long_period}) trend-following system",
            parameters=parameters,
            long_only=long_only
        )
        self.short_period = short_period
        self.long_period = long_period
        self.ma_type = ma_type

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df, ["Close"])
        res = df.copy()

        if self.ma_type == "SMA":
            ma_s = compute_sma(res["Close"], period=self.short_period)
            ma_l = compute_sma(res["Close"], period=self.long_period)
        elif self.ma_type == "EMA":
            ma_s = compute_ema(res["Close"], period=self.short_period)
            ma_l = compute_ema(res["Close"], period=self.long_period)
        else:
            raise ValueError(f"Unsupported ma_type '{self.ma_type}'")

        res["ma_short"] = ma_s
        res["ma_long"] = ma_l

        # Raw signal: +1 when short > long, -1 or 0 when short < long
        raw_signal = np.where(res["ma_short"] > res["ma_long"], 1, 0 if self.long_only else -1)
        res["signal"] = raw_signal

        # Zero out warm-up period strictly
        res.iloc[:self.long_period, res.columns.get_loc("signal")] = 0

        # Crossover events for visualization and audit
        res["crossover"] = res["signal"].diff().fillna(0)

        return res
