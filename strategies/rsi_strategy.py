"""RSI Mean Reversion quantitative trading strategy."""
from typing import Dict, Any
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy
from analytics.indicators import compute_rsi


class RSIMeanReversion(BaseStrategy):
    """
    RSI Mean Reversion Strategy.
    
    Exploits price overextensions by buying when RSI falls below the oversold threshold
    and exiting (or shorting) when RSI crosses above the exit threshold or overbought band.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        exit_at_neutral: bool = True,
        neutral_threshold: float = 50.0,
        long_only: bool = True
    ):
        if oversold >= overbought:
            raise ValueError(f"oversold ({oversold}) must be strictly less than overbought ({overbought})")

        parameters = {
            "rsi_period": rsi_period,
            "oversold": oversold,
            "overbought": overbought,
            "exit_at_neutral": exit_at_neutral,
            "neutral_threshold": neutral_threshold,
            "long_only": long_only
        }
        super().__init__(
            name="RSI Mean Reversion",
            description=f"RSI({rsi_period}) counter-trend system (Oversold: {oversold}, Overbought: {overbought})",
            parameters=parameters,
            long_only=long_only
        )
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.exit_at_neutral = exit_at_neutral
        self.neutral_threshold = neutral_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df, ["Close"])
        res = df.copy()

        rsi = compute_rsi(res["Close"], period=self.rsi_period)
        res["rsi"] = rsi

        signals = np.zeros(len(res), dtype=int)
        current_pos = 0

        # Sequential state machine to maintain realistic position holding without look-ahead
        for i in range(len(res)):
            if i <= self.rsi_period or np.isnan(rsi.iloc[i]):
                signals[i] = 0
                continue

            r_val = rsi.iloc[i]

            if current_pos == 0:
                # Flat: look for entries
                if r_val < self.oversold:
                    current_pos = 1
                elif not self.long_only and r_val > self.overbought:
                    current_pos = -1
            elif current_pos == 1:
                # Long: look for exit or short flip
                if self.exit_at_neutral and r_val >= self.neutral_threshold:
                    current_pos = 0
                elif r_val >= self.overbought:
                    current_pos = -1 if not self.long_only else 0
            elif current_pos == -1:
                # Short: look for exit or long flip
                if self.exit_at_neutral and r_val <= self.neutral_threshold:
                    current_pos = 0
                elif r_val <= self.oversold:
                    current_pos = 1

            signals[i] = current_pos

        res["signal"] = signals
        return res
