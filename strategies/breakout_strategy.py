"""Donchian Channel volatility and momentum breakout strategy."""
from typing import Dict, Any
import pandas as pd
import numpy as np
from strategies.base_strategy import BaseStrategy


class DonchianBreakout(BaseStrategy):
    """
    Donchian Channel Breakout Strategy (Turtle Trading Variant).
    
    Generates long breakout signals when price breaks above the N-period rolling high,
    and exits / shorts when price breaks below the rolling low or channel midpoint.
    
    Zero Look-Ahead Guarantee:
    Rolling high and low channels are shifted by 1 period so that the breakout threshold
    is strictly computed using information available prior to the current bar's close.
    """

    def __init__(
        self,
        lookback_window: int = 20,
        exit_lookback_window: int = 10,
        use_midpoint_exit: bool = False,
        long_only: bool = True
    ):
        if lookback_window <= 0 or exit_lookback_window <= 0:
            raise ValueError("Lookback windows must be strictly positive.")

        parameters = {
            "lookback_window": lookback_window,
            "exit_lookback_window": exit_lookback_window,
            "use_midpoint_exit": use_midpoint_exit,
            "long_only": long_only
        }
        super().__init__(
            name="Donchian Breakout",
            description=f"Donchian breakout system ({lookback_window}-day entry / {exit_lookback_window}-day exit)",
            parameters=parameters,
            long_only=long_only
        )
        self.lookback_window = lookback_window
        self.exit_lookback_window = exit_lookback_window
        self.use_midpoint_exit = use_midpoint_exit

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df, ["High", "Low", "Close"])
        res = df.copy()

        # Strict lag by 1 to prevent look-ahead bias
        upper_channel = res["High"].shift(1).rolling(window=self.lookback_window).max()
        lower_channel = res["Low"].shift(1).rolling(window=self.lookback_window).min()
        exit_upper = res["High"].shift(1).rolling(window=self.exit_lookback_window).max()
        exit_lower = res["Low"].shift(1).rolling(window=self.exit_lookback_window).min()
        midpoint = (upper_channel + lower_channel) / 2.0

        res["breakout_upper"] = upper_channel
        res["breakout_lower"] = lower_channel
        res["exit_upper"] = exit_upper
        res["exit_lower"] = exit_lower
        res["channel_midpoint"] = midpoint

        signals = np.zeros(len(res), dtype=int)
        current_pos = 0
        warmup = max(self.lookback_window, self.exit_lookback_window) + 1

        for i in range(len(res)):
            if i < warmup:
                signals[i] = 0
                continue

            c = res["Close"].iloc[i]
            u = upper_channel.iloc[i]
            l = lower_channel.iloc[i]
            eu = exit_upper.iloc[i]
            el = exit_lower.iloc[i]
            mid = midpoint.iloc[i]

            if np.isnan(u) or np.isnan(l):
                signals[i] = 0
                continue

            if current_pos == 0:
                if c > u:
                    current_pos = 1
                elif not self.long_only and c < l:
                    current_pos = -1
            elif current_pos == 1:
                # Long exit criteria
                exit_level = mid if self.use_midpoint_exit else el
                if c < exit_level:
                    current_pos = -1 if (not self.long_only and c < l) else 0
            elif current_pos == -1:
                # Short exit criteria
                exit_level = mid if self.use_midpoint_exit else eu
                if c > exit_level:
                    current_pos = 1 if c > u else 0

            signals[i] = current_pos

        res["signal"] = signals
        return res
