"""Abstract base class and contract for quantitative trading strategies."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base strategy defining signal generation and parameter interfaces.
    
    Signal Convention:
        +1 : Long position
        -1 : Short position (or 0 if long_only)
         0 : Flat / Cash
         
    Guarantees:
        Signals generated at timestamp t must utilize information up to t strictly (P_s where s <= t).
        No future peeking / look-ahead bias is permitted.
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any], long_only: bool = True):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.long_only = long_only

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates timestamped signals from historical market OHLCV DataFrame.
        
        Parameters:
            df: Historical price DataFrame (must have standard OHLCV columns)
            
        Returns:
            DataFrame indexed by timestamp containing at minimum:
            - 'signal': Integer {-1, 0, 1}
            - Any intermediate indicators (e.g. 'sma_short', 'rsi', etc.)
        """
        pass

    def validate_input(self, df: pd.DataFrame, required_cols: List[str] = None) -> None:
        """Verifies input data meets minimum prerequisites."""
        if required_cols is None:
            required_cols = ["Close"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Strategy '{self.name}' requires columns: {missing}. Found: {list(df.columns)}")
        if df.empty:
            raise ValueError(f"Strategy '{self.name}' cannot generate signals on empty DataFrame.")
