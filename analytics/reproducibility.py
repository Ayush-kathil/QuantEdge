"""Experiment provenance and cryptographic reproducibility tracking for institutional research."""
import uuid
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from config.settings import APP_VERSION, PLATFORM_NAME


@dataclass
class ExperimentMetadata:
    """Standardized record of research configuration and cryptographic lineage."""
    experiment_id: str
    timestamp_utc: str
    platform_name: str
    platform_version: str
    symbol: str
    dataset_fingerprint_sha256: str
    data_source: str
    date_range_start: str
    date_range_end: str
    row_count: int
    strategy_name: str
    strategy_parameters: Dict[str, Any]
    initial_capital: float
    commission_bps: float
    slippage_bps: float
    execution_mode: str
    risk_free_rate: float
    trading_days: int
    validation_type: str
    metrics_summary: Dict[str, Any]

    def to_json(self, indent: int = 2) -> str:
        """Serializes experiment manifest into formatted JSON."""
        return json.dumps(asdict(self), indent=indent, default=str)


def generate_experiment_manifest(
    symbol: str,
    dataset_fingerprint: str,
    data_source: str,
    date_start: str,
    date_end: str,
    row_count: int,
    strategy_name: str,
    strategy_params: Dict[str, Any],
    initial_capital: float,
    commission_bps: float,
    slippage_bps: float,
    execution_mode: str,
    risk_free_rate: float,
    trading_days: int,
    validation_type: str,
    metrics_dict: Dict[str, Any]
) -> ExperimentMetadata:
    """
    Creates an immutable, audited experiment record.
    """
    exp_id = f"QEDGE-EXP-{uuid.uuid4().hex[:12].upper()}"
    ts_now = datetime.now(timezone.utc).isoformat()

    return ExperimentMetadata(
        experiment_id=exp_id,
        timestamp_utc=ts_now,
        platform_name=PLATFORM_NAME,
        platform_version=APP_VERSION,
        symbol=symbol,
        dataset_fingerprint_sha256=dataset_fingerprint,
        data_source=data_source,
        date_range_start=date_start,
        date_range_end=date_end,
        row_count=row_count,
        strategy_name=strategy_name,
        strategy_parameters=strategy_params,
        initial_capital=initial_capital,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        execution_mode=execution_mode,
        risk_free_rate=risk_free_rate,
        trading_days=trading_days,
        validation_type=validation_type,
        metrics_summary=metrics_dict
    )
