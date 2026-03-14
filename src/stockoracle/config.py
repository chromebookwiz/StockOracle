from __future__ import annotations

from dataclasses import dataclass, field

from .universe import DEFAULT_UNIVERSE


@dataclass(slots=True)
class AppConfig:
    universe: list[str] = field(default_factory=lambda: DEFAULT_UNIVERSE.copy())
    benchmark: str = "SPY"
    start_date: str = "2021-01-01"
    holdout_days: int = 45
    top_k: int = 10
    min_history_days: int = 180
    intraday_period_days: int = 45
    intraday_interval: str = "15m"
    prediction_mode: str = "same_day"
    signal_bar_index: int | None = None
    enable_live_news: bool = True
    enable_live_options: bool = True
    enable_earnings_features: bool = True
    starting_capital: float = 25_000.0
    execution_mode: str = "paper"
    max_notional_per_trade: float = 5_000.0
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 5.0
    max_position_weight: float = 0.20
    random_state: int = 42

    def normalized_universe(self) -> list[str]:
        ordered = [symbol.strip().upper() for symbol in self.universe if symbol.strip()]
        deduped = list(dict.fromkeys(ordered))
        if self.benchmark not in deduped:
            deduped.append(self.benchmark)
        return deduped

    def tradable_universe(self) -> list[str]:
        return [symbol for symbol in self.normalized_universe() if symbol != self.benchmark]