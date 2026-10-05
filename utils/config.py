"""
utils/config.py — Configuración central del sistema.

Cambios respecto a la versión anterior:

  1. Duplicados eliminados — cada concepto tiene UNA variable base.
     Los alias (risk_per_trade, commission_rate, slippage_bps) son
     @property de solo lectura para compatibilidad con código existente.

  2. Streams dinámicos — se generan desde self.symbol, sin hardcodear
     "ethusdt". Cambiar el símbolo actualiza los streams automáticamente.

  3. Validaciones en __post_init__ — rangos razonables para parámetros
     críticos. Errores de configuración fallan en el arranque, no en
     medio de un trade.

  4. Modo único — live_trading + paper_trading reemplazados por
     mode: str = "paper"  ("paper" | "live" | "backtest").
     Elimina la posibilidad de tener live=True y paper=True simultáneamente.

  5. Protección de live trading — SystemConfig valida que haya API keys
     si mode == "live".

  6. Parámetros ajustados para ETH Futures:
     absorption_volume_threshold: 50 → 100 ETH
     imbalance_threshold: 0.25 → 0.35
     tick_size: 1.0 → 0.10 (ETH cotiza con decimales relevantes)

  7. StrategyConfig expone dynamic_sl y dynamic_tp para futura
     optimización sin cambiar la interfaz.

  8. Logging con rotación configurable.

  9. min_risk_reward añadido a StrategyConfig (requerido por signals.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Exchange
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ExchangeConfig:
    symbol:                  str   = "ethusdt"
    base_url:                str   = "wss://fstream.binance.com/stream"
    rest_url:                str   = "https://fapi.binance.com"
    reconnect_delay:         float = 3.0
    max_reconnect_attempts:  int   = 20
    api_key:                 str   = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret:              str   = field(default_factory=lambda: os.getenv("BINANCE_SECRET", ""))

    @property
    def streams(self) -> List[str]:
        """
        Streams construidos dinámicamente desde self.symbol.
        Cambiar el símbolo actualiza los streams automáticamente.
        """
        s = self.symbol.lower()
        return [
            f"{s}@aggTrade",
            f"{s}@depth20@100ms",
        ]

    def __post_init__(self):
        if not self.symbol:
            raise ValueError("ExchangeConfig.symbol no puede estar vacío.")
        self.symbol = self.symbol.lower()


# ──────────────────────────────────────────────────────────────────────────────
# Buffer
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BufferConfig:
    max_trades:      int = 50_000
    max_book_levels: int = 20


# ──────────────────────────────────────────────────────────────────────────────
# Order Flow
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class OrderFlowConfig:
    """
    Parámetros de microestructura ajustados para ETH Futures.

    absorption_volume_threshold: 100 ETH (antes 50).
      ETH tiene trades grandes frecuentes; 50 ETH generaba demasiadas
      señales de absorción falsas en períodos de alto volumen.

    imbalance_threshold: 0.35 (antes 0.25).
      Un imbalance de 0.25 es alcanzable por ruido normal del libro.
      0.35 requiere una asimetría bid/ask más pronunciada y real.
    """
    window_seconds:              int   = 300
    absorption_price_threshold:  float = 0.5
    absorption_volume_threshold: float = 100.0   # ETH (era 50.0)
    imbalance_threshold:         float = 0.35    # (era 0.25)
    cvd_divergence_window:       int   = 10

    def __post_init__(self):
        if self.absorption_volume_threshold <= 0:
            raise ValueError("absorption_volume_threshold debe ser > 0")
        if not (0 < self.imbalance_threshold < 1):
            raise ValueError("imbalance_threshold debe estar entre 0 y 1")


# ──────────────────────────────────────────────────────────────────────────────
# Volume Profile
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VolumeProfileConfig:
    """
    tick_size: 0.10 USD (antes 1.0).
      Con tick_size = $1 en ETH, bins de un dólar dan muy pocos niveles
      en rangos estrechos y pierden resolución. $0.10 es más apropiado
      para la volatilidad típica de ETH ($5-$30 ATR diario).
    """
    tick_size:       float = 0.10    # USD por bin (era 1.0)
    value_area_pct:  float = 0.70
    hvn_percentile:  float = 0.90    # alineado con el motor (era 0.80)
    lvn_percentile:  float = 0.10    # alineado con el motor (era 0.20)
    session_hours:   int   = 24

    def __post_init__(self):
        if self.tick_size <= 0:
            raise ValueError("tick_size debe ser > 0")
        if not (0 < self.value_area_pct < 1):
            raise ValueError("value_area_pct debe estar entre 0 y 1")
        if not (0 < self.hvn_percentile < 1):
            raise ValueError("hvn_percentile debe estar entre 0 y 1")
        if not (0 < self.lvn_percentile < self.hvn_percentile):
            raise ValueError("lvn_percentile debe ser menor que hvn_percentile")


# ──────────────────────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StrategyConfig:
    """
    risk_per_trade_pct es la variable base.
    risk_per_trade es un @property alias para compatibilidad con código
    existente que use ese nombre.
    """
    risk_per_trade_pct:        float = 0.01    # 1% del capital por trade
    max_open_trades:           int   = 1       # una posición a la vez
    min_confirmations:         int   = 3
    signal_expiry_seconds:     int   = 60
    stop_atr_multiplier:       float = 1.5
    take_profit_ratio:         float = 2.0
    min_risk_reward:           float = 1.5     # requerido por signals.py

    # Flags para optimización futura (no afectan la lógica actual)
    dynamic_sl:  bool = True
    dynamic_tp:  bool = True

    def __post_init__(self):
        if not (0 < self.risk_per_trade_pct <= 0.05):
            raise ValueError(
                f"risk_per_trade_pct debe estar entre 0 y 5% "
                f"(recibido: {self.risk_per_trade_pct:.2%})"
            )
        if self.max_open_trades < 1:
            raise ValueError("max_open_trades debe ser al menos 1")
        if self.min_risk_reward < 1.0:
            raise ValueError("min_risk_reward debe ser al menos 1.0")
        if self.signal_expiry_seconds < 5:
            raise ValueError("signal_expiry_seconds debe ser al menos 5s")

    # ── Alias de compatibilidad ───────────────────────────────────────────

    @property
    def risk_per_trade(self) -> float:
        """Alias de risk_per_trade_pct para compatibilidad con código existente."""
        return self.risk_per_trade_pct

    @property
    def signal_confirmation_ticks(self) -> int:
        """Alias de min_confirmations."""
        return self.min_confirmations


# ──────────────────────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestConfig:
    """
    commission_pct y slippage_pct son las variables base.
    commission_rate y slippage_bps son @property alias para compatibilidad.
    """
    initial_capital: float = 10_000.0
    commission_pct:  float = 0.0004   # 0.04% taker fee Binance Futures
    slippage_pct:    float = 0.0003   # 0.03% (actualizado para ser consistente con trader.py)
    leverage:        int   = 3
    symbol:          str   = "ETHUSDT"

    def __post_init__(self):
        if self.initial_capital <= 0:
            raise ValueError("initial_capital debe ser > 0")
        if not (0 <= self.commission_pct <= 0.01):
            raise ValueError("commission_pct fuera de rango razonable (0–1%)")
        if self.leverage < 1 or self.leverage > 125:
            raise ValueError("leverage debe estar entre 1 y 125")

    # ── Alias de compatibilidad ───────────────────────────────────────────

    @property
    def commission_rate(self) -> float:
        """Alias de commission_pct."""
        return self.commission_pct

    @property
    def slippage_bps(self) -> float:
        """Slippage expresado en basis points (1 bps = 0.01%)."""
        return self.slippage_pct * 10_000


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard / Visualization
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DashboardConfig:
    host:               str = "127.0.0.1"
    port:               int = 8050
    update_interval_ms: int = 500


@dataclass
class VisualizationConfig:
    theme:          str  = "plotly_dark"
    chart_height:   int  = 900
    profile_bins:   int  = 100
    show_hvn_lvn:   bool = True
    show_cvd:       bool = True
    show_delta:     bool = True
    show_imbalance: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Sistema completo
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SystemConfig:
    """
    Configuración global del sistema.

    mode: "paper" | "live" | "backtest"
      Reemplaza el par live_trading / paper_trading que podía quedar en
      estado inconsistente (ambos True, ambos False, etc.).

      "paper"    → paper trading en tiempo real, sin riesgo
      "live"     → trading real con Binance API (requiere API keys)
      "backtest" → correr engine.py sobre datos históricos

    Validaciones en __post_init__:
      - mode debe ser uno de los tres valores válidos
      - mode == "live" requiere API keys presentes
    """
    exchange:      ExchangeConfig      = field(default_factory=ExchangeConfig)
    buffer:        BufferConfig        = field(default_factory=BufferConfig)
    order_flow:    OrderFlowConfig     = field(default_factory=OrderFlowConfig)
    volume_profile: VolumeProfileConfig = field(default_factory=VolumeProfileConfig)
    strategy:      StrategyConfig      = field(default_factory=StrategyConfig)
    backtest:      BacktestConfig      = field(default_factory=BacktestConfig)
    dashboard:     DashboardConfig     = field(default_factory=DashboardConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    mode:      str = "paper"           # "paper" | "live" | "backtest"
    log_level: str = "INFO"
    log_file:  str = "logs/system.log"
    log_rotation: bool = True          # rotación automática de logs

    VALID_MODES = ("paper", "live", "backtest")

    def __post_init__(self):
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"mode='{self.mode}' no válido. "
                f"Opciones: {self.VALID_MODES}"
            )
        if self.mode == "live":
            if not self.exchange.api_key or not self.exchange.api_secret:
                raise ValueError(
                    "mode='live' requiere BINANCE_API_KEY y BINANCE_SECRET "
                    "configurados como variables de entorno."
                )
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"log_level='{self.log_level}' no es un nivel válido de logging.")

    # ── Propiedades de conveniencia ───────────────────────────────────────

    @property
    def is_paper(self) -> bool:
        return self.mode == "paper"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    @property
    def is_backtest(self) -> bool:
        return self.mode == "backtest"

    # ── Alias de compatibilidad con código que use live_trading/paper_trading

    @property
    def live_trading(self) -> bool:
        return self.mode == "live"

    @property
    def paper_trading(self) -> bool:
        return self.mode == "paper"


# ──────────────────────────────────────────────────────────────────────────────
# Instancia global
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = SystemConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Test rápido (python -m utils.config)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(CONFIG)
    print()
    print("Streams:", CONFIG.exchange.streams)
    print("Mode:", CONFIG.mode)
    print("Is paper:", CONFIG.is_paper)
    print("Risk per trade:", CONFIG.strategy.risk_per_trade)
    print("Commission rate:", CONFIG.backtest.commission_rate)
    print("Slippage bps:", CONFIG.backtest.slippage_bps)
    print("Tick size:", CONFIG.volume_profile.tick_size)
    print("Imbalance threshold:", CONFIG.order_flow.imbalance_threshold)
    print("Absorption vol threshold:", CONFIG.order_flow.absorption_volume_threshold)
    print()
    print("✅ Config cargada sin errores.")
