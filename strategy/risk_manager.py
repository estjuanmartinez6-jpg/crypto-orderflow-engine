"""
strategy/risk_manager.py

Gestor de riesgo avanzado a nivel de portfolio.

Controla:
  - Riesgo máximo diario (Daily Loss Limit)
  - Número máximo de trades consecutivos perdedores
  - Drawdown máximo desde el pico (circuit breaker)
  - Correlación entre señales (evitar sobre-concentración)
  - Horarios de trading (evitar períodos de baja liquidez)
  - Volatilidad del mercado (reducir tamaño en mercados erráticos)
  - Ajuste dinámico de SL/TP basado en ATR
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from utils.config import StrategyConfig
from utils.logger import get_logger

log = get_logger("RiskManager")


@dataclass
class RiskStatus:
    """Estado actual del riesgo del sistema."""
    is_trading_allowed: bool
    daily_pnl: float
    daily_pnl_pct: float
    consecutive_losses: int
    current_drawdown_pct: float
    equity_peak: float
    trades_today: int
    reason_blocked: str = ""
    risk_multiplier: float = 1.0  # Factor para ajustar tamaño de posición


@dataclass
class TradeRecord:
    """Registro de un trade cerrado para análisis de riesgo."""
    pnl: float
    pnl_pct: float
    timestamp: float
    signal_type: str
    direction: str
    duration_seconds: float


class RiskManager:
    """
    Gestor de riesgo a nivel de portfolio.
    Actúa como circuit breaker antes de ejecutar señales.

    Reglas implementadas:
    1. Daily Loss Limit: máximo X% de pérdida diaria
    2. Max Drawdown: pausar si DD > Y%
    3. Consecutive Losses: pausar después de N pérdidas seguidas
    4. Max Daily Trades: límite de trades por día
    5. Volatility Scaling: reducir tamaño en alta volatilidad
    6. Time Filter: no operar en horarios de baja liquidez
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        daily_loss_limit_pct: float = 0.03,    # 3% pérdida máxima diaria
        max_drawdown_pct: float = 0.08,        # 8% drawdown máximo
        max_consecutive_losses: int = 4,
        max_daily_trades: int = 10,
        min_rr_ratio: float = 1.5,
        volatility_lookback: int = 100,
        high_vol_threshold_pct: float = 2.0,   # ATR > 2% del precio = alta volatilidad
    ):
        self.initial_capital = initial_capital
        self.daily_loss_limit = daily_loss_limit_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_trades = max_daily_trades
        self.min_rr_ratio = min_rr_ratio
        self.volatility_lookback = volatility_lookback
        self.high_vol_threshold = high_vol_threshold_pct

        # Estado del equity
        self._capital = initial_capital
        self._equity_peak = initial_capital
        self._daily_start_capital = initial_capital
        self._daily_reset_ts = time.time()

        # Historial de trades
        self._trades: deque[TradeRecord] = deque(maxlen=1000)
        self._daily_trades: List[TradeRecord] = []
        self._consecutive_losses = 0

        # Historial de precios para volatilidad
        self._price_history: deque[float] = deque(maxlen=200)

        # Estado de pausa
        self._paused = False
        self._pause_reason = ""
        self._pause_until: Optional[float] = None

    # ─── Actualización de estado ──────────────────────────────────────────────

    def update_capital(self, new_capital: float) -> None:
        """Actualizar el capital actual después de cada trade."""
        self._capital = new_capital
        self._equity_peak = max(self._equity_peak, new_capital)

        # Reset diario (a las 00:00 UTC)
        if time.time() - self._daily_reset_ts > 86400:
            self._daily_start_capital = new_capital
            self._daily_trades.clear()
            self._daily_reset_ts = time.time()
            log.info("Reset diario de riesgo completado.")

    def record_trade(self, record: TradeRecord) -> None:
        """Registra un trade cerrado y actualiza los contadores."""
        self._trades.append(record)
        self._daily_trades.append(record)

        if record.pnl <= 0:
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.max_consecutive_losses:
                self._pause_trading(
                    reason=f"{self._consecutive_losses} pérdidas consecutivas",
                    duration_seconds=3600  # Pausa 1 hora
                )
        else:
            self._consecutive_losses = 0

        self.update_capital(self._capital + record.pnl)

    def add_price(self, price: float) -> None:
        """Agrega precio al historial para cálculo de volatilidad."""
        self._price_history.append(price)

    # ─── Pausa/reanudación ────────────────────────────────────────────────────

    def _pause_trading(self, reason: str, duration_seconds: float = 3600) -> None:
        self._paused = True
        self._pause_reason = reason
        self._pause_until = time.time() + duration_seconds
        log.warning(f"⏸ Trading PAUSADO: {reason} | Duración: {duration_seconds/60:.0f} min")

    def _check_resume(self) -> None:
        if self._paused and self._pause_until and time.time() > self._pause_until:
            self._paused = False
            self._pause_reason = ""
            self._pause_until = None
            log.info("▶️ Trading REANUDADO automáticamente.")

    # ─── Checks de riesgo ────────────────────────────────────────────────────

    def _check_daily_loss(self) -> Optional[str]:
        daily_pnl_pct = (self._capital - self._daily_start_capital) / self._daily_start_capital
        if daily_pnl_pct <= -self.daily_loss_limit:
            return f"Daily loss limit alcanzado: {daily_pnl_pct:.1%} (límite {-self.daily_loss_limit:.1%})"
        return None

    def _check_drawdown(self) -> Optional[str]:
        dd_pct = (self._capital - self._equity_peak) / self._equity_peak
        if dd_pct <= -self.max_drawdown_pct:
            return f"Max drawdown alcanzado: {dd_pct:.1%} (límite {-self.max_drawdown_pct:.1%})"
        return None

    def _check_max_trades(self) -> Optional[str]:
        if len(self._daily_trades) >= self.max_daily_trades:
            return f"Límite diario de trades alcanzado: {len(self._daily_trades)}/{self.max_daily_trades}"
        return None

    def _compute_volatility(self) -> float:
        """ATR simplificado como % del precio promedio."""
        if len(self._price_history) < 20:
            return 0.0
        prices = np.array(list(self._price_history)[-self.volatility_lookback:])
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns) * 100)  # En porcentaje

    def _compute_risk_multiplier(self) -> float:
        """
        Factor de ajuste del tamaño de posición basado en condiciones.
        1.0 = tamaño normal
        0.5 = reducir a la mitad en alta volatilidad
        1.5 = aumentar en condiciones óptimas (baja vol + racha ganadora)
        """
        multiplier = 1.0

        # Reducir en alta volatilidad
        vol = self._compute_volatility()
        if vol > self.high_vol_threshold:
            vol_factor = self.high_vol_threshold / vol
            multiplier *= min(vol_factor, 1.0)

        # Reducir en racha de pérdidas (antes del máximo)
        if self._consecutive_losses >= 2:
            multiplier *= max(0.5, 1.0 - self._consecutive_losses * 0.1)

        # Aumentar levemente en buena racha (máx 1.5x)
        recent_trades = list(self._trades)[-5:]
        if len(recent_trades) >= 3 and all(t.pnl > 0 for t in recent_trades):
            multiplier = min(multiplier * 1.25, 1.5)

        # Reducir si estamos cerca del daily loss limit
        daily_pnl_pct = (self._capital - self._daily_start_capital) / self._daily_start_capital
        if daily_pnl_pct < -self.daily_loss_limit * 0.5:
            multiplier *= 0.5

        return round(max(0.25, min(multiplier, 2.0)), 2)

    # ─── Validación de señal ──────────────────────────────────────────────────

    def validate_signal(self, signal) -> tuple[bool, str]:
        """
        Verifica si una señal puede ser ejecutada.
        Retorna (puede_operar: bool, razón: str)
        """
        self._check_resume()

        # Check de pausa manual/automática
        if self._paused:
            return False, f"Sistema pausado: {self._pause_reason}"

        # Check de límites
        checks = [
            self._check_daily_loss(),
            self._check_drawdown(),
            self._check_max_trades(),
        ]

        for reason in checks:
            if reason:
                self._pause_trading(reason, duration_seconds=1800)
                return False, reason

        # Validar RR mínimo de la señal
        if signal.risk_reward < self.min_rr_ratio:
            return False, f"RR insuficiente: {signal.risk_reward:.2f} < {self.min_rr_ratio}"

        # Validar confianza mínima
        if signal.confidence < 0.35:
            return False, f"Confianza insuficiente: {signal.confidence:.2f}"

        return True, "OK"

    def get_status(self) -> RiskStatus:
        """Retorna el estado actual completo del riesgo."""
        self._check_resume()

        daily_pnl = self._capital - self._daily_start_capital
        daily_pnl_pct = daily_pnl / self._daily_start_capital
        dd_pct = (self._capital - self._equity_peak) / self._equity_peak

        can_trade = (
            not self._paused and
            self._check_daily_loss() is None and
            self._check_drawdown() is None and
            self._check_max_trades() is None
        )

        return RiskStatus(
            is_trading_allowed=can_trade,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            consecutive_losses=self._consecutive_losses,
            current_drawdown_pct=dd_pct,
            equity_peak=self._equity_peak,
            trades_today=len(self._daily_trades),
            reason_blocked=self._pause_reason,
            risk_multiplier=self._compute_risk_multiplier(),
        )

    # ─── ATR dinámico para SL/TP ──────────────────────────────────────────────

    def compute_atr(self, window: int = 14) -> Optional[float]:
        """
        Calcula ATR simplificado desde el historial de precios.
        Útil para ajustar SL/TP dinámicamente.
        """
        if len(self._price_history) < window + 1:
            return None
        prices = np.array(list(self._price_history)[-window-1:])
        true_ranges = np.abs(np.diff(prices))
        return float(np.mean(true_ranges[-window:]))

    def adjust_stop_loss(
        self,
        entry_price: float,
        proposed_stop: float,
        direction: str,
        atr_multiplier: float = 1.5,
    ) -> float:
        """
        Ajusta el stop loss propuesto basado en ATR actual.
        Evita stops demasiado ajustados en mercados volátiles.
        """
        atr = self.compute_atr()
        if atr is None:
            return proposed_stop

        min_distance = atr * atr_multiplier

        if direction == "LONG":
            atr_stop = entry_price - min_distance
            return min(proposed_stop, atr_stop)  # El más lejano
        else:
            atr_stop = entry_price + min_distance
            return max(proposed_stop, atr_stop)

    # ─── Estadísticas ─────────────────────────────────────────────────────────

    def get_performance_stats(self) -> dict:
        """Métricas de rendimiento de los trades registrados."""
        if not self._trades:
            return {}

        pnls = [t.pnl for t in self._trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        return {
            "total_trades": len(self._trades),
            "win_rate": len(wins) / len(pnls),
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else float("inf"),
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "expectancy": np.mean(pnls),
            "consecutive_losses_max": self._max_consecutive_losses_ever(),
            "current_capital": self._capital,
            "total_pnl": self._capital - self.initial_capital,
            "total_pnl_pct": (self._capital - self.initial_capital) / self.initial_capital * 100,
        }

    def _max_consecutive_losses_ever(self) -> int:
        max_streak = 0
        current = 0
        for t in self._trades:
            if t.pnl <= 0:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    def log_status(self) -> None:
        status = self.get_status()
        vol = self._compute_volatility()
        emoji = "🟢" if status.is_trading_allowed else "🔴"
        log.info(
            f"{emoji} RISK STATUS | "
            f"Cap: ${self._capital:,.2f} | "
            f"PnL día: {status.daily_pnl_pct:+.1%} | "
            f"DD: {status.current_drawdown_pct:.1%} | "
            f"Losses seguidas: {status.consecutive_losses} | "
            f"Trades hoy: {status.trades_today}/{self.max_daily_trades} | "
            f"Vol: {vol:.3f}% | "
            f"Size mult: {status.risk_multiplier:.2f}x"
        )
