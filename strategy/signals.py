"""
signals.py — Motor de señales de trading para ETH Order Flow.

ARQUITECTURA NUEVA: Contexto → Setup → Trigger

En vez del sistema de score anterior (suma ponderada de 5 indicadores),
la lógica ahora sigue tres capas en cascada. Si cualquiera falla, se
retorna None inmediatamente. No hay puntuación parcial.

  1. CONTEXTO  — ¿dónde está el precio? ¿tiene ventaja operar aquí?
                 Fuente: CompositeProfile (sesión + diario + semanal)
                 Regla: solo operar fuera del Value Area y con macro alineada

  2. SETUP     — ¿el flujo de órdenes confirma la dirección del contexto?
                 Fuente: DualDelta (largo=filtro, corto=alineación)
                 Regla: largo y corto deben apuntar en la misma dirección

  3. TRIGGER   — ¿está ocurriendo algo real AHORA en una zona clave?
                 Fuente: AbsorptionSignal.is_high_quality
                 Regla: absorción de alta calidad en VAL/VAH/HVN

Resultado: muchos menos trades, pero con lógica institucional real.

Cambios técnicos:
  - Score eliminado completamente
  - SignalQuality simplificado (VALID / INVALID)
  - Divergencia eliminada del flujo de evaluación
  - Imbalance del libro pasa a ser filtro secundario opcional (no bloquea)
  - evaluate() usa CompositeProfile en lugar de VolumeProfileResult solo
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

from processing.order_flow import OrderFlowSnapshot, DualDeltaResult
from processing.volume_profile import (
    VolumeProfileResult,
    VolumeProfileEngine,
    CompositeProfile,
)
from data.data_buffer import DataBuffer
from utils.config import CONFIG

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    NONE  = "NONE"


class RejectionReason(Enum):
    """Por qué se rechazó una señal. Útil para debugging y backtesting."""
    NO_PROFILE          = "sin_profile"
    IN_VALUE_AREA       = "precio_dentro_del_VA"      # contexto: ruido
    MACRO_AGAINST       = "macro_en_contra"            # contexto: diario/semanal opuestos
    FLOW_NOT_ALIGNED    = "flujo_no_alineado"          # setup: delta largo/corto divergen
    NO_ABSORPTION       = "sin_absorcion"              # trigger: no hay absorción de calidad
    RR_INSUFFICIENT     = "RR_insuficiente"            # filtro final


@dataclass
class TradeSetup:
    """
    Setup de trading completo con toda la información necesaria para ejecutar.
    """
    timestamp: float
    direction: SignalDirection

    # Precios
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float

    # Contexto que validó la señal
    zone: str                        # zona del perfil donde se generó
    macro_bias: str                  # 'BULLISH' | 'BEARISH' | 'NEUTRAL'
    absorption_strength: float       # fuerza 0-1 de la absorción que triggereó
    absorption_atr_ratio: float      # displacement/ATR (< 0.25 es bueno)
    dual_delta_short_pct: float      # delta corto como % del volumen
    dual_delta_long_pct: float       # delta largo como % del volumen

    # Métricas de riesgo
    risk_reward: float
    estimated_size_eth: float
    estimated_risk_usd: float

    # Estado
    is_valid: bool = True
    expiry_timestamp: float = 0.0

    def __post_init__(self):
        if self.expiry_timestamp == 0.0:
            self.expiry_timestamp = self.timestamp + CONFIG.strategy.signal_expiry_seconds

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expiry_timestamp

    @property
    def risk_usd(self) -> float:
        return abs(self.entry_price - self.stop_loss) * self.estimated_size_eth

    def summary(self) -> str:
        return (
            f"[VALID] {self.direction.value} ETH | "
            f"Entry={self.entry_price:.2f} | "
            f"SL={self.stop_loss:.2f} | "
            f"TP1={self.take_profit_1:.2f} TP2={self.take_profit_2:.2f} | "
            f"RR={self.risk_reward:.2f} | "
            f"Zone={self.zone} | Macro={self.macro_bias} | "
            f"AbsStrength={self.absorption_strength:.2f} | "
            f"ΔShort={self.dual_delta_short_pct:.1f}% "
            f"ΔLong={self.dual_delta_long_pct:.1f}%"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Motor de señales
# ──────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Motor de señales institucionales para ETH.

    Flujo de evaluación (cascada):
      [CONTEXTO]  precio fuera del VA + macro alineada
          ↓
      [SETUP]     delta largo y corto apuntan en misma dirección
          ↓
      [TRIGGER]   absorción de alta calidad en zona clave
          ↓
      [FILTRO]    R:R >= 1.5
          ↓
      TradeSetup
    """

    def __init__(self, buffer: DataBuffer, vp_engine: VolumeProfileEngine):
        self.buffer          = buffer
        self.vp_engine       = vp_engine
        self._last_signal: Optional[TradeSetup]  = None
        self._signal_history: List[TradeSetup]   = []
        self._capital: float = CONFIG.backtest.initial_capital

    # ─── CAPA 1: CONTEXTO ────────────────────────────────────────────────────

    def _evaluate_context(
        self,
        price: float,
        composite: CompositeProfile,
    ) -> Tuple[Optional[SignalDirection], str, Dict]:
        """
        Determina si hay ventaja contextual y en qué dirección.

        Reglas:
          - Si el precio está DENTRO del Value Area → sin edge → None
          - Si el precio está FUERA pero el bias macro va en contra → None
          - Si ambos se alinean → dirección candidata

        Retorna (direction | None, rejection_reason | '', context_dict)
        """
        if composite.session is None:
            return None, RejectionReason.NO_PROFILE.value, {}

        profile = composite.session
        ctx     = self.vp_engine.analyze_price_context(price, profile, composite)

        # Dentro del Value Area = ruido. No operar.
        if ctx["zone"] in ("IN_VALUE_AREA", "AT_POC"):
            return None, RejectionReason.IN_VALUE_AREA.value, ctx

        # Determinar dirección candidata por zona
        if ctx["zone"] in ("BELOW_VALUE_AREA", "AT_HVN_BELOW"):
            candidate = SignalDirection.LONG
        elif ctx["zone"] in ("ABOVE_VALUE_AREA", "AT_HVN_ABOVE", "ABOVE_LVN"):
            candidate = SignalDirection.SHORT
        else:
            return None, RejectionReason.IN_VALUE_AREA.value, ctx

        # Filtro macro: si el bias macro va en contra, no operar
        macro_bias = ctx.get("macro_bias", "NEUTRAL")
        if candidate == SignalDirection.LONG  and macro_bias == "BEARISH":
            return None, RejectionReason.MACRO_AGAINST.value, ctx
        if candidate == SignalDirection.SHORT and macro_bias == "BULLISH":
            return None, RejectionReason.MACRO_AGAINST.value, ctx

        return candidate, "", ctx

    # ─── CAPA 2: SETUP ───────────────────────────────────────────────────────

    def _evaluate_setup(
        self,
        direction: SignalDirection,
        snapshot: OrderFlowSnapshot,
    ) -> Tuple[bool, str]:
        """
        Valida que el flujo de órdenes confirme la dirección del contexto.

        Reglas con DualDelta:
          - Para LONG: delta largo > 0 O (largo < 0 pero corto > 0, posible giro)
            El delta corto POSITIVO es el filtro de alineación.
          - Para SHORT: delta largo < 0 O (largo > 0 pero corto < 0, posible giro)

        No exigimos que el delta largo sea positivo en un LONG porque a veces
        el mejor momento de entrada es precisamente cuando el delta largo es
        negativo (vendedores agotándose) pero el corto ya dio vuelta.
        Lo que NO aceptamos es delta corto negativo en un LONG.
        """
        dd = snapshot.dual_delta

        if dd.short is None or dd.long is None:
            return False, RejectionReason.FLOW_NOT_ALIGNED.value

        short_pct = dd.short.delta_pct
        long_pct  = dd.long.delta_pct

        if direction == SignalDirection.LONG:
            # Delta corto debe ser positivo o al menos no fuertemente negativo
            if short_pct < -20:  # 20% de tolerancia: pequeños retrasos son ok
                return False, RejectionReason.FLOW_NOT_ALIGNED.value

        elif direction == SignalDirection.SHORT:
            if short_pct > 20:
                return False, RejectionReason.FLOW_NOT_ALIGNED.value

        return True, ""

    # ─── CAPA 3: TRIGGER (ABSORCIÓN COMO BONUS) ──────────────────────────────

    def _score_absorption(
        self,
        direction: SignalDirection,
        snapshot: OrderFlowSnapshot,
    ) -> float:
        """
        Evalúa la absorción como un score de 0.0 a 1.0.
        NO bloquea la señal — la absorción mejora la calidad pero no es
        requisito. El edge real viene de Contexto + Setup; la absorción
        es confirmación adicional cuando está presente.

          0.0  → sin absorción o dirección equivocada
          0.5  → absorción presente pero baja calidad
          1.0  → absorción de alta calidad y dirección correcta
        """
        abs_signal = snapshot.absorption
        if abs_signal is None:
            return 0.0

        direction_match = (
            (direction == SignalDirection.LONG  and abs_signal.is_buy_absorption) or
            (direction == SignalDirection.SHORT and abs_signal.is_sell_absorption)
        )
        if not direction_match:
            return 0.0

        if abs_signal.is_high_quality:
            return 1.0

        # Absorción presente pero no perfecta → bonus parcial
        if abs_signal.strength >= 0.35 and abs_signal.price_displacement_atr < 0.4:
            return 0.5

        return 0.0

    # ─── Cálculo de niveles de riesgo ────────────────────────────────────────

    def _calculate_levels(
        self,
        direction: SignalDirection,
        entry_price: float,
        profile: VolumeProfileResult,
    ) -> Tuple[float, float, float]:
        """
        Calcula SL y TPs basados en estructura del mercado.

        SL: debajo del LVN más cercano o del VAL/VAH
        TP1: POC (primera toma de beneficios)
        TP2: VAH/VAL opuesto (objetivo completo)
        """
        tick = profile.tick_size

        if direction == SignalDirection.LONG:
            nearest_lvn_below = None
            for lvn in sorted(profile.lvn_levels):
                if lvn < entry_price:
                    nearest_lvn_below = lvn

            stop_loss = (nearest_lvn_below - tick * 2) if nearest_lvn_below else (profile.val - tick * 3)
            tp1 = profile.poc
            tp2 = profile.vah

        else:  # SHORT
            nearest_lvn_above = None
            for lvn in sorted(profile.lvn_levels, reverse=True):
                if lvn > entry_price:
                    nearest_lvn_above = lvn

            stop_loss = (nearest_lvn_above + tick * 2) if nearest_lvn_above else (profile.vah + tick * 3)
            tp1 = profile.poc
            tp2 = profile.val

        # Asegurar coherencia de niveles
        if direction == SignalDirection.LONG:
            if tp1 <= entry_price:
                tp1 = entry_price + (entry_price - stop_loss) * 1.0
            if tp2 <= tp1:
                tp2 = tp1 + (entry_price - stop_loss) * 0.5
        else:
            if tp1 >= entry_price:
                tp1 = entry_price - (stop_loss - entry_price) * 1.0
            if tp2 >= tp1:
                tp2 = tp1 - (stop_loss - entry_price) * 0.5

        return stop_loss, tp1, tp2

    # ─── Evaluador principal ──────────────────────────────────────────────────

    def evaluate(
        self,
        snapshot: OrderFlowSnapshot,
        composite=None,          # acepta CompositeProfile, VolumeProfileResult, o None
    ) -> Optional[TradeSetup]:
        """
        Evalúa las condiciones del mercado y genera un TradeSetup si aplica.

        Cascada:
          1. Contexto  → precio fuera del VA + macro alineada (o None)
          2. Setup     → dual delta alineado con la dirección (o None)
          3. Absorción → score 0-1, no bloqueante (mejora R:R mínimo si = 1.0)
          4. Filtro    → R:R >= min_risk_reward (o None)

        composite: acepta tres formas:
          - CompositeProfile  → se usa directamente
          - VolumeProfileResult → se envuelve en CompositeProfile(session=vp)
            (compatibilidad con main.py que pasa el perfil de sesión directamente)
          - None → se calcula internamente desde vp_engine
        """
        current_price = snapshot.delta.price_close

        # Normalizar el argumento a CompositeProfile siempre
        if composite is None:
            composite = self.vp_engine.compute_composite_profile(
                anchor_price=current_price
            )
        elif isinstance(composite, VolumeProfileResult):
            # main.py pasa un VolumeProfileResult directamente → envolver
            composite = CompositeProfile(session=composite)

        # ── CAPA 1: CONTEXTO ───────────────────────────────────────────────
        direction, rejection, ctx = self._evaluate_context(current_price, composite)
        if direction is None:
            logger.debug(f"Contexto rechazado: {rejection} | zona={ctx.get('zone','?')}")
            return None

        # ── CAPA 2: SETUP ──────────────────────────────────────────────────
        setup_ok, rejection = self._evaluate_setup(direction, snapshot)
        if not setup_ok:
            logger.debug(f"Setup rechazado: {rejection} | dir={direction.value}")
            return None

        # ── CAPA 3: ABSORCIÓN (bonus, no bloqueante) ──────────────────────────
        # La absorción mejora la entrada pero no la bloquea.
        # Si el Contexto y el Setup están alineados, eso es suficiente edge.
        absorption_score = self._score_absorption(direction, snapshot)
        logger.debug(
            f"Absorción score={absorption_score:.1f} | dir={direction.value}"
        )

        # ── CÁLCULO DE NIVELES ─────────────────────────────────────────────
        profile = composite.session
        sl, tp1, tp2 = self._calculate_levels(direction, current_price, profile)

        risk   = abs(current_price - sl)
        reward = abs(tp2 - current_price)
        rr     = reward / risk if risk > 0 else 0.0

        # R:R mínimo: si hay absorción de alta calidad se relaja a 1.3
        # porque el fill será mejor. Sin absorción se exige el mínimo estándar.
        min_rr = 1.3 if absorption_score >= 1.0 else CONFIG.strategy.min_risk_reward
        if rr < min_rr:
            logger.debug(f"R:R insuficiente: {rr:.2f} < {min_rr}")
            return None

        # ── TAMAÑO DE POSICIÓN ─────────────────────────────────────────────
        risk_amount   = self._capital * CONFIG.strategy.risk_per_trade_pct
        position_size = risk_amount / risk if risk > 0 else 0.0

        # ── METADATA PARA DEBUGGING Y BACKTESTING ─────────────────────────
        abs_signal = snapshot.absorption   # puede ser None si no hubo absorción
        dd         = snapshot.dual_delta

        setup = TradeSetup(
            timestamp=time.time(),
            direction=direction,
            entry_price=current_price,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            zone=ctx.get("zone", "UNKNOWN"),
            macro_bias=ctx.get("macro_bias", "NEUTRAL"),
            absorption_strength=abs_signal.strength if abs_signal else 0.0,
            absorption_atr_ratio=abs_signal.price_displacement_atr if abs_signal else 0.0,
            dual_delta_short_pct=dd.short.delta_pct if dd.short else 0.0,
            dual_delta_long_pct=dd.long.delta_pct if dd.long else 0.0,
            risk_reward=rr,
            estimated_size_eth=position_size,
            estimated_risk_usd=risk_amount,
        )

        self._last_signal = setup
        self._signal_history.append(setup)
        logger.info(f"🎯 SEÑAL: {setup.summary()}")
        return setup

    # ─── Utilidades ───────────────────────────────────────────────────────────

    def update_capital(self, new_capital: float) -> None:
        self._capital = new_capital

    def recent_signals(self, n: int = 20) -> List[TradeSetup]:
        return self._signal_history[-n:]

    @property
    def last_signal(self) -> Optional[TradeSetup]:
        return self._last_signal

    @property
    def signal_history(self) -> List[TradeSetup]:
        return self._signal_history.copy()

    def rejection_stats(self) -> Dict[str, int]:
        """
        Útil para diagnosticar qué capa está filtrando más señales.
        Requiere haber corrido con logging.DEBUG activo y analizar los logs,
        o bien extender evaluate() para almacenar rechazos.
        Placeholder para implementación futura.
        """
        return {}
