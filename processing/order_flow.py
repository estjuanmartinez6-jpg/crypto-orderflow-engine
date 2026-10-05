"""
order_flow.py — Motor de Order Flow para ETH.

Cambios respecto a la versión anterior:
  - Delta en DOS capas: corto (trigger) y largo (filtro de dirección)
  - Absorción mejorada: requiere volatilidad real, ubicación en zona clave,
    persistencia mínima y delta fuerte relativo al ATR
  - Divergencia eliminada: demasiado ruido en crypto intradía
  - compute_atr() añadido: necesario para normalizar absorción

Toda la aritmética usa NumPy para máxima velocidad.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np

from data.data_buffer import DataBuffer, OrderBook
from utils.config import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Estructuras de resultado
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DeltaResult:
    """Resultado del cálculo de delta en una ventana temporal."""
    timestamp: float
    delta: float                  # delta acumulado de la ventana
    buy_volume: float
    sell_volume: float
    total_volume: float
    delta_pct: float              # delta / total_volume * 100
    price_open: float
    price_close: float
    price_high: float
    price_low: float
    n_trades: int
    n_buy_trades: int
    n_sell_trades: int
    window_seconds: int           # duración de la ventana usada

    @property
    def is_bullish(self) -> bool:
        return self.delta > 0

    @property
    def buy_sell_ratio(self) -> float:
        if self.sell_volume == 0:
            return float('inf')
        return self.buy_volume / self.sell_volume


@dataclass
class DualDeltaResult:
    """
    Delta en dos capas temporales.

    short → trigger de entrada (10-30 segundos)
    long  → filtro de dirección (2-5 minutos)

    Uso correcto en signals.py:
      - long confirma la dirección (¿es coherente con el contexto?)
      - short es el gatillo (¿está pasando AHORA?)
    """
    short: Optional[DeltaResult]   # ventana corta: reacción inmediata
    long: Optional[DeltaResult]    # ventana larga: contexto de flujo

    @property
    def aligned_bullish(self) -> bool:
        """Ambas capas apuntan al alza."""
        if self.short is None or self.long is None:
            return False
        return self.short.delta > 0 and self.long.delta > 0

    @property
    def aligned_bearish(self) -> bool:
        """Ambas capas apuntan a la baja."""
        if self.short is None or self.long is None:
            return False
        return self.short.delta < 0 and self.long.delta < 0

    @property
    def short_reversal_bullish(self) -> bool:
        """
        Long negativo pero short positivo: posible giro alcista.
        Útil para detectar capitulaciones donde los vendedores se agotan.
        """
        if self.short is None or self.long is None:
            return False
        return self.long.delta < 0 and self.short.delta > 0

    @property
    def short_reversal_bearish(self) -> bool:
        """Long positivo pero short negativo: posible giro bajista."""
        if self.short is None or self.long is None:
            return False
        return self.long.delta > 0 and self.short.delta < 0


@dataclass
class CVDResult:
    """Serie temporal del CVD."""
    timestamps: np.ndarray
    cvd_values: np.ndarray
    delta_series: np.ndarray
    price_series: np.ndarray

    @property
    def current_cvd(self) -> float:
        return float(self.cvd_values[-1]) if len(self.cvd_values) > 0 else 0.0

    @property
    def cvd_slope(self) -> float:
        """Pendiente del CVD en las últimas 5 observaciones."""
        if len(self.cvd_values) < 5:
            return 0.0
        y = self.cvd_values[-5:]
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)


@dataclass
class AbsorptionSignal:
    """
    Señal de absorción institucional.

    Criterios endurecidos respecto a la versión anterior:
      - price_displacement normalizado por ATR (no absoluto)
      - requiere persistencia (n_ticks_confirmed >= umbral)
      - solo válida si ocurre cerca de zona clave (VAL/VAH/HVN)
    """
    timestamp: float
    price: float
    direction: str                # 'BUY_ABSORPTION' | 'SELL_ABSORPTION'
    absorbed_volume: float
    price_displacement: float     # absoluto en USD
    price_displacement_atr: float # normalizado: displacement / ATR
    delta: float
    strength: float               # 0-1
    n_ticks_confirmed: int        # cuántos ticks duró la absorción
    at_key_zone: bool             # si ocurre en VAL/VAH/HVN (contexto externo)

    @property
    def is_buy_absorption(self) -> bool:
        """Volumen vendedor absorbido → señal alcista."""
        return self.direction == 'BUY_ABSORPTION'

    @property
    def is_sell_absorption(self) -> bool:
        """Volumen comprador absorbido → señal bajista."""
        return self.direction == 'SELL_ABSORPTION'

    @property
    def is_high_quality(self) -> bool:
        """
        Absorción de alta calidad: fuerte, persistente y en zona relevante.
        Solo estas deben usarse como trigger de entrada.
        """
        return (
            self.strength >= 0.6
            and self.n_ticks_confirmed >= 3
            and self.at_key_zone
            and self.price_displacement_atr < 0.3
        )


@dataclass
class ImbalanceResult:
    """Resultado del análisis de imbalance del libro."""
    timestamp: float
    imbalance: float              # -1 a +1
    bid_volume: float
    ask_volume: float
    best_bid: float
    best_ask: float
    spread: float
    weighted_mid: float
    n_levels_used: int
    signal: str                   # 'BULLISH' | 'BEARISH' | 'NEUTRAL'

    @property
    def is_bullish(self) -> bool:
        return self.imbalance > CONFIG.order_flow.imbalance_threshold

    @property
    def is_bearish(self) -> bool:
        return self.imbalance < -CONFIG.order_flow.imbalance_threshold


@dataclass
class FootprintLevel:
    """Actividad compradora/vendedora en un nivel de precio específico."""
    price: float
    buy_volume: float
    sell_volume: float
    delta: float
    total_volume: float

    @property
    def imbalance(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return self.delta / self.total_volume


@dataclass
class OrderFlowSnapshot:
    """
    Snapshot completo del estado del order flow.

    dual_delta reemplaza a delta simple: expone capas corta y larga.
    divergence_detected/type eliminados: demasiado ruidosos en crypto intradía.
    """
    timestamp: float
    delta: DeltaResult            # delta largo (compatibilidad con dashboard)
    dual_delta: DualDeltaResult   # delta en dos capas (uso en signals)
    cvd: CVDResult
    imbalance: Optional[ImbalanceResult]
    absorption: Optional[AbsorptionSignal]
    footprint: List[FootprintLevel]
    atr: float                    # ATR de la sesión (para normalización)


# ──────────────────────────────────────────────────────────────────────────────
# Motor principal
# ──────────────────────────────────────────────────────────────────────────────

class OrderFlowEngine:
    """
    Motor de Order Flow de alta precisión.

    Métodos principales:
      compute_delta(window_s)          → DeltaResult
      compute_dual_delta()             → DualDeltaResult
      compute_cvd(window_s, bins)      → CVDResult
      compute_imbalance(book)          → ImbalanceResult
      detect_absorption(atr, context)  → Optional[AbsorptionSignal]
      compute_atr(window_s)            → float
      compute_footprint(tick)          → List[FootprintLevel]
      snapshot()                       → OrderFlowSnapshot completo
    """

    # Ventanas de delta (segundos)
    DELTA_SHORT_WINDOW = 20    # trigger: reacción inmediata
    DELTA_LONG_WINDOW  = 300   # filtro: contexto de 5 minutos

    # Absorción: mínimos de persistencia
    ABSORPTION_MIN_TICKS = 3   # ticks consecutivos que deben confirmar

    def __init__(self, buffer: DataBuffer):
        self.buffer = buffer

    # ─── Delta ────────────────────────────────────────────────────────────────

    def compute_delta(self, window_seconds: Optional[int] = None) -> Optional[DeltaResult]:
        """
        Calcula delta acumulado en la ventana dada.
        Delta = Σ(qty_i * side_i), side_i ∈ {+1, -1}
        """
        ws = window_seconds or self.DELTA_LONG_WINDOW
        trades = self.buffer.get_trades_in_window(ws)

        if len(trades) < 2:
            return None

        data = self.buffer.to_numpy(len(trades))
        prices = data["prices"]
        qtys   = data["quantities"]
        sides  = data["sides"]

        buy_mask  = sides > 0
        sell_mask = sides < 0

        buy_vol   = float(np.sum(qtys[buy_mask]))
        sell_vol  = float(np.sum(qtys[sell_mask]))
        total_vol = buy_vol + sell_vol
        delta     = buy_vol - sell_vol
        delta_pct = (delta / total_vol * 100) if total_vol > 0 else 0.0

        return DeltaResult(
            timestamp=time.time(),
            delta=delta,
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            total_volume=total_vol,
            delta_pct=delta_pct,
            price_open=float(prices[0]),
            price_close=float(prices[-1]),
            price_high=float(np.max(prices)),
            price_low=float(np.min(prices)),
            n_trades=len(trades),
            n_buy_trades=int(np.sum(buy_mask)),
            n_sell_trades=int(np.sum(sell_mask)),
            window_seconds=ws,
        )

    def compute_dual_delta(self) -> DualDeltaResult:
        """
        Calcula delta en dos capas temporales simultáneas.

        short (20s): ¿qué está pasando AHORA? → trigger
        long (5min): ¿qué ha pasado recientemente? → filtro de dirección

        La capa corta debe alinearse con la dirección del contexto (VP)
        y con la capa larga antes de generar cualquier señal.
        """
        short = self.compute_delta(self.DELTA_SHORT_WINDOW)
        long_ = self.compute_delta(self.DELTA_LONG_WINDOW)
        return DualDeltaResult(short=short, long=long_)

    # ─── CVD ──────────────────────────────────────────────────────────────────

    def compute_cvd(
        self,
        window_seconds: int = 3600,
        n_bins: int = 60,
    ) -> Optional[CVDResult]:
        """
        Calcula el CVD segmentado en n_bins ventanas temporales.
        CVD[t] = CVD[t-1] + Delta[t]
        """
        trades = self.buffer.get_trades_in_window(window_seconds)
        if len(trades) < n_bins:
            return None

        data = self.buffer.to_numpy(len(trades))
        timestamps = data["timestamps"]
        qtys       = data["quantities"]
        sides      = data["sides"]
        prices     = data["prices"]

        t_start   = timestamps[0]
        t_end     = timestamps[-1]
        bin_edges = np.linspace(t_start, t_end, n_bins + 1)

        deltas       = np.zeros(n_bins)
        close_prices = np.zeros(n_bins)

        for i in range(n_bins):
            mask = (timestamps >= bin_edges[i]) & (timestamps < bin_edges[i + 1])
            if np.any(mask):
                deltas[i]       = float(np.sum(qtys[mask] * sides[mask]))
                close_prices[i] = float(prices[mask][-1])
            else:
                close_prices[i] = close_prices[i - 1] if i > 0 else float(prices[0])

        cvd         = np.cumsum(deltas)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return CVDResult(
            timestamps=bin_centers,
            cvd_values=cvd,
            delta_series=deltas,
            price_series=close_prices,
        )

    # ─── ATR ──────────────────────────────────────────────────────────────────

    def compute_atr(self, window_seconds: int = 300, n_bars: int = 20) -> float:
        """
        Calcula el ATR (Average True Range) de la ventana dada.

        Necesario para normalizar la absorción: un precio que se mueve $5
        en un mercado volátil ($20 ATR) no es absorción; el mismo $5 en un
        mercado tranquilo ($8 ATR) sí lo es.

        Divide la ventana en n_bars pseudo-velas y calcula el True Range promedio.
        Retorna 0.0 si no hay suficientes datos.
        """
        trades = self.buffer.get_trades_in_window(window_seconds)
        if len(trades) < n_bars * 2:
            return 0.0

        data   = self.buffer.to_numpy(len(trades))
        prices = data["prices"]
        times  = data["timestamps"]

        t_start   = times[0]
        t_end     = times[-1]
        bin_edges = np.linspace(t_start, t_end, n_bars + 1)

        highs  = np.zeros(n_bars)
        lows   = np.zeros(n_bars)
        closes = np.zeros(n_bars)

        for i in range(n_bars):
            mask = (times >= bin_edges[i]) & (times < bin_edges[i + 1])
            if np.any(mask):
                bar_prices = prices[mask]
                highs[i]   = float(np.max(bar_prices))
                lows[i]    = float(np.min(bar_prices))
                closes[i]  = float(bar_prices[-1])
            else:
                highs[i]  = closes[i - 1] if i > 0 else float(prices[0])
                lows[i]   = closes[i - 1] if i > 0 else float(prices[0])
                closes[i] = closes[i - 1] if i > 0 else float(prices[0])

        # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
        prev_closes = np.roll(closes, 1)
        prev_closes[0] = closes[0]

        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - prev_closes),
                np.abs(lows - prev_closes),
            )
        )

        return float(np.mean(tr))

    # ─── Imbalance del libro ──────────────────────────────────────────────────

    def compute_imbalance(
        self,
        book: Optional[OrderBook] = None,
        n_levels: int = 10,
    ) -> Optional[ImbalanceResult]:
        """
        Analiza el imbalance del order book.
        Imbalance = (BidVol - AskVol) / (BidVol + AskVol), rango [-1, +1].
        """
        b = book or self.buffer.get_orderbook()
        if b is None or not b.bids or not b.asks:
            return None

        imb = b.imbalance(n_levels)
        bv  = b.bid_volume(n_levels)
        av  = b.ask_volume(n_levels)

        threshold = CONFIG.order_flow.imbalance_threshold
        if imb > threshold:
            signal = "BULLISH"
        elif imb < -threshold:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"

        return ImbalanceResult(
            timestamp=time.time(),
            imbalance=imb,
            bid_volume=bv,
            ask_volume=av,
            best_bid=b.best_bid or 0.0,
            best_ask=b.best_ask or 0.0,
            spread=b.spread or 0.0,
            weighted_mid=b.weighted_mid(n_levels) or b.mid_price or 0.0,
            n_levels_used=n_levels,
            signal=signal,
        )

    # ─── Absorción ────────────────────────────────────────────────────────────

    def detect_absorption(
        self,
        atr: float,
        near_key_zone: bool = False,
        window_seconds: int = 30,
        volume_threshold: Optional[float] = None,
    ) -> Optional[AbsorptionSignal]:
        """
        Detecta absorción institucional con criterios más estrictos.

        Mejoras sobre la versión anterior:
          1. El desplazamiento de precio se normaliza por el ATR actual,
             no por un umbral absoluto. Un $3 de movimiento es poco en un
             mercado con ATR $20, pero es normal en ATR $5.
          2. Se requiere persistencia: el mercado debe mantener el patrón
             durante al menos ABSORPTION_MIN_TICKS periodos cortos.
          3. near_key_zone: la absorción en medio del rango no tiene edge;
             el caller (signals.py) debe pasar si estamos en zona clave.

        Parámetros:
          atr             ATR actual (normaliza el umbral de desplazamiento)
          near_key_zone   True si el precio está en VAL/VAH/HVN
          window_seconds  ventana de análisis (default 30s = absorción reciente)
          volume_threshold umbral de volumen mínimo (default config)
        """
        # Sin ATR no podemos normalizar → no podemos detectar absorción real
        if atr <= 0:
            logger.debug("ATR <= 0, imposible detectar absorción")
            return None

        v_thresh = volume_threshold or CONFIG.order_flow.absorption_volume_threshold

        trades = self.buffer.get_trades_in_window(window_seconds)
        if len(trades) < 10:
            return None

        data   = self.buffer.to_numpy(len(trades))
        prices = data["prices"]
        qtys   = data["quantities"]
        sides  = data["sides"]
        times  = data["timestamps"]

        total_delta = float(np.sum(qtys * sides))
        total_vol   = float(np.sum(qtys))
        price_displacement = abs(float(prices[-1]) - float(prices[0]))

        # Volumen insuficiente para considerar absorción real
        if total_vol < v_thresh:
            return None

        # Delta debe ser suficientemente fuerte (>50% del volumen total)
        delta_ratio = abs(total_delta) / (total_vol + 1e-9)
        if delta_ratio < 0.40:
            return None

        # Movimiento de precio normalizado por ATR
        # Si el precio se mueve menos del 25% del ATR con mucho volumen → absorción
        displacement_atr = price_displacement / atr
        DISPLACEMENT_ATR_THRESHOLD = 0.25

        if displacement_atr >= DISPLACEMENT_ATR_THRESHOLD:
            return None

        # Verificar persistencia: dividir en ABSORPTION_MIN_TICKS sub-ventanas
        # En cada una debe mantenerse la misma dirección de delta
        n_ticks = self.ABSORPTION_MIN_TICKS
        if len(times) < n_ticks:
            n_ticks_confirmed = 1
        else:
            t_start   = times[0]
            t_end     = times[-1]
            bin_edges = np.linspace(t_start, t_end, n_ticks + 1)
            target_sign = np.sign(total_delta)
            confirmed = 0
            for i in range(n_ticks):
                mask = (times >= bin_edges[i]) & (times < bin_edges[i + 1])
                if np.any(mask):
                    tick_delta = float(np.sum(qtys[mask] * sides[mask]))
                    if np.sign(tick_delta) == target_sign:
                        confirmed += 1
            n_ticks_confirmed = confirmed

        if n_ticks_confirmed < self.ABSORPTION_MIN_TICKS:
            return None

        # Dirección de la absorción
        buy_absorbed  = total_delta < 0  # ventas absorbidas → bullish
        sell_absorbed = total_delta > 0  # compras absorbidas → bearish
        direction = "BUY_ABSORPTION" if buy_absorbed else "SELL_ABSORPTION"

        # Fuerza: penaliza movimiento de precio y premia volumen
        # Más volumen + menos movimiento relativo al ATR = señal más fuerte
        strength = min(
            delta_ratio * (1 - displacement_atr / DISPLACEMENT_ATR_THRESHOLD),
            1.0
        )

        return AbsorptionSignal(
            timestamp=time.time(),
            price=float(prices[-1]),
            direction=direction,
            absorbed_volume=abs(total_delta),
            price_displacement=price_displacement,
            price_displacement_atr=displacement_atr,
            delta=total_delta,
            strength=strength,
            n_ticks_confirmed=n_ticks_confirmed,
            at_key_zone=near_key_zone,
        )

    # ─── Footprint ────────────────────────────────────────────────────────────

    def compute_footprint(
        self,
        window_seconds: int = 300,
        tick_size: Optional[float] = None,
    ) -> List[FootprintLevel]:
        """
        Genera footprint chart: volumen comprador/vendedor por nivel de precio.
        Permite ver exactamente dónde compraron y vendieron los agresivos.
        """
        tick   = tick_size or CONFIG.volume_profile.tick_size
        trades = self.buffer.get_trades_in_window(window_seconds)
        if not trades:
            return []

        data: Dict[float, Dict[str, float]] = {}

        for t in trades:
            bin_price = round(t.price / tick) * tick
            if bin_price not in data:
                data[bin_price] = {"buy": 0.0, "sell": 0.0}
            if t.side > 0:
                data[bin_price]["buy"] += t.quantity
            else:
                data[bin_price]["sell"] += t.quantity

        result = []
        for price, vols in sorted(data.items()):
            bv = vols["buy"]
            sv = vols["sell"]
            result.append(FootprintLevel(
                price=price,
                buy_volume=bv,
                sell_volume=sv,
                delta=bv - sv,
                total_volume=bv + sv,
            ))

        return result

    # ─── Snapshot completo ────────────────────────────────────────────────────

    def snapshot(self, near_key_zone: bool = False) -> Optional[OrderFlowSnapshot]:
        """
        Calcula y retorna un snapshot completo del estado del order flow.

        near_key_zone: el caller debe indicar si el precio está en zona clave
        para que la detección de absorción sea contextualmente válida.
        """
        dual_delta = self.compute_dual_delta()

        # Usamos el delta largo como referencia primaria (compatibilidad dashboard)
        delta = dual_delta.long
        if delta is None:
            return None

        atr        = self.compute_atr()
        cvd        = self.compute_cvd()
        imbalance  = self.compute_imbalance()
        absorption = self.detect_absorption(atr=atr, near_key_zone=near_key_zone)
        footprint  = self.compute_footprint()

        return OrderFlowSnapshot(
            timestamp=time.time(),
            delta=delta,
            dual_delta=dual_delta,
            cvd=cvd or CVDResult(
                timestamps=np.array([]),
                cvd_values=np.array([]),
                delta_series=np.array([]),
                price_series=np.array([]),
            ),
            imbalance=imbalance,
            absorption=absorption,
            footprint=footprint,
            atr=atr,
        )
