"""
volume_profile.py — Motor de Volume Profile para ETH.

Cambios respecto a la versión anterior:
  - Rango de precio fijado alrededor del precio actual (no dinámico por sesión)
    → los niveles VAH/VAL/POC dejan de saltar entre evaluaciones
  - HVN/LVN con umbrales más estrictos (percentiles subidos)
    → menos ruido, solo niveles realmente significativos
  - CompositeProfile integrado en analyze_price_context()
    → la estrategia ve el contexto diario/semanal sin esfuerzo extra
  - compute_composite_profile() expuesto como método de primera clase
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import numpy as np

from data.data_buffer import DataBuffer
from utils.config import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Estructuras de resultado
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VolumeProfileResult:
    """Resultado completo del Volume Profile para una sesión/ventana."""
    timestamp: float
    price_levels: np.ndarray        # centros de bin (precios)
    volumes: np.ndarray
    buy_volumes: np.ndarray
    sell_volumes: np.ndarray

    poc: float                      # Point of Control
    poc_volume: float

    vah: float                      # Value Area High
    val: float                      # Value Area Low
    value_area_pct: float

    hvn_levels: List[float] = field(default_factory=list)
    lvn_levels: List[float] = field(default_factory=list)

    tick_size: float = 1.0
    total_volume: float = 0.0
    n_trades: int = 0

    developing_poc: Optional[float] = None
    volume_skew: float = 0.0

    @property
    def is_balanced(self) -> bool:
        price_range = self.vah - self.val
        poc_offset = abs(self.poc - (self.val + price_range / 2))
        return poc_offset < price_range * 0.2 if price_range > 0 else True

    @property
    def rotation_factor(self) -> float:
        if self.total_volume == 0 or self.poc_volume == 0:
            return 0.0
        return self.poc_volume / self.total_volume

    def price_in_value_area(self, price: float) -> bool:
        return self.val <= price <= self.vah

    def price_above_value(self, price: float) -> bool:
        return price > self.vah

    def price_below_value(self, price: float) -> bool:
        return price < self.val

    def nearest_hvn(self, price: float) -> Optional[float]:
        if not self.hvn_levels:
            return None
        return min(self.hvn_levels, key=lambda h: abs(h - price))

    def nearest_lvn(self, price: float) -> Optional[float]:
        if not self.lvn_levels:
            return None
        return min(self.lvn_levels, key=lambda l: abs(l - price))

    def support_resistance_levels(self) -> Dict[str, float]:
        return {"POC": self.poc, "VAH": self.vah, "VAL": self.val}


@dataclass
class CompositeProfile:
    """
    Perfil compuesto: diario + semanal.
    Da el contexto macro que faltaba en la versión anterior.

    Uso correcto:
      - Antes de buscar LONG, verificar que el diario no esté en tendencia bajista fuerte
      - El semanal define si la zona de valor actual es relevante a largo plazo
    """
    session: Optional[VolumeProfileResult] = None  # perfil de la sesión actual
    daily: Optional[VolumeProfileResult] = None
    weekly: Optional[VolumeProfileResult] = None

    def get_all_poc_levels(self) -> List[float]:
        pocs = []
        for profile in [self.session, self.daily, self.weekly]:
            if profile:
                pocs.append(profile.poc)
        return pocs

    def get_all_value_areas(self) -> List[Tuple[float, float]]:
        """[(VAL, VAH), ...]"""
        areas = []
        for profile in [self.session, self.daily, self.weekly]:
            if profile:
                areas.append((profile.val, profile.vah))
        return areas

    def macro_bias(self, price: float) -> str:
        """
        Bias macro combinando diario y semanal.
        Retorna 'BULLISH', 'BEARISH' o 'NEUTRAL'.

        Lógica:
          - Si precio está debajo de VAL en ambos → bearish macro
          - Si precio está arriba de VAH en ambos → bullish macro
          - Mixto → neutral

        Este bias NO genera señal por sí solo, pero actúa como filtro:
        un LONG en contexto macro bearish debe omitirse aunque el perfil
        de sesión diga "por debajo del VAL".
        """
        scores = []
        for profile in [self.daily, self.weekly]:
            if profile is None:
                continue
            if profile.price_below_value(price):
                scores.append(-1)
            elif profile.price_above_value(price):
                scores.append(+1)
            else:
                scores.append(0)

        if not scores:
            return "NEUTRAL"
        avg = sum(scores) / len(scores)
        if avg <= -0.5:
            return "BEARISH"
        if avg >= 0.5:
            return "BULLISH"
        return "NEUTRAL"

    def price_in_macro_value(self, price: float) -> bool:
        """True si el precio está dentro del Value Area en diario Y semanal."""
        for profile in [self.daily, self.weekly]:
            if profile and not profile.price_in_value_area(price):
                return False
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Motor principal
# ──────────────────────────────────────────────────────────────────────────────

class VolumeProfileEngine:
    """
    Motor de Volume Profile de alta precisión.

    Cambios clave:
    - compute_profile() acepta ahora anchor_price para fijar el rango
    - _compute_hvn_lvn() usa umbrales percentilados más estrictos (85/15 → 90/10)
    - compute_composite_profile() pasa la sesión actual + diario + semanal
    - analyze_price_context() incluye bias macro del CompositeProfile
    """

    # Rango fijo alrededor del precio actual: ±N dólares
    # Esto estabiliza los bins entre evaluaciones.
    FIXED_RANGE_USD = 200.0  # rango total: precio ± $100

    def __init__(self, buffer: DataBuffer):
        self.buffer = buffer

    # ─── Perfil principal ─────────────────────────────────────────────────────

    def compute_profile(
        self,
        window_seconds: Optional[int] = None,
        tick_size: Optional[float] = None,
        value_area_pct: Optional[float] = None,
        anchor_price: Optional[float] = None,
    ) -> Optional[VolumeProfileResult]:
        """
        Calcula el volume profile completo para la ventana dada.

        anchor_price: si se provee, el rango de bins se fija alrededor de
        este precio (±FIXED_RANGE_USD/2). Esto evita que los niveles cambien
        constantemente al expandirse el rango con nuevos máximos/mínimos.
        Si no se provee, usa el rango dinámico (comportamiento anterior).

        Algoritmo Value Area (estándar CME):
          1. Partir del POC
          2. Expandir hacia arriba o abajo un tick a la vez
          3. Añadir el lado con más volumen
          4. Repetir hasta alcanzar el % objetivo
        """
        ws     = window_seconds or (CONFIG.volume_profile.session_hours * 3600)
        tick   = tick_size or CONFIG.volume_profile.tick_size
        va_pct = value_area_pct or CONFIG.volume_profile.value_area_pct

        trades = self.buffer.get_trades_in_window(ws)
        if len(trades) < 10:
            logger.debug("Insuficientes trades para calcular profile")
            return None

        prices = np.array([t.price for t in trades])
        qtys   = np.array([t.quantity for t in trades])
        sides  = np.array([t.side for t in trades])

        # ── Rango de precio fijado vs dinámico ──────────────────────────────
        if anchor_price is not None:
            half = self.FIXED_RANGE_USD / 2.0
            price_min = anchor_price - half
            price_max = anchor_price + half
        else:
            price_min = float(np.min(prices))
            price_max = float(np.max(prices))

        # Bins de precio
        n_bins    = max(int((price_max - price_min) / tick), 1)
        bin_edges = np.linspace(price_min, price_max + tick, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Volúmenes por bin
        volumes      = np.zeros(n_bins)
        buy_volumes  = np.zeros(n_bins)
        sell_volumes = np.zeros(n_bins)

        bin_indices = np.searchsorted(bin_edges, prices, side='right') - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        for qty, side, idx in zip(qtys, sides, bin_indices):
            volumes[idx] += qty
            if side > 0:
                buy_volumes[idx] += qty
            else:
                sell_volumes[idx] += qty

        total_volume = float(np.sum(volumes))

        # POC
        poc_idx    = int(np.argmax(volumes))
        poc        = float(bin_centers[poc_idx])
        poc_volume = float(volumes[poc_idx])

        # Value Area
        vah_idx, val_idx = self._compute_value_area(volumes, poc_idx, va_pct)
        vah = float(bin_centers[vah_idx])
        val = float(bin_centers[val_idx])

        # HVN / LVN (umbrales más estrictos)
        hvn_levels, lvn_levels = self._compute_hvn_lvn(
            bin_centers, volumes, val_idx, vah_idx
        )

        # Volume skew
        above_poc_vol = float(np.sum(volumes[poc_idx:]))
        below_poc_vol = float(np.sum(volumes[:poc_idx]))
        skew_denom    = above_poc_vol + below_poc_vol
        volume_skew   = (above_poc_vol - below_poc_vol) / skew_denom if skew_denom > 0 else 0.0

        return VolumeProfileResult(
            timestamp=time.time(),
            price_levels=bin_centers,
            volumes=volumes,
            buy_volumes=buy_volumes,
            sell_volumes=sell_volumes,
            poc=poc,
            poc_volume=poc_volume,
            vah=vah,
            val=val,
            value_area_pct=va_pct,
            hvn_levels=hvn_levels,
            lvn_levels=lvn_levels,
            tick_size=tick,
            total_volume=total_volume,
            n_trades=len(trades),
            volume_skew=volume_skew,
        )

    # ─── Algoritmo Value Area ─────────────────────────────────────────────────

    def _compute_value_area(
        self,
        volumes: np.ndarray,
        poc_idx: int,
        va_pct: float,
    ) -> Tuple[int, int]:
        """
        Algoritmo de expansión de Value Area desde el POC (estándar CME).
        Retorna (vah_idx, val_idx).
        """
        total_vol   = float(np.sum(volumes))
        target_vol  = total_vol * va_pct
        accumulated = float(volumes[poc_idx])

        high_idx = poc_idx
        low_idx  = poc_idx
        n        = len(volumes)

        while accumulated < target_vol:
            next_high = high_idx + 1 if high_idx + 1 < n else None
            next_low  = low_idx  - 1 if low_idx  - 1 >= 0 else None

            if next_high is None and next_low is None:
                break

            vol_high = float(volumes[next_high]) if next_high is not None else -1
            vol_low  = float(volumes[next_low])  if next_low  is not None else -1

            if vol_high >= vol_low:
                accumulated += vol_high
                high_idx = next_high
            else:
                accumulated += vol_low
                low_idx = next_low

        return high_idx, low_idx

    # ─── HVN y LVN ────────────────────────────────────────────────────────────

    def _compute_hvn_lvn(
        self,
        bin_centers: np.ndarray,
        volumes: np.ndarray,
        val_idx: int,
        vah_idx: int,
    ) -> Tuple[List[float], List[float]]:
        """
        Detecta HVN y LVN con umbrales más estrictos.

        Versión anterior: percentil 75 (HVN) / percentil 25 (LVN)
        Versión actual:   percentil 90 (HVN) / percentil 10 (LVN)

        Resultado: menos niveles detectados pero todos con relevancia real.
        Un HVN al percentil 75 equivale a decir "el 25% de los precios tiene
        más volumen que este", lo que no es suficientemente selectivo.
        Al percentil 90 solo el 10% lo supera → es un nivel verdaderamente
        significativo donde el mercado quiere estar.
        """
        # Umbrales más estrictos (antes: hvn_percentile / lvn_percentile del config)
        HVN_PERCENTILE = 90  # solo el top 10% de volumen
        LVN_PERCENTILE = 10  # solo el bottom 10%

        nonzero_mask = volumes > 0
        if not np.any(nonzero_mask):
            return [], []

        nonzero_vols  = volumes[nonzero_mask]
        hvn_threshold = float(np.percentile(nonzero_vols, HVN_PERCENTILE))
        lvn_threshold = float(np.percentile(nonzero_vols, LVN_PERCENTILE))

        hvn_levels: List[float] = []
        lvn_levels: List[float] = []

        for i, (price, vol) in enumerate(zip(bin_centers, volumes)):
            if vol >= hvn_threshold:
                hvn_levels.append(float(price))
            elif vol > 0 and vol <= lvn_threshold:
                lvn_levels.append(float(price))

        # Merge clusters cercanos
        hvn_levels = self._merge_nearby_levels(hvn_levels, merge_ticks=5)
        lvn_levels = self._merge_nearby_levels(lvn_levels, merge_ticks=5)

        return hvn_levels, lvn_levels

    def _merge_nearby_levels(
        self, levels: List[float], merge_ticks: int = 5
    ) -> List[float]:
        """Fusiona niveles muy cercanos para evitar ruido."""
        if not levels:
            return []

        tick       = CONFIG.volume_profile.tick_size
        merge_dist = tick * merge_ticks
        sorted_lvls = sorted(levels)
        merged      = [sorted_lvls[0]]

        for lvl in sorted_lvls[1:]:
            if lvl - merged[-1] > merge_dist:
                merged.append(lvl)

        return merged

    # ─── Perfiles por timeframe ────────────────────────────────────────────────

    def compute_session_profile(
        self, anchor_price: Optional[float] = None
    ) -> Optional[VolumeProfileResult]:
        """Perfil de la sesión actual con rango fijo si se da anchor_price."""
        return self.compute_profile(
            window_seconds=CONFIG.volume_profile.session_hours * 3600,
            anchor_price=anchor_price,
        )

    def compute_composite_profile(
        self, anchor_price: Optional[float] = None
    ) -> CompositeProfile:
        """
        Perfil compuesto: sesión + diario + semanal.
        Todos usan anchor_price para coherencia de bins.

        Este es el método que debe usar signals.py como fuente de contexto.
        Sustituye al uso exclusivo del perfil de sesión.
        """
        session = self.compute_profile(
            window_seconds=CONFIG.volume_profile.session_hours * 3600,
            anchor_price=anchor_price,
        )
        daily = self.compute_profile(
            window_seconds=24 * 3600,
            anchor_price=anchor_price,
        )
        weekly = self.compute_profile(
            window_seconds=7 * 24 * 3600,
            anchor_price=anchor_price,
        )

        return CompositeProfile(session=session, daily=daily, weekly=weekly)

    # ─── Análisis contextual ──────────────────────────────────────────────────

    def analyze_price_context(
        self,
        price: float,
        profile: VolumeProfileResult,
        composite: Optional[CompositeProfile] = None,
    ) -> Dict[str, object]:
        """
        Analiza el contexto del precio actual respecto al profile.

        Novedad: si se pasa composite, incluye macro_bias y
        price_in_macro_value para que signals.py pueda filtrar operaciones
        contra tendencia macro.
        """
        context = {
            "price": price,
            "poc": profile.poc,
            "vah": profile.vah,
            "val": profile.val,
            "in_value_area": profile.price_in_value_area(price),
            "above_value": profile.price_above_value(price),
            "below_value": profile.price_below_value(price),
            "distance_to_poc": price - profile.poc,
            "distance_to_vah": price - profile.vah,
            "distance_to_val": price - profile.val,
            "nearest_hvn": profile.nearest_hvn(price),
            "nearest_lvn": profile.nearest_lvn(price),
            "is_balanced": profile.is_balanced,
            "rotation_factor": profile.rotation_factor,
            "volume_skew": profile.volume_skew,
            # Contexto macro (por defecto neutral si no hay composite)
            "macro_bias": "NEUTRAL",
            "price_in_macro_value": False,
        }

        if composite is not None:
            context["macro_bias"]            = composite.macro_bias(price)
            context["price_in_macro_value"]  = composite.price_in_macro_value(price)

        # Clasificar zona del precio respecto al perfil de sesión
        if profile.price_in_value_area(price):
            if abs(price - profile.poc) < profile.tick_size * 2:
                context["zone"] = "AT_POC"
            else:
                context["zone"] = "IN_VALUE_AREA"
        elif profile.price_above_value(price):
            nearest_hvn = profile.nearest_hvn(price)
            if nearest_hvn and abs(price - nearest_hvn) < profile.tick_size * 3:
                context["zone"] = "AT_HVN_ABOVE"
            else:
                nearest_lvn = profile.nearest_lvn(price)
                if nearest_lvn and price > nearest_lvn:
                    context["zone"] = "ABOVE_LVN"
                else:
                    context["zone"] = "ABOVE_VALUE_AREA"
        else:
            nearest_hvn = profile.nearest_hvn(price)
            if nearest_hvn and abs(price - nearest_hvn) < profile.tick_size * 3:
                context["zone"] = "AT_HVN_BELOW"
            else:
                context["zone"] = "BELOW_VALUE_AREA"

        return context
