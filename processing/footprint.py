"""
processing/footprint.py

Footprint Chart (Cluster Chart) para ETH Futures.

El footprint chart es la herramienta de microestructura MÁS PRECISA:
muestra el buy/sell volume por cada precio dentro de cada vela (barra).

Estructura de una "celda" del footprint:
  [precio] [buy_vol × sell_vol] → delta de ese nivel

Patrones clave detectados:
  - Stacked Imbalances: múltiples niveles con imbalance unidireccional
  - Finishing Drive: última celda con volumen muy alto (climax de tendencia)
  - Absorption Pattern: mucho volumen pero precio no avanza
  - POC Flip: el nivel de máximo volumen cambia de barra en barra
  - Unfinished Auction: precio cierra en el extremo (mercado quiere seguir)

Referencia: metodología de Jigsaw Trading y Bookmap footprint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.config import VolumeProfileConfig
from utils.logger import get_logger

log = get_logger("FootprintChart")


# ─── Estructuras ──────────────────────────────────────────────────────────────

@dataclass
class FootprintCell:
    """Una celda del footprint: un precio dentro de una barra."""
    price: float
    buy_volume: float
    sell_volume: float

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def delta_pct(self) -> float:
        return self.delta / self.total_volume if self.total_volume > 0 else 0.0

    @property
    def imbalance_ratio(self) -> float:
        """
        Ratio de imbalance (usado por Jigsaw):
        buy/sell > 3.0 o sell/buy > 3.0 → imbalance significativo
        """
        if self.sell_volume > 0:
            return self.buy_volume / self.sell_volume
        return float("inf") if self.buy_volume > 0 else 1.0

    @property
    def has_bid_imbalance(self) -> bool:
        """Más compradores que vendedores por factor 3x."""
        return self.imbalance_ratio >= 3.0

    @property
    def has_ask_imbalance(self) -> bool:
        """Más vendedores que compradores por factor 3x."""
        return self.sell_volume > 0 and (self.sell_volume / max(self.buy_volume, 0.001)) >= 3.0

    def format(self) -> str:
        return f"${self.price:.1f} [{self.buy_volume:.2f} × {self.sell_volume:.2f}] Δ={self.delta:+.2f}"


@dataclass
class FootprintBar:
    """
    Una barra completa del footprint chart.
    Equivale a una vela OHLC pero con microestructura interna.
    """
    bar_id: int
    open_time: float
    close_time: float
    duration_seconds: float

    # OHLC estándar
    open: float
    high: float
    low: float
    close: float

    # Cells: dict precio → FootprintCell
    cells: Dict[float, FootprintCell] = field(default_factory=dict)

    # Métricas de la barra
    total_volume: float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    delta: float = 0.0
    cvd_at_close: float = 0.0
    poc_price: float = 0.0

    # Patrones detectados
    stacked_bid_imbalances: int = 0    # Niveles consecutivos con bid imbalance
    stacked_ask_imbalances: int = 0
    is_finishing_drive: bool = False   # Volumen climático al final
    is_absorption: bool = False
    is_unfinished_auction: bool = False

    @property
    def body_size(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def delta_pct(self) -> float:
        return self.delta / self.total_volume if self.total_volume > 0 else 0.0

    def get_sorted_cells(self, ascending: bool = True) -> List[FootprintCell]:
        return sorted(self.cells.values(), key=lambda c: c.price, reverse=not ascending)

    def get_poc_cell(self) -> Optional[FootprintCell]:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda c: c.total_volume)


@dataclass
class StackedImbalance:
    """
    Zona de stacked imbalances: múltiples niveles consecutivos
    con imbalance en la misma dirección.
    Son zonas de referencia muy fuertes para rebotes.
    """
    bar_id: int
    direction: str       # "BID" (bullish) o "ASK" (bearish)
    price_high: float
    price_low: float
    count: int           # Número de niveles apilados
    avg_imbalance_ratio: float
    timestamp: float


@dataclass
class FootprintPattern:
    """Patrón detectado en el footprint."""
    pattern_type: str
    bar_id: int
    price: float
    direction: str       # "BULLISH" o "BEARISH"
    strength: float      # 0-1
    description: str
    timestamp: float


# ─── Motor de Footprint ───────────────────────────────────────────────────────

class FootprintEngine:
    """
    Construye y analiza footprint charts en tiempo real.

    Proceso:
    1. Recibe trades del buffer
    2. Agrupa por barra de tiempo (ej: 1 minuto)
    3. Dentro de cada barra, agrupa por precio (tick_size)
    4. Detecta patrones de microestructura
    """

    def __init__(
        self,
        tick_size: float = 1.0,
        bar_duration_seconds: int = 60,   # Barras de 1 minuto por defecto
        min_imbalance_ratio: float = 3.0,
        min_stacked_levels: int = 3,
    ):
        self.tick_size = tick_size
        self.bar_duration = bar_duration_seconds
        self.min_imbalance_ratio = min_imbalance_ratio
        self.min_stacked_levels = min_stacked_levels

        self._bars: List[FootprintBar] = []
        self._current_bar: Optional[FootprintBar] = None
        self._bar_counter = 0
        self._session_cvd = 0.0
        self._detected_patterns: List[FootprintPattern] = []
        self._stacked_imbalances: List[StackedImbalance] = []

    def _price_bin(self, price: float) -> float:
        return round(price / self.tick_size) * self.tick_size

    def _bar_open_time(self, timestamp: float) -> float:
        """Retorna el timestamp de inicio de la barra que contiene este ts."""
        return (timestamp // self.bar_duration) * self.bar_duration

    def _new_bar(self, timestamp: float, first_price: float) -> FootprintBar:
        open_t = self._bar_open_time(timestamp)
        self._bar_counter += 1
        return FootprintBar(
            bar_id=self._bar_counter,
            open_time=open_t,
            close_time=open_t + self.bar_duration,
            duration_seconds=self.bar_duration,
            open=first_price, high=first_price,
            low=first_price, close=first_price,
        )

    # ─── Ingesta de trades ────────────────────────────────────────────────────

    def process_trades_array(self, arr: np.ndarray) -> List[FootprintBar]:
        """
        Procesa un array numpy de trades [timestamp, price, qty, side].
        Retorna las barras completadas en esta actualización.
        """
        if len(arr) == 0:
            return []

        newly_closed: List[FootprintBar] = []

        for row in arr:
            ts, price, qty, side = row

            # Inicializar primera barra
            if self._current_bar is None:
                self._current_bar = self._new_bar(ts, price)

            # ¿El trade pertenece a una nueva barra?
            if ts >= self._current_bar.close_time:
                closed = self._close_current_bar()
                newly_closed.append(closed)
                self._bars.append(closed)

                # Puede haber saltos de múltiples barras (datos de baja frecuencia)
                next_open = self._bar_open_time(ts)
                while next_open > self._current_bar.close_time:
                    empty_bar = FootprintBar(
                        bar_id=self._bar_counter,
                        open_time=self._current_bar.close_time,
                        close_time=self._current_bar.close_time + self.bar_duration,
                        duration_seconds=self.bar_duration,
                        open=price, high=price, low=price, close=price,
                    )
                    self._bar_counter += 1
                    self._bars.append(empty_bar)
                    newly_closed.append(empty_bar)

                self._current_bar = self._new_bar(ts, price)

            # Actualizar barra actual
            bar = self._current_bar
            bar.close = price
            bar.high = max(bar.high, price)
            bar.low = min(bar.low, price)

            bin_p = self._price_bin(price)
            if bin_p not in bar.cells:
                bar.cells[bin_p] = FootprintCell(price=bin_p, buy_volume=0.0, sell_volume=0.0)

            cell = bar.cells[bin_p]
            if side == 1:
                cell.buy_volume += qty
                bar.buy_volume += qty
            elif side == -1:
                cell.sell_volume += qty
                bar.sell_volume += qty

            bar.total_volume += qty
            bar.delta = bar.buy_volume - bar.sell_volume

        # Detectar patrones en barras recién cerradas
        for bar in newly_closed:
            self._compute_bar_metrics(bar)
            self._detect_patterns(bar)

        return newly_closed

    def _close_current_bar(self) -> FootprintBar:
        bar = self._current_bar
        self._session_cvd += bar.delta
        bar.cvd_at_close = self._session_cvd
        self._compute_bar_metrics(bar)
        return bar

    def _compute_bar_metrics(self, bar: FootprintBar) -> None:
        if not bar.cells:
            return

        # POC de la barra
        poc_cell = bar.get_poc_cell()
        bar.poc_price = poc_cell.price if poc_cell else bar.close

        # Contar stacked imbalances
        bar.stacked_bid_imbalances = self._count_stacked_imbalances(bar, "BID")
        bar.stacked_ask_imbalances = self._count_stacked_imbalances(bar, "ASK")

        # Finishing Drive: última celda (dirección de la barra) con volumen alto
        bar.is_finishing_drive = self._detect_finishing_drive(bar)

        # Absorción: mucho volumen pero body pequeño
        bar.is_absorption = (
            bar.total_volume > 0 and
            bar.body_size / max(bar.high - bar.low, 0.001) < 0.2 and
            abs(bar.delta_pct) > 0.4
        )

        # Unfinished Auction: precio cierra en el extremo
        price_range = bar.high - bar.low
        if price_range > 0:
            close_position = (bar.close - bar.low) / price_range
            bar.is_unfinished_auction = close_position > 0.95 or close_position < 0.05

    def _count_stacked_imbalances(self, bar: FootprintBar, direction: str) -> int:
        """Cuenta el máximo número de imbalances apilados consecutivos."""
        cells = bar.get_sorted_cells(ascending=True)
        if len(cells) < 2:
            return 0

        max_stack = 0
        current_stack = 0

        for i, cell in enumerate(cells[:-1]):
            # Comparar celda actual con la celda un nivel arriba
            next_cell = cells[i + 1]

            if direction == "BID":
                # Imbalance comprador: bid de celda actual vs ask de celda superior
                # Regla: bid_vol_actual >= factor * ask_vol_next
                ratio = cell.buy_volume / max(next_cell.sell_volume, 0.001)
            else:
                # Imbalance vendedor: ask de celda superior vs bid de celda actual
                ratio = next_cell.sell_volume / max(cell.buy_volume, 0.001)

            if ratio >= self.min_imbalance_ratio:
                current_stack += 1
                max_stack = max(max_stack, current_stack)
            else:
                current_stack = 0

        return max_stack

    def _detect_finishing_drive(self, bar: FootprintBar) -> bool:
        """
        Finishing Drive: el volumen en el extremo de la barra es
        significativamente más alto que el promedio de las demás celdas.
        Indica posible agotamiento de la tendencia (reversión inminente).
        """
        if not bar.cells or len(bar.cells) < 3:
            return False

        cells = bar.get_sorted_cells(ascending=bar.is_bullish)
        extreme_cell = cells[-1]  # La celda más alejada en la dirección de la barra
        other_vols = [c.total_volume for c in cells[:-1]]

        if not other_vols:
            return False

        avg_other = np.mean(other_vols)
        return extreme_cell.total_volume >= avg_other * 2.5

    # ─── Detección de patrones ────────────────────────────────────────────────

    def _detect_patterns(self, bar: FootprintBar) -> None:
        """Detecta y registra patrones de alta probabilidad."""
        patterns = []

        # Stacked Bid Imbalances → Bullish
        if bar.stacked_bid_imbalances >= self.min_stacked_levels:
            cells = [c for c in bar.cells.values() if c.has_bid_imbalance]
            if cells:
                bottom_price = min(c.price for c in cells)
                top_price = max(c.price for c in cells)
                strength = min(1.0, bar.stacked_bid_imbalances / 10)

                si = StackedImbalance(
                    bar_id=bar.bar_id,
                    direction="BID",
                    price_high=top_price,
                    price_low=bottom_price,
                    count=bar.stacked_bid_imbalances,
                    avg_imbalance_ratio=self.min_imbalance_ratio,
                    timestamp=bar.open_time,
                )
                self._stacked_imbalances.append(si)

                patterns.append(FootprintPattern(
                    pattern_type="STACKED_BID_IMBALANCE",
                    bar_id=bar.bar_id,
                    price=(top_price + bottom_price) / 2,
                    direction="BULLISH",
                    strength=strength,
                    description=f"{bar.stacked_bid_imbalances} niveles de bid imbalance apilados en ${bottom_price:.1f}-${top_price:.1f}",
                    timestamp=bar.open_time,
                ))

        # Stacked Ask Imbalances → Bearish
        if bar.stacked_ask_imbalances >= self.min_stacked_levels:
            cells = [c for c in bar.cells.values() if c.has_ask_imbalance]
            if cells:
                bottom_price = min(c.price for c in cells)
                top_price = max(c.price for c in cells)
                strength = min(1.0, bar.stacked_ask_imbalances / 10)

                si = StackedImbalance(
                    bar_id=bar.bar_id,
                    direction="ASK",
                    price_high=top_price,
                    price_low=bottom_price,
                    count=bar.stacked_ask_imbalances,
                    avg_imbalance_ratio=self.min_imbalance_ratio,
                    timestamp=bar.open_time,
                )
                self._stacked_imbalances.append(si)

                patterns.append(FootprintPattern(
                    pattern_type="STACKED_ASK_IMBALANCE",
                    bar_id=bar.bar_id,
                    price=(top_price + bottom_price) / 2,
                    direction="BEARISH",
                    strength=strength,
                    description=f"{bar.stacked_ask_imbalances} niveles de ask imbalance apilados en ${bottom_price:.1f}-${top_price:.1f}",
                    timestamp=bar.open_time,
                ))

        # Finishing Drive con divergencia delta
        if bar.is_finishing_drive:
            direction_str = "BEARISH" if bar.is_bullish else "BULLISH"
            patterns.append(FootprintPattern(
                pattern_type="FINISHING_DRIVE",
                bar_id=bar.bar_id,
                price=bar.close,
                direction=direction_str,
                strength=0.7,
                description=f"Finishing drive en ${bar.close:.2f} — posible agotamiento de {'alcista' if bar.is_bullish else 'bajista'}",
                timestamp=bar.open_time,
            ))

        # Absorción
        if bar.is_absorption:
            direction_str = "BULLISH" if bar.delta < 0 else "BEARISH"
            patterns.append(FootprintPattern(
                pattern_type="ABSORPTION",
                bar_id=bar.bar_id,
                price=bar.poc_price,
                direction=direction_str,
                strength=min(1.0, abs(bar.delta_pct) * 1.5),
                description=f"Absorción: delta {bar.delta:+.2f} pero cuerpo pequeño ({bar.body_size:.2f})",
                timestamp=bar.open_time,
            ))

        for p in patterns:
            self._detected_patterns.append(p)
            if p.strength > 0.6:
                log.info(f"🔎 FOOTPRINT PATTERN: {p.pattern_type} {p.direction} @ ${p.price:.2f} | {p.description}")

    # ─── Consultas ────────────────────────────────────────────────────────────

    def get_recent_bars(self, n: int = 20) -> List[FootprintBar]:
        bars = self._bars.copy()
        if self._current_bar:
            bars.append(self._current_bar)
        return bars[-n:]

    def get_stacked_imbalances_near_price(
        self, price: float, tolerance_pct: float = 0.02
    ) -> List[StackedImbalance]:
        """Retorna stacked imbalances cercanos al precio actual."""
        tolerance = price * tolerance_pct
        recent = self._stacked_imbalances[-50:]
        return [
            si for si in recent
            if abs((si.price_high + si.price_low) / 2 - price) <= tolerance
        ]

    def get_recent_patterns(self, n: int = 10) -> List[FootprintPattern]:
        return self._detected_patterns[-n:]

    def get_current_bar_summary(self) -> Optional[dict]:
        bar = self._current_bar
        if not bar:
            return None
        return {
            "bar_id": bar.bar_id,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "total_volume": bar.total_volume,
            "delta": bar.delta,
            "delta_pct": bar.delta_pct,
            "num_price_levels": len(bar.cells),
            "poc": bar.poc_price,
            "bid_imbalances": bar.stacked_bid_imbalances,
            "ask_imbalances": bar.stacked_ask_imbalances,
        }

    def print_current_bar(self) -> None:
        """Imprime el footprint de la barra actual en la consola."""
        bar = self._current_bar
        if not bar or not bar.cells:
            print("Sin datos de barra actual.")
            return

        print(f"\n{'─'*55}")
        print(f"  FOOTPRINT BAR #{bar.bar_id} | O:{bar.open:.1f} H:{bar.high:.1f} L:{bar.low:.1f} C:{bar.close:.1f}")
        print(f"  Vol:{bar.total_volume:.2f} | Δ:{bar.delta:+.2f} ({bar.delta_pct:+.1%})")
        print(f"{'─'*55}")
        print(f"  {'Price':>10}  {'Buy':>8}  {'Sell':>8}  {'Delta':>8}  {'Note':>10}")
        print(f"{'─'*55}")

        for cell in bar.get_sorted_cells(ascending=False):
            note = ""
            if cell.has_bid_imbalance:
                note = "▲ BID IMB"
            elif cell.has_ask_imbalance:
                note = "▼ ASK IMB"
            if cell.price == bar.poc_price:
                note += " ★POC"

            print(
                f"  ${cell.price:>9.1f}  {cell.buy_volume:>8.2f}  {cell.sell_volume:>8.2f}  "
                f"{cell.delta:>+8.2f}  {note:>10}"
            )
        print(f"{'─'*55}\n")
