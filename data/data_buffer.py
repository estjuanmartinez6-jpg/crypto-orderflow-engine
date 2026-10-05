"""
data_buffer.py — Buffer thread-safe de trades y order book en tiempo real.

Diseño:
- Usa collections.deque para O(1) en append/pop
- Lock para acceso concurrente (asyncio + threads)
- Snapshot inmutable del order book para cálculos sin race conditions
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class Trade:
    """Estructura de un trade individual."""
    timestamp: float          # Unix timestamp en segundos
    price: float
    quantity: float
    is_buyer_maker: bool      # True = vendedor agresivo, False = comprador agresivo
    trade_id: int = 0

    @property
    def side(self) -> int:
        """
        +1 = compra agresiva (market buy)
        -1 = venta agresiva (market sell)
        Binance: is_buyer_maker=True significa que el BUYER estaba en el libro
        → por tanto el SELLER fue el agresivo → venta.
        """
        return -1 if self.is_buyer_maker else +1

    @property
    def signed_qty(self) -> float:
        return self.quantity * self.side


@dataclass
class OrderBook:
    """Snapshot del order book en un instante."""
    timestamp: float
    bids: List[Tuple[float, float]] = field(default_factory=list)  # (price, qty)
    asks: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    def bid_volume(self, n_levels: int = 10) -> float:
        return sum(qty for _, qty in self.bids[:n_levels])

    def ask_volume(self, n_levels: int = 10) -> float:
        return sum(qty for _, qty in self.asks[:n_levels])

    def imbalance(self, n_levels: int = 10) -> float:
        """
        Imbalance del libro: rango [-1, +1]
        +1 = presión compradora total
        -1 = presión vendedora total
        """
        bv = self.bid_volume(n_levels)
        av = self.ask_volume(n_levels)
        total = bv + av
        if total == 0:
            return 0.0
        return (bv - av) / total

    def weighted_mid(self, n_levels: int = 5) -> Optional[float]:
        """Precio medio ponderado por volumen (más preciso que mid)."""
        if not self.bids or not self.asks:
            return None
        bv = self.bid_volume(n_levels)
        av = self.ask_volume(n_levels)
        if bv + av == 0:
            return self.mid_price
        return (self.best_ask * bv + self.best_bid * av) / (bv + av)


class DataBuffer:
    """
    Buffer central del sistema.

    Thread-safe mediante RLock.
    Provee acceso eficiente a:
    - Trades recientes (deque limitado)
    - Order book actual
    - Estadísticas rápidas
    """

    def __init__(self, max_trades: int = 50_000):
        self._max_trades = max_trades
        self._trades: deque[Trade] = deque(maxlen=max_trades)
        self._book: Optional[OrderBook] = None
        self._lock = threading.RLock()
        self._trade_count_total: int = 0
        self._last_price: Optional[float] = None

    # ─── Trades ────────────────────────────────────────────────────────────────

    def add_trade(self, trade: Trade) -> None:
        with self._lock:
            self._trades.append(trade)
            self._trade_count_total += 1
            self._last_price = trade.price

    def add_trades_batch(self, trades: List[Trade]) -> None:
        with self._lock:
            for t in trades:
                self._trades.append(t)
            self._trade_count_total += len(trades)
            if trades:
                self._last_price = trades[-1].price

    def get_recent_trades(self, n: Optional[int] = None) -> List[Trade]:
        """Devuelve los últimos n trades (copia segura)."""
        with self._lock:
            if n is None:
                return list(self._trades)
            return list(self._trades)[-n:]

    def get_trades_since(self, timestamp: float) -> List[Trade]:
        """Devuelve trades desde un timestamp dado."""
        with self._lock:
            return [t for t in self._trades if t.timestamp >= timestamp]

    def get_trades_in_window(self, seconds: int) -> List[Trade]:
        """Devuelve trades de los últimos N segundos."""
        cutoff = time.time() - seconds
        return self.get_trades_since(cutoff)

    # ─── Order Book ────────────────────────────────────────────────────────────

    def update_orderbook(self, book: OrderBook) -> None:
        with self._lock:
            self._book = book

    def get_orderbook(self) -> Optional[OrderBook]:
        with self._lock:
            return self._book

    # ─── Propiedades rápidas ───────────────────────────────────────────────────

    @property
    def last_price(self) -> Optional[float]:
        with self._lock:
            return self._last_price

    @property
    def trade_count(self) -> int:
        with self._lock:
            return len(self._trades)

    @property
    def total_trades_received(self) -> int:
        with self._lock:
            return self._trade_count_total

    def to_numpy(self, n: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Convierte los trades a arrays NumPy para procesamiento vectorizado.
        Retorna dict: {timestamps, prices, quantities, sides}
        """
        trades = self.get_recent_trades(n)
        if not trades:
            return {
                "timestamps": np.array([]),
                "prices": np.array([]),
                "quantities": np.array([]),
                "sides": np.array([]),
            }
        return {
            "timestamps": np.array([t.timestamp for t in trades]),
            "prices":     np.array([t.price for t in trades]),
            "quantities": np.array([t.quantity for t in trades]),
            "sides":      np.array([t.side for t in trades]),
        }

    def size(self) -> int:
        """Alias de trade_count para compatibilidad."""
        return self.trade_count

    def get_stats(self) -> dict:
        """Estadísticas básicas del buffer."""
        with self._lock:
            return {
                'trades_in_buffer': len(self._trades),
                'total_trades_received': self._trade_count_total,
                'last_price': self._last_price,
                'buffer_fill_pct': len(self._trades) / self._max_trades * 100,
            }

    def __len__(self) -> int:
        return self.trade_count
