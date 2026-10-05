"""
websocket_client.py — Cliente WebSocket asíncrono para Binance (ETH).

Suscribe simultáneamente a:
  - ethusdt@aggTrade   → trades en tiempo real
  - ethusdt@depth20    → top 20 niveles del order book

Diseño:
- asyncio + websockets
- Reconexión automática con backoff exponencial
- Parser separado para cada stream
- Inyecta datos directamente en DataBuffer
"""

import asyncio
import json
import logging
import time
from typing import Optional, Callable

try:
    import websockets
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
except ImportError:
    raise ImportError("Instala websockets: pip install websockets")

from data.data_buffer import DataBuffer, Trade, OrderBook
from utils.config import CONFIG

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    Cliente WebSocket para Binance Futures/Spot ETH.

    Características:
    - Reconexión automática con backoff exponencial (máx 60s)
    - Callback opcional para eventos de señal externos
    - Estadísticas de latencia en tiempo real
    - Soporte para modo combined stream (/stream?streams=)
    """

    def __init__(
        self,
        buffer: DataBuffer,
        on_trade_callback: Optional[Callable] = None,
        on_book_callback: Optional[Callable] = None,
    ):
        self.buffer = buffer
        self.on_trade_callback = on_trade_callback
        self.on_book_callback = on_book_callback
        self._running = False
        self._reconnect_count = 0
        self._total_messages = 0
        self._trade_messages = 0
        self._book_messages = 0
        self._last_latency_ms: float = 0.0

    # ─── URL de conexión ───────────────────────────────────────────────────────

    def _build_url(self) -> str:
        streams = "/".join(CONFIG.exchange.streams)
        return f"{CONFIG.exchange.base_url}?streams={streams}"

    # ─── Parsers ───────────────────────────────────────────────────────────────

    def _parse_agg_trade(self, data: dict) -> Optional[Trade]:
        """
        Parsea un aggTrade de Binance.

        Campos clave:
          T  → timestamp en ms
          p  → precio
          q  → cantidad
          m  → is_buyer_maker (True = seller agresivo)
          a  → aggTradeId
        """
        try:
            now_ms = time.time() * 1000
            event_ms = float(data.get("T", now_ms))
            self._last_latency_ms = now_ms - event_ms

            return Trade(
                timestamp=event_ms / 1000.0,
                price=float(data["p"]),
                quantity=float(data["q"]),
                is_buyer_maker=bool(data["m"]),
                trade_id=int(data.get("a", 0)),
            )
        except (KeyError, ValueError) as e:
            logger.warning(f"Error parseando aggTrade: {e} | data={data}")
            return None

    def _parse_depth(self, data: dict) -> Optional[OrderBook]:
        """
        Parsea el order book partial depth (depth20@100ms).

        Binance Futures usa claves cortas:
          "b" → bids [[price, qty], ...]   ordenado desc
          "a" → asks [[price, qty], ...]   ordenado asc

        Binance Spot usa claves largas:
          "bids" / "asks"

        Soportamos ambos formatos.
        """
        try:
            # Binance Futures: claves "b" y "a"
            # Binance Spot:    claves "bids" y "asks"
            raw_bids = data.get("b") or data.get("bids") or []
            raw_asks = data.get("a") or data.get("asks") or []

            bids = [(float(p), float(q)) for p, q in raw_bids if float(q) > 0]
            asks = [(float(p), float(q)) for p, q in raw_asks if float(q) > 0]

            if not bids and not asks:
                return None   # mensaje vacio, ignorar

            # Ordenar: bids descendente, asks ascendente
            bids.sort(key=lambda x: x[0], reverse=True)
            asks.sort(key=lambda x: x[0])

            return OrderBook(
                timestamp=time.time(),
                bids=bids,
                asks=asks,
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Error parseando depth: {e} | keys={list(data.keys())}")
            return None

    # ─── Dispatcher ───────────────────────────────────────────────────────────

    def _dispatch(self, message: dict) -> None:
        """Enruta el mensaje al parser correcto según el stream."""
        stream_name = message.get("stream", "")
        data = message.get("data", {})
        event_type = data.get("e", "")

        self._total_messages += 1

        if "aggTrade" in stream_name or event_type == "aggTrade":
            trade = self._parse_agg_trade(data)
            if trade:
                self.buffer.add_trade(trade)
                self._trade_messages += 1
                if self.on_trade_callback:
                    self.on_trade_callback(trade)

        elif "depth" in stream_name:
            book = self._parse_depth(data)
            if book:
                self.buffer.update_orderbook(book)
                self._book_messages += 1
                if self._book_messages == 1:
                    logger.info(
                        f"📖 Primer libro recibido: "
                        f"best_bid={book.best_bid:.2f} "
                        f"best_ask={book.best_ask:.2f} "
                        f"spread={book.spread:.3f} "
                        f"niveles={len(book.bids)}b/{len(book.asks)}a"
                    )
                if self.on_book_callback:
                    self.on_book_callback(book)

    # ─── Bucle principal ──────────────────────────────────────────────────────

    async def _listen(self) -> None:
        url = self._build_url()
        logger.info(f"Conectando a: {url}")

        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
        ) as ws:
            self._reconnect_count = 0
            logger.info("WebSocket conectado. Escuchando streams ETH...")

            async for raw_msg in ws:
                if not self._running:
                    break
                try:
                    message = json.loads(raw_msg)
                    self._dispatch(message)
                except json.JSONDecodeError:
                    logger.warning(f"Mensaje JSON inválido: {raw_msg[:100]}")

    async def run(self) -> None:
        """
        Bucle principal con reconexión automática y backoff exponencial.
        Máximo espera CONFIG.exchange.reconnect_delay * 2^n (cap 60s).
        """
        self._running = True
        max_attempts = CONFIG.exchange.max_reconnect_attempts
        base_delay = CONFIG.exchange.reconnect_delay

        while self._running and self._reconnect_count < max_attempts:
            try:
                await self._listen()
            except (ConnectionClosedError, ConnectionClosedOK) as e:
                if not self._running:
                    break
                self._reconnect_count += 1
                delay = min(base_delay * (2 ** self._reconnect_count), 60.0)
                logger.warning(
                    f"WebSocket cerrado ({e}). "
                    f"Reconectando en {delay:.1f}s "
                    f"(intento {self._reconnect_count}/{max_attempts})"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                if not self._running:
                    break
                self._reconnect_count += 1
                delay = min(base_delay * (2 ** self._reconnect_count), 60.0)
                logger.error(
                    f"Error inesperado en WebSocket: {e}. "
                    f"Reintentando en {delay:.1f}s"
                )
                await asyncio.sleep(delay)

        if self._reconnect_count >= max_attempts:
            logger.critical(
                f"Se alcanzó el máximo de reintentos ({max_attempts}). "
                "Sistema detenido."
            )

    def stop(self) -> None:
        self._running = False
        logger.info("WebSocket client detenido.")

    # ─── Estadísticas ─────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "total_messages": self._total_messages,
            "trade_messages": self._trade_messages,
            "book_messages": self._book_messages,
            "reconnect_count": self._reconnect_count,
            "last_latency_ms": round(self._last_latency_ms, 2),
        }
