"""
data/data_collector.py

Recolector de datos históricos en tiempo real.
Guarda trades del WebSocket a CSV para backtesting posterior.

Características:
  - Escritura en buffer (evita I/O en cada trade)
  - Rotación de archivos por sesión/día
  - Compresión automática de archivos viejos (gzip)
  - Estadísticas de recolección
  - Carga de datos históricos desde Binance REST API
"""
from __future__ import annotations

import asyncio
import csv
import gzip
import io
import json
import shutil
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiohttp

from data.data_buffer import Trade
from utils.logger import get_logger

log = get_logger("DataCollector")

BINANCE_FUTURES_REST = "https://fapi.binance.com"


class DataCollector:
    """
    Persiste trades del stream WebSocket a archivos CSV.

    Arquitectura:
    - Recibe trades desde el DataBuffer
    - Los acumula en una deque interna (write_buffer)
    - Cada N trades o M segundos, los escribe al CSV
    - Evita I/O en el hot path del procesamiento
    """

    def __init__(
        self,
        output_dir: str = "data/historical",
        symbol: str = "ETHUSDT",
        write_buffer_size: int = 1000,
        flush_interval_seconds: float = 5.0,
        compress_after_days: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol.upper()
        self.write_buffer_size = write_buffer_size
        self.flush_interval = flush_interval_seconds
        self.compress_after_days = compress_after_days

        self._write_buffer: deque[Trade] = deque()
        self._current_file: Optional[Path] = None
        self._csv_writer = None
        self._file_handle = None
        self._session_start = datetime.now(timezone.utc)
        self._total_written = 0
        self._last_flush = time.time()
        self._running = False
        self._last_trade_id = 0

        # Inicializar archivo de sesión
        self._open_session_file()

    def _session_filename(self) -> Path:
        ts = self._session_start.strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{self.symbol}_{ts}.csv"

    def _open_session_file(self) -> None:
        filepath = self._session_filename()
        self._current_file = filepath
        self._file_handle = open(filepath, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._file_handle)
        # Cabecera
        self._csv_writer.writerow(["timestamp", "price", "quantity", "side", "trade_id"])
        self._file_handle.flush()
        log.info(f"📂 Archivo de datos abierto: {filepath}")

    def add_trade(self, trade: Trade) -> None:
        """Agrega un trade al buffer de escritura (non-blocking)."""
        self._write_buffer.append(trade)
        # Escribir si el buffer está lleno
        if len(self._write_buffer) >= self.write_buffer_size:
            self._flush_sync()

    def _flush_sync(self) -> None:
        """Escribe el buffer al CSV de forma síncrona."""
        if not self._write_buffer or not self._csv_writer:
            return

        rows_written = 0
        while self._write_buffer:
            t = self._write_buffer.popleft()
            self._csv_writer.writerow([
                f"{t.timestamp:.6f}",
                f"{t.price:.4f}",
                f"{t.quantity:.6f}",
                t.side,
                t.trade_id,
            ])
            rows_written += 1

        self._file_handle.flush()
        self._total_written += rows_written
        self._last_flush = time.time()

    async def flush(self) -> None:
        """Versión async del flush."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._flush_sync)

    async def run_flush_loop(self) -> None:
        """
        Task async: hace flush periódico cada N segundos.
        Llamar como asyncio.create_task().
        """
        self._running = True
        log.info("💾 DataCollector flush loop iniciado.")

        while self._running:
            await asyncio.sleep(self.flush_interval)
            elapsed_since_flush = time.time() - self._last_flush
            if self._write_buffer and elapsed_since_flush >= self.flush_interval:
                await self.flush()
                log.debug(f"Flush: {self._total_written:,} trades guardados.")

    async def stop(self) -> None:
        self._running = False
        self._flush_sync()  # Flush final
        if self._file_handle:
            self._file_handle.close()
            log.info(f"✅ DataCollector cerrado. Total guardado: {self._total_written:,} trades")
        # Comprimir el archivo de sesión actual
        if self._current_file and self._current_file.exists():
            await self._compress_file(self._current_file)

    async def _compress_file(self, filepath: Path) -> None:
        """Comprime un archivo CSV a .csv.gz."""
        gz_path = filepath.with_suffix(".csv.gz")
        loop = asyncio.get_event_loop()

        def _do_compress():
            with open(filepath, "rb") as f_in:
                with gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            filepath.unlink()  # Eliminar original
            log.info(f"🗜 Comprimido: {gz_path.name} ({gz_path.stat().st_size / 1024:.1f} KB)")

        await loop.run_in_executor(None, _do_compress)

    def get_stats(self) -> dict:
        return {
            "total_written": self._total_written,
            "buffer_pending": len(self._write_buffer),
            "current_file": str(self._current_file),
            "session_start": self._session_start.isoformat(),
        }


# ─── Descarga de datos históricos desde Binance REST ─────────────────────────

class BinanceHistoricalLoader:
    """
    Descarga datos históricos de aggTrades desde Binance Futures REST API.
    Útil para construir datasets de backtesting sin necesidad de haber
    tenido el WebSocket activo.

    Binance permite descargar hasta 1000 aggTrades por request (máximo).
    Para períodos largos, itera usando el campo 'fromId'.
    """

    BASE = "https://fapi.binance.com"

    def __init__(self, symbol: str = "ETHUSDT"):
        self.symbol = symbol.upper()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _get_agg_trades(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        from_id: Optional[int] = None,
        limit: int = 1000,
    ) -> List[dict]:
        params = {"symbol": self.symbol, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        if from_id:
            params["fromId"] = from_id

        session = await self._get_session()
        url = f"{self.BASE}/fapi/v1/aggTrades"
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ValueError(f"HTTP {resp.status}: {text}")
            return await resp.json()

    def _parse_agg_trade(self, trade: dict) -> dict:
        """Parsea un aggTrade de Binance al formato estándar del sistema."""
        price = float(trade["p"])
        qty = float(trade["q"])
        is_buyer_maker = bool(trade["m"])
        # is_buyer_maker=True → venta agresiva (taker vendió)
        side = -1 if is_buyer_maker else 1
        return {
            "timestamp": trade["T"] / 1000.0,
            "price": price,
            "quantity": qty,
            "side": side,
            "trade_id": trade["a"],
        }

    async def download_range(
        self,
        start_dt: datetime,
        end_dt: datetime,
        output_path: str,
        batch_size: int = 1000,
    ) -> int:
        """
        Descarga todos los aggTrades entre dos fechas y los guarda en CSV.

        Args:
            start_dt: Datetime de inicio (timezone-aware recomendado)
            end_dt: Datetime de fin
            output_path: Path del archivo CSV de salida
            batch_size: Trades por request (máx 1000)

        Returns:
            Total de trades descargados
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        total = 0

        log.info(f"📥 Descargando {self.symbol} aggTrades: {start_dt} → {end_dt}")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "price", "quantity", "side", "trade_id"])

            current_start = start_ms
            last_id = None
            batch_count = 0

            while current_start < end_ms:
                try:
                    # Usar fromId si disponemos de él (más eficiente)
                    if last_id:
                        raw = await self._get_agg_trades(from_id=last_id + 1, limit=batch_size)
                    else:
                        raw = await self._get_agg_trades(
                            start_time=current_start,
                            end_time=min(current_start + 3_600_000, end_ms),  # Max 1h por request
                            limit=batch_size,
                        )

                    if not raw:
                        break

                    # Filtrar trades fuera del rango
                    raw = [t for t in raw if t["T"] <= end_ms]
                    if not raw:
                        break

                    for trade in raw:
                        parsed = self._parse_agg_trade(trade)
                        writer.writerow([
                            f"{parsed['timestamp']:.6f}",
                            f"{parsed['price']:.4f}",
                            f"{parsed['quantity']:.6f}",
                            parsed["side"],
                            parsed["trade_id"],
                        ])

                    total += len(raw)
                    batch_count += 1
                    last_trade = raw[-1]
                    last_id = last_trade["a"]
                    current_start = last_trade["T"] + 1

                    if batch_count % 10 == 0:
                        pct = (current_start - start_ms) / (end_ms - start_ms) * 100
                        log.info(f"  Progreso: {pct:.1f}% | Trades: {total:,}")

                    # Rate limit: Binance permite ~1200 req/min en endpoint público
                    await asyncio.sleep(0.05)

                except Exception as e:
                    log.error(f"Error en batch {batch_count}: {e}")
                    await asyncio.sleep(2.0)

        log.info(f"✅ Descarga completa: {total:,} trades → {output_path}")
        return total

    async def download_recent(
        self,
        hours: int = 24,
        output_path: str = "data/historical/recent.csv",
    ) -> int:
        """Descarga las últimas N horas de datos."""
        from datetime import timedelta
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        return await self.download_range(start, end, output_path)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# ─── Script de descarga standalone ───────────────────────────────────────────

async def download_backtest_data(
    symbol: str = "ETHUSDT",
    hours: int = 72,
    output_path: str = "data/historical/eth_backtest.csv",
) -> str:
    """
    Función utilitaria para descargar datos de backtest.
    Usar desde CLI o scripts de setup.
    """
    loader = BinanceHistoricalLoader(symbol)
    try:
        n = await loader.download_recent(hours=hours, output_path=output_path)
        log.info(f"Dataset listo: {n:,} trades en {output_path}")
        return output_path
    finally:
        await loader.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Descarga datos históricos de Binance Futures")
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--hours", type=int, default=72, help="Horas hacia atrás")
    parser.add_argument("--output", default="data/historical/eth_backtest.csv")
    args = parser.parse_args()

    asyncio.run(download_backtest_data(args.symbol, args.hours, args.output))
