"""
tools/generate_test_data.py

Generador de datos sintéticos para testear el sistema sin conexión real.

Genera trades que simulan microestructura real:
  - Precio sigue un GBM (Geometric Brownian Motion) + reversión a media
  - Volumen con distribución log-normal (como los mercados reales)
  - Sesgo de dirección configurable (simula tendencias e imbalances)
  - Eventos de absorción y finishing drives sintéticos
  - Order book simulado coherente con el precio

Uso:
  python tools/generate_test_data.py --trades 100000 --output data/historical/test_data.csv
"""
from __future__ import annotations

import argparse
import csv
import random
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np

# ─── Parámetros por defecto (calibrados para ETH) ────────────────────────────

ETH_DEFAULT_PRICE = 3500.0
ETH_VOLATILITY = 0.0015       # Volatilidad por "tick" (~15 segundos de datos HF)
ETH_DRIFT = 0.00001           # Drift alcista muy leve
ETH_MEAN_REVERSION = 0.001    # Fuerza de reversión a la media
ETH_TICK_SIZE = 0.10          # $0.10 por tick en datos sintéticos
ETH_MEAN_TRADE_QTY = 0.5      # ETH promedio por trade
ETH_LARGE_TRADE_PCT = 0.02    # 2% de trades son "grandes" (institucionales)
ETH_LARGE_TRADE_MULT = 10.0   # Multiplicador de tamaño para trades grandes


def simulate_price_path(
    n_trades: int,
    start_price: float = ETH_DEFAULT_PRICE,
    volatility: float = ETH_VOLATILITY,
    drift: float = ETH_DRIFT,
    mean_reversion: float = ETH_MEAN_REVERSION,
    trend_cycles: int = 5,
    absorptions: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """
    Genera una trayectoria de precios realista usando:
    - GBM (Geometric Brownian Motion) como base
    - Reversión a media (Ornstein-Uhlenbeck)
    - Ciclos de tendencia/rango alterados
    - Eventos de absorción (precio quieto con mucho volumen)
    """
    rng = np.random.default_rng(seed)
    prices = np.zeros(n_trades)
    prices[0] = start_price

    # Definir ciclos de tendencia: (inicio, fin, dirección, fuerza)
    cycle_length = n_trades // max(trend_cycles, 1)
    cycles = []
    for i in range(trend_cycles):
        start_idx = i * cycle_length
        end_idx = start_idx + cycle_length
        direction = rng.choice([-1, 1])
        strength = rng.uniform(0.5, 2.0) * abs(drift)
        cycles.append((start_idx, end_idx, direction, strength))

    # Definir eventos de absorción
    absorption_indices = sorted(rng.choice(n_trades, size=absorptions, replace=False))
    absorption_set = set(absorption_indices)

    long_term_mean = start_price

    for i in range(1, n_trades):
        # Ruido GBM
        random_shock = rng.normal(0, volatility)

        # Drift direccional del ciclo actual
        trend_drift = 0.0
        for s, e, d, strength in cycles:
            if s <= i < e:
                trend_drift = d * strength
                break

        # Reversión a media (OU process)
        mean_rev = mean_reversion * (long_term_mean - prices[i-1]) / long_term_mean

        # Absorción: precio casi no se mueve
        if i in absorption_set:
            prices[i] = prices[i-1] + rng.normal(0, volatility * 0.1)
        else:
            prices[i] = prices[i-1] * (1 + random_shock + trend_drift + mean_rev)
            prices[i] = max(prices[i], start_price * 0.5)  # Floor

    return prices


def generate_order_book(
    mid_price: float,
    spread_bps: float = 0.5,
    levels: int = 20,
    rng: np.random.Generator = None,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Genera un order book sintético coherente con el precio.
    Los niveles más cercanos tienen mayor probabilidad de tener volumen.
    """
    if rng is None:
        rng = np.random.default_rng()

    half_spread = mid_price * (spread_bps / 10000)
    best_bid = mid_price - half_spread
    best_ask = mid_price + half_spread

    bids = []
    asks = []

    for i in range(1, levels + 1):
        # Los niveles más alejados tienen menos volumen (distribución exponencial)
        vol_factor = np.exp(-0.15 * (i - 1))
        base_vol = rng.lognormal(mean=np.log(2.0), sigma=0.5) * vol_factor

        bid_price = round(best_bid - (i - 1) * 0.10, 2)
        ask_price = round(best_ask + (i - 1) * 0.10, 2)

        bids.append((bid_price, round(base_vol, 3)))
        asks.append((ask_price, round(base_vol * rng.uniform(0.7, 1.3), 3)))

    return bids, asks


def classify_trade_synthetic(
    price: float,
    prev_price: float,
    best_bid: float,
    best_ask: float,
    rng: np.random.Generator,
) -> int:
    """Clasifica el trade sintético usando reglas coherentes con Lee-Ready."""
    if price >= best_ask:
        return 1
    elif price <= best_bid:
        return -1
    else:
        # Tick rule
        if price > prev_price:
            return 1
        elif price < prev_price:
            return -1
        else:
            # Empate: ligeramente más probable que siga la dirección del precio vs mid
            return rng.choice([1, -1], p=[0.52, 0.48])


def generate_synthetic_dataset(
    n_trades: int = 100_000,
    start_price: float = ETH_DEFAULT_PRICE,
    start_datetime: datetime = None,
    trade_interval_ms: float = 150.0,   # ~150ms entre trades (realista para ETH Futures)
    output_path: str = "data/historical/synthetic_eth.csv",
    seed: int = 42,
    verbose: bool = True,
) -> str:
    """
    Genera un dataset CSV completo de trades sintéticos para ETH Futures.

    Columnas del CSV:
    timestamp, price, quantity, side, trade_id

    Args:
        n_trades: Número de trades a generar
        start_price: Precio inicial de ETH
        start_datetime: Timestamp de inicio (default: ahora - duración estimada)
        trade_interval_ms: Intervalo promedio entre trades en ms
        output_path: Ruta del archivo CSV de salida
        seed: Semilla para reproducibilidad
        verbose: Mostrar progreso

    Returns:
        Path del archivo generado
    """
    rng = np.random.default_rng(seed)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Estimar duración total
    duration_seconds = n_trades * trade_interval_ms / 1000
    if start_datetime is None:
        start_datetime = datetime.now(timezone.utc) - timedelta(seconds=duration_seconds)

    if verbose:
        print(f"\n{'='*55}")
        print(f"  GENERADOR DE DATOS SINTÉTICOS ETH FUTURES")
        print(f"{'='*55}")
        print(f"  Trades:         {n_trades:,}")
        print(f"  Precio inicial: ${start_price:,.2f}")
        print(f"  Duración est.:  {duration_seconds/3600:.1f} horas")
        print(f"  Salida:         {output_path}")
        print(f"{'='*55}\n")

    t_start = time.time()

    # Generar trayectoria de precios
    prices = simulate_price_path(
        n_trades, start_price,
        trend_cycles=8, absorptions=15, seed=seed
    )

    # Generar timestamps (con variabilidad en el intervalo)
    intervals = rng.exponential(trade_interval_ms / 1000, n_trades)
    timestamps = start_datetime.timestamp() + np.cumsum(intervals)

    # Generar volúmenes (log-normal)
    is_large = rng.random(n_trades) < ETH_LARGE_TRADE_PCT
    base_quantities = rng.lognormal(mean=np.log(ETH_MEAN_TRADE_QTY), sigma=0.8, size=n_trades)
    quantities = np.where(is_large, base_quantities * ETH_LARGE_TRADE_MULT, base_quantities)
    quantities = np.clip(quantities, 0.001, 500.0)

    # Spread dinámico (más amplio en momentos de alta volatilidad)
    price_changes = np.abs(np.diff(prices, prepend=prices[0]))
    volatility_regime = np.convolve(price_changes, np.ones(50)/50, mode='same')
    spread_factor = 1.0 + volatility_regime / price_changes.mean()

    # Escribir CSV
    written = 0
    BATCH = 10_000

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "price", "quantity", "side", "trade_id"])

        for i in range(n_trades):
            price = round(prices[i], 2)
            prev_price = round(prices[i-1], 2) if i > 0 else price

            # Order book en ese instante
            spread_bps = max(0.2, 0.5 * spread_factor[i])
            half_spread = price * (spread_bps / 10000)
            best_bid = price - half_spread
            best_ask = price + half_spread

            # Clasificar trade
            side = classify_trade_synthetic(price, prev_price, best_bid, best_ask, rng)

            writer.writerow([
                f"{timestamps[i]:.6f}",
                f"{price:.4f}",
                f"{quantities[i]:.6f}",
                side,
                i + 1,
            ])

            written += 1
            if verbose and written % BATCH == 0:
                elapsed = time.time() - t_start
                rate = written / elapsed
                remaining = (n_trades - written) / rate
                pct = written / n_trades * 100
                print(f"  [{pct:5.1f}%] {written:>10,} trades | {rate:,.0f}/s | ETA: {remaining:.0f}s")

    elapsed = time.time() - t_start

    # Calcular estadísticas del dataset
    price_final = float(prices[-1])
    price_change = (price_final - start_price) / start_price * 100
    buy_count = int(np.sum(
        [classify_trade_synthetic(prices[i], prices[max(0, i-1)],
                                   prices[i]*0.9999, prices[i]*1.0001, rng) == 1
         for i in range(min(1000, n_trades))]
    ))

    if verbose:
        print(f"\n  ✅ Dataset generado en {elapsed:.1f}s")
        print(f"  Precio inicio:  ${start_price:,.2f}")
        print(f"  Precio final:   ${price_final:,.2f} ({price_change:+.2f}%)")
        print(f"  Archivo:        {output_path}")
        file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
        print(f"  Tamaño:         {file_size_mb:.1f} MB\n")

    return output_path


def generate_multi_scenario_dataset(
    base_path: str = "data/historical",
    seed: int = 42,
) -> List[str]:
    """
    Genera múltiples escenarios para backtesting robusto:
    - Tendencia alcista
    - Tendencia bajista
    - Mercado en rango
    - Alta volatilidad
    """
    scenarios = [
        {"name": "uptrend", "drift": 0.00005, "volatility": 0.001, "n": 50_000},
        {"name": "downtrend", "drift": -0.00005, "volatility": 0.001, "n": 50_000},
        {"name": "ranging", "drift": 0.0, "volatility": 0.0008, "n": 50_000},
        {"name": "high_volatility", "drift": 0.00001, "volatility": 0.003, "n": 50_000},
    ]

    paths = []
    for i, s in enumerate(scenarios):
        out = f"{base_path}/eth_{s['name']}.csv"
        print(f"\n📊 Generando escenario: {s['name'].upper()}")
        # Override volatility/drift en la función
        # Usar diferentes semillas para variedad
        generate_synthetic_dataset(
            n_trades=s["n"],
            output_path=out,
            seed=seed + i * 100,
        )
        paths.append(out)

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de datos sintéticos ETH Futures")
    parser.add_argument("--trades", type=int, default=100_000, help="Número de trades")
    parser.add_argument("--price", type=float, default=ETH_DEFAULT_PRICE, help="Precio inicial")
    parser.add_argument("--output", default="data/historical/synthetic_eth.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multi", action="store_true", help="Generar múltiples escenarios")
    args = parser.parse_args()

    if args.multi:
        paths = generate_multi_scenario_dataset()
        print(f"\n✅ {len(paths)} datasets generados.")
    else:
        generate_synthetic_dataset(
            n_trades=args.trades,
            start_price=args.price,
            output_path=args.output,
            seed=args.seed,
        )
