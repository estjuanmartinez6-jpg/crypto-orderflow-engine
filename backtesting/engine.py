"""
engine.py — Motor de backtesting realista para ETH Order Flow.

Cambios respecto a la versión anterior:

  BLOQUE 1 — Estrategia delegada
    run_from_csv() ya NO toma decisiones. Acepta un signal_fn opcional
    (callable que recibe bars + índice y retorna un dict de señal o None).
    Si no se pasa, usa una lógica simplificada interna de dos capas
    (contexto VP + flujo delta) que NO incluye absorción. Esto es honesto:
    la absorción real no puede reconstruirse fielmente desde barras de 10s.

  BLOQUE 2 — Barras con métricas de flujo
    Cada barra incluye delta, delta_pct, buy_vol, sell_vol. Se usan high/low
    para fill realista. Absorción, footprint y divergencias desactivadas.

  BLOQUE 3 — VP_WINDOW y tick correctos
    VP_WINDOW = 360 barras (~1 hora a 10s). tick = TICK_SIZE * 2.

  BLOQUE 4 — Umbral de delta dinámico
    Se calcula np.percentile(all_deltas, 70) antes del loop. Adaptativo
    al régimen de volatilidad del período.

  BLOQUE 5 — Ejecución realista
    - Slippage dinámico: base * (1 + volatility * 10)
    - Spread en precio de entrada
    - Cierre parcial al TP1: 50% cierra, 50% sigue hacia TP2

  BLOQUE 6 — Log de trades
    Cada trade imprime dirección, precios, delta y contexto.
"""

import time
import logging
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Callable
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constantes de ejecución
# ──────────────────────────────────────────────────────────────────────────────

COMMISSION_PCT  = 0.0004   # taker fee real de Binance Futures (0.04%)
BASE_SLIPPAGE   = 0.0003   # base; se escala por volatilidad de la barra
SPREAD_PCT      = 0.0002   # spread mid-to-touch aproximado en ETH perp

TP1_CLOSE_RATIO = 0.5      # fracción de la posición que se cierra en TP1


# ──────────────────────────────────────────────────────────────────────────────
# Estructuras de datos
# ──────────────────────────────────────────────────────────────────────────────

class TradeStatus(Enum):
    OPEN           = "OPEN"
    PARTIAL_TP1    = "PARTIAL_TP1"     # TP1 alcanzado, posición reducida
    CLOSED_TP2     = "CLOSED_TP2"
    CLOSED_SL      = "CLOSED_SL"
    CLOSED_EXPIRY  = "CLOSED_EXPIRY"


@dataclass
class BacktestTrade:
    """
    Trade con soporte para cierre parcial en TP1.

    Ciclo de vida:
      OPEN → (TP1 hit) → PARTIAL_TP1 → (TP2 o SL hit) → CLOSED_TP2 / CLOSED_SL
      OPEN → (SL hit antes de TP1) → CLOSED_SL
    """
    trade_id:       int
    direction:      str       # 'LONG' | 'SHORT'
    entry_price:    float
    stop_loss:      float
    take_profit_1:  float
    take_profit_2:  float
    size_eth:       float     # tamaño inicial

    entry_time:     float
    zone:           str       # zona del VP donde se generó
    delta_pct:      float     # delta% de la barra de entrada
    macro_bias:     str       # bias macro en el momento de entrada

    exit_price:     float = 0.0
    exit_time:      float = 0.0
    status:         TradeStatus = TradeStatus.OPEN

    # Cierre parcial al TP1
    tp1_hit:        bool  = False
    size_remaining: float = 0.0   # tamaño después de parcial
    partial_pnl:    float = 0.0   # PnL realizado en el cierre parcial

    # Totales al cierre final
    pnl_usd:        float = 0.0
    pnl_pct:        float = 0.0
    commission_paid: float = 0.0

    def __post_init__(self):
        self.size_remaining = self.size_eth

    # ── Cierre parcial en TP1 ────────────────────────────────────────────────

    def apply_partial_tp1(
        self,
        tp1_price: float,
        timestamp: float,
        slippage: float,
    ) -> None:
        """
        Cierra TP1_CLOSE_RATIO de la posición al alcanzar TP1.
        El resto continúa abierto hacia TP2.
        """
        closed_size = self.size_eth * TP1_CLOSE_RATIO
        self.size_remaining = self.size_eth * (1 - TP1_CLOSE_RATIO)
        self.tp1_hit = True
        self.status = TradeStatus.PARTIAL_TP1

        if self.direction == "LONG":
            eff_exit = tp1_price * (1 - slippage)
            gross = (eff_exit - self.entry_price) * closed_size
        else:
            eff_exit = tp1_price * (1 + slippage)
            gross = (self.entry_price - eff_exit) * closed_size

        comm = (self.entry_price + tp1_price) * closed_size * COMMISSION_PCT
        self.partial_pnl = gross - comm
        self.commission_paid += comm

    # ── Cierre final ─────────────────────────────────────────────────────────

    def close_final(
        self,
        exit_price: float,
        timestamp: float,
        status: TradeStatus,
        slippage: float,
    ) -> None:
        """Cierra el tamaño restante de la posición."""
        self.exit_price = exit_price
        self.exit_time  = timestamp
        self.status     = status

        remaining = self.size_remaining if self.tp1_hit else self.size_eth

        if self.direction == "LONG":
            eff_exit = exit_price * (1 - slippage)
            gross = (eff_exit - self.entry_price) * remaining
        else:
            eff_exit = exit_price * (1 + slippage)
            gross = (self.entry_price - eff_exit) * remaining

        comm = (self.entry_price + exit_price) * remaining * COMMISSION_PCT
        self.commission_paid += comm
        self.pnl_usd = self.partial_pnl + gross - comm
        self.pnl_pct = self.pnl_usd / (self.entry_price * self.size_eth + 1e-9)

    @property
    def duration_seconds(self) -> float:
        return self.exit_time - self.entry_time if self.exit_time > 0 else 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl_usd > 0

    def log_entry(self) -> str:
        return (
            f"[ENTRADA] #{self.trade_id} {self.direction} "
            f"@ {self.entry_price:.2f} | "
            f"SL={self.stop_loss:.2f} TP1={self.take_profit_1:.2f} "
            f"TP2={self.take_profit_2:.2f} | "
            f"Δ%={self.delta_pct*100:.1f}% | "
            f"Zona={self.zone} | Macro={self.macro_bias}"
        )

    def log_exit(self) -> str:
        sign = "+" if self.pnl_usd >= 0 else ""
        return (
            f"[SALIDA]  #{self.trade_id} {self.direction} "
            f"→ {self.status.value} @ {self.exit_price:.2f} | "
            f"PnL={sign}{self.pnl_usd:.2f} USD "
            f"({sign}{self.pnl_pct*100:.2f}%) | "
            f"Duración={self.duration_seconds:.0f}s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Resultados del backtest
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResults:
    trades:          List[BacktestTrade]
    initial_capital: float
    final_capital:   float
    symbol:          str
    period_start:    float
    period_end:      float

    total_trades:           int   = 0
    winning_trades:         int   = 0
    losing_trades:          int   = 0
    win_rate:               float = 0.0
    profit_factor:          float = 0.0
    total_pnl:              float = 0.0
    total_commission:       float = 0.0
    max_drawdown:           float = 0.0
    max_drawdown_pct:       float = 0.0
    sharpe_ratio:           float = 0.0
    sortino_ratio:          float = 0.0
    calmar_ratio:           float = 0.0
    avg_win:                float = 0.0
    avg_loss:               float = 0.0
    largest_win:            float = 0.0
    largest_loss:           float = 0.0
    consecutive_wins_max:   int   = 0
    consecutive_losses_max: int   = 0
    equity_curve:           List[float] = field(default_factory=list)

    def compute(self) -> None:
        closed = [
            t for t in self.trades
            if t.status not in (TradeStatus.OPEN, TradeStatus.PARTIAL_TP1)
        ]
        self.total_trades = len(closed)
        if not closed:
            return

        winners = [t for t in closed if t.is_winner]
        losers  = [t for t in closed if not t.is_winner]
        self.winning_trades   = len(winners)
        self.losing_trades    = len(losers)
        self.win_rate         = self.winning_trades / self.total_trades
        self.total_pnl        = sum(t.pnl_usd for t in closed)
        self.total_commission = sum(t.commission_paid for t in closed)

        gross_profit = sum(t.pnl_usd for t in winners)
        gross_loss   = abs(sum(t.pnl_usd for t in losers))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        self.avg_win  = gross_profit / len(winners) if winners else 0.0
        self.avg_loss = gross_loss   / len(losers)  if losers  else 0.0
        if winners:
            self.largest_win  = max(t.pnl_usd for t in winners)
        if losers:
            self.largest_loss = min(t.pnl_usd for t in losers)

        equity = self.initial_capital
        self.equity_curve = [equity]
        for t in sorted(closed, key=lambda x: x.entry_time):
            equity += t.pnl_usd
            self.equity_curve.append(equity)
        self.final_capital = equity

        arr         = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(arr)
        dd          = arr - running_max
        self.max_drawdown     = float(abs(np.min(dd)))
        self.max_drawdown_pct = float(abs(np.min(dd / np.where(running_max > 0, running_max, 1))))

        pnls       = np.array([t.pnl_usd for t in sorted(closed, key=lambda x: x.entry_time)])
        daily_rets = pnls / (self.initial_capital + 1e-9)
        if len(daily_rets) > 1:
            mu    = np.mean(daily_rets)
            sigma = np.std(daily_rets)
            if sigma > 0:
                self.sharpe_ratio = float(mu / sigma * np.sqrt(252))
            neg = daily_rets[daily_rets < 0]
            if len(neg) > 0 and np.std(neg) > 0:
                self.sortino_ratio = float(mu / np.std(neg) * np.sqrt(252))

        ann_ret = (self.final_capital / self.initial_capital - 1)
        if self.max_drawdown_pct > 0:
            self.calmar_ratio = ann_ret / self.max_drawdown_pct

        max_w = max_l = cur_w = cur_l = 0
        for t in sorted(closed, key=lambda x: x.entry_time):
            if t.is_winner:
                cur_w += 1; cur_l = 0
            else:
                cur_l += 1; cur_w = 0
            max_w = max(max_w, cur_w)
            max_l = max(max_l, cur_l)
        self.consecutive_wins_max  = max_w
        self.consecutive_losses_max = max_l

    def summary(self) -> str:
        lines = [
            "=" * 62,
            f"  BACKTEST RESULTS — {self.symbol}",
            "=" * 62,
            f"  Capital inicial:    ${self.initial_capital:>10,.2f}",
            f"  Capital final:      ${self.final_capital:>10,.2f}",
            f"  PnL Total:          ${self.total_pnl:>10,.2f}",
            f"  Comisiones:         ${self.total_commission:>10,.2f}",
            f"  Retorno:            {(self.final_capital/self.initial_capital-1)*100:>9.2f}%",
            "-" * 62,
            f"  Total trades:       {self.total_trades:>10}",
            f"  Win Rate:           {self.win_rate*100:>9.1f}%",
            f"  Profit Factor:      {self.profit_factor:>10.2f}",
            f"  Avg Win / Loss:     ${self.avg_win:>8,.2f} / ${self.avg_loss:,.2f}",
            f"  Largest Win / Loss: ${self.largest_win:>8,.2f} / ${self.largest_loss:,.2f}",
            "-" * 62,
            f"  Max Drawdown:       ${self.max_drawdown:>10,.2f}  ({self.max_drawdown_pct*100:.1f}%)",
            f"  Sharpe Ratio:       {self.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio:      {self.sortino_ratio:>10.3f}",
            f"  Calmar Ratio:       {self.calmar_ratio:>10.3f}",
            "-" * 62,
            f"  Racha máx wins:     {self.consecutive_wins_max:>10}",
            f"  Racha máx losses:   {self.consecutive_losses_max:>10}",
            "=" * 62,
        ]
        return "\n".join(lines)

    def export_csv(self, filepath: str) -> None:
        fields = [
            "trade_id", "direction", "entry_price", "exit_price",
            "stop_loss", "take_profit_1", "take_profit_2",
            "size_eth", "pnl_usd", "pnl_pct", "commission_paid",
            "status", "zone", "macro_bias", "delta_pct",
            "tp1_hit", "partial_pnl", "duration_seconds",
        ]
        with open(filepath, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for t in self.trades:
                w.writerow({
                    "trade_id":         t.trade_id,
                    "direction":        t.direction,
                    "entry_price":      t.entry_price,
                    "exit_price":       t.exit_price,
                    "stop_loss":        t.stop_loss,
                    "take_profit_1":    t.take_profit_1,
                    "take_profit_2":    t.take_profit_2,
                    "size_eth":         t.size_eth,
                    "pnl_usd":          round(t.pnl_usd, 4),
                    "pnl_pct":          round(t.pnl_pct * 100, 4),
                    "commission_paid":  round(t.commission_paid, 4),
                    "status":           t.status.value,
                    "zone":             t.zone,
                    "macro_bias":       t.macro_bias,
                    "delta_pct":        round(t.delta_pct * 100, 2),
                    "tp1_hit":          t.tp1_hit,
                    "partial_pnl":      round(t.partial_pnl, 4),
                    "duration_seconds": t.duration_seconds,
                })
        logger.info(f"Exportado a {filepath}")


# ──────────────────────────────────────────────────────────────────────────────
# Motor de backtest
# ──────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Simulador de ejecución. NO toma decisiones de trading.

    Responsabilidades:
      - Simular fills realistas (OHLC + slippage dinámico + spread)
      - Gestionar apertura y cierre parcial de posiciones
      - Calcular PnL, comisiones y métricas

    La lógica de señales viene siempre de fuera (signal_fn en run_from_csv).
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        commission_pct:  float = COMMISSION_PCT,   # alias para compatibilidad con main.py
        slippage_pct:    float = BASE_SLIPPAGE,    # alias — se usa como base_slippage
        base_slippage:   float = None,             # alternativa explícita
        risk_per_trade:  float = 0.01,
        min_rr:          float = 1.5,
    ):
        self.initial_capital = initial_capital
        # base_slippage tiene prioridad; si no se pasa, se usa slippage_pct (alias legacy)
        self.base_slippage   = base_slippage if base_slippage is not None else slippage_pct
        self.commission_pct  = commission_pct
        self.risk_per_trade  = risk_per_trade
        self.min_rr          = min_rr
        self.capital         = initial_capital

        self._trades:        List[BacktestTrade]      = []
        self._trade_counter: int                      = 0
        self._open_trade:    Optional[BacktestTrade]  = None

    # ─── Slippage dinámico ────────────────────────────────────────────────────

    def _compute_slippage(self, high: float, low: float, close: float) -> float:
        """
        Slippage escalado por la volatilidad de la barra.
        En barras muy volátiles (high-low grande relativo al close) el fill
        real será peor que en barras tranquilas.
        """
        volatility = (high - low) / (close + 1e-9)
        return self.base_slippage * (1.0 + volatility * 10.0)

    # ─── Proceso de barra ────────────────────────────────────────────────────

    def process_candle(
        self,
        timestamp:        float,
        open_:            float,
        high:             float,
        low:              float,
        close:            float,
        signal_direction: Optional[str] = None,
        stop_loss:        Optional[float] = None,
        take_profit_1:    Optional[float] = None,
        take_profit_2:    Optional[float] = None,
        size_eth:         float = 0.1,
        zone:             str   = "UNKNOWN",
        delta_pct:        float = 0.0,
        macro_bias:       str   = "NEUTRAL",
    ) -> Optional[BacktestTrade]:
        """
        Procesa una barra OHLC.

        Orden de verificación dentro de la barra:
          1. Si hay posición abierta: comprobar SL, TP1, TP2
          2. Si se alcanzó TP1 pero no TP2/SL: cierre parcial
          3. Si no hay posición: intentar abrir con la señal recibida

        Retorna el trade cerrado (si hubo cierre) o None.
        """
        slippage    = self._compute_slippage(high, low, close)
        closed_trade = None

        # ── Gestión de posición abierta ───────────────────────────────────────
        if self._open_trade is not None:
            t = self._open_trade

            if t.direction == "LONG":
                hit_sl  = low  <= t.stop_loss
                hit_tp1 = high >= t.take_profit_1 and not t.tp1_hit
                hit_tp2 = high >= t.take_profit_2
            else:
                hit_sl  = high >= t.stop_loss
                hit_tp1 = low  <= t.take_profit_1 and not t.tp1_hit
                hit_tp2 = low  <= t.take_profit_2

            # Ambigüedad SL + TP en la misma barra: usar apertura como desempate
            if hit_sl and (hit_tp1 or hit_tp2):
                closer_sl = abs(open_ - t.stop_loss) < abs(open_ - t.take_profit_1)
                if closer_sl:
                    hit_tp1 = hit_tp2 = False
                else:
                    hit_sl = False

            # TP2 directo (si TP1 ya fue parcialmente cerrado o llegó antes)
            if hit_tp2:
                t.close_final(t.take_profit_2, timestamp, TradeStatus.CLOSED_TP2, slippage)
                self.capital += t.pnl_usd
                logger.info(t.log_exit())
                closed_trade    = t
                self._open_trade = None

            # TP1: cierre parcial. La posición sigue abierta con size_remaining.
            elif hit_tp1:
                t.apply_partial_tp1(t.take_profit_1, timestamp, slippage)
                self.capital += t.partial_pnl
                logger.debug(
                    f"[TP1 PARCIAL] #{t.trade_id} → cerrado {TP1_CLOSE_RATIO*100:.0f}% "
                    f"@ {t.take_profit_1:.2f} | PnL parcial={t.partial_pnl:.2f} USD"
                )
                # El trade sigue en self._open_trade (estado PARTIAL_TP1)

            elif hit_sl:
                t.close_final(t.stop_loss, timestamp, TradeStatus.CLOSED_SL, slippage)
                self.capital += t.pnl_usd
                logger.info(t.log_exit())
                closed_trade    = t
                self._open_trade = None

        # ── Apertura de nueva posición ────────────────────────────────────────
        if (
            signal_direction is not None
            and self._open_trade is None
            and stop_loss is not None
            and take_profit_2 is not None
        ):
            # Precio de entrada con spread
            if signal_direction == "LONG":
                entry_price = close * (1 + SPREAD_PCT) * (1 + slippage)
            else:
                entry_price = close * (1 - SPREAD_PCT) * (1 - slippage)

            # TP1 por defecto: punto medio entre entrada y TP2
            if take_profit_1 is None:
                if signal_direction == "LONG":
                    take_profit_1 = entry_price + (take_profit_2 - entry_price) * 0.5
                else:
                    take_profit_1 = entry_price - (entry_price - take_profit_2) * 0.5

            self._trade_counter += 1
            t = BacktestTrade(
                trade_id=self._trade_counter,
                direction=signal_direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=take_profit_1,
                take_profit_2=take_profit_2,
                size_eth=size_eth,
                entry_time=timestamp,
                zone=zone,
                delta_pct=delta_pct,
                macro_bias=macro_bias,
            )
            self._open_trade = t
            self._trades.append(t)
            logger.info(t.log_entry())

        return closed_trade

    # ─── Resultados ───────────────────────────────────────────────────────────

    def get_results(self, symbol: str = "ETHUSDT") -> BacktestResults:
        r = BacktestResults(
            trades=self._trades.copy(),
            initial_capital=self.initial_capital,
            final_capital=self.capital,
            symbol=symbol,
            period_start=self._trades[0].entry_time if self._trades else time.time(),
            period_end=time.time(),
        )
        r.compute()
        return r

    def reset(self) -> None:
        self.capital         = self.initial_capital
        self._trades         = []
        self._trade_counter  = 0
        self._open_trade     = None

    # ─── Runner principal ─────────────────────────────────────────────────────

    def run_from_csv(
        self,
        filepath:    str,
        bar_seconds: int = 10,
        signal_fn:   Optional[Callable] = None,
    ) -> BacktestResults:
        """
        Lee un CSV de trades (timestamp, price, quantity, side) y corre el backtest.

        signal_fn:
            Callable(bars: list, bar_idx: int) → Optional[dict]
            El dict debe tener: direction, stop_loss, take_profit_1,
            take_profit_2, zone, delta_pct, macro_bias.
            Si es None, se usa la lógica simplificada interna (2 capas:
            contexto VP + flujo delta dinámico). Útil para pruebas rápidas.

        Para usar el SignalEngine real:
            signal_fn = lambda bars, i: _wrap_signal_engine(signal_engine, bars, i)

        IMPORTANTE: la absorción NO se evalúa en backtest. La información de
        microestructura al nivel de tick no puede reconstruirse fielmente desde
        barras de 10 segundos. Los resultados de backtest con la lógica
        Context + Setup (VP + Delta) son el baseline honesto de la estrategia.
        """
        # ── Leer trades del CSV ───────────────────────────────────────────────
        logger.info(f"Leyendo CSV: {filepath}")
        rows = []
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append({
                        "ts":    float(row["timestamp"]),
                        "price": float(row["price"]),
                        "qty":   float(row["quantity"]),
                        "side":  int(row["side"]),
                    })
                except (KeyError, ValueError):
                    continue

        if not rows:
            raise ValueError(
                "CSV vacío o formato incorrecto. "
                "Columnas requeridas: timestamp, price, quantity, side"
            )

        logger.info(f"{len(rows):,} trades cargados. Agrupando en barras de {bar_seconds}s...")

        # ── Construir barras OHLCV + métricas de flujo ────────────────────────
        bars: List[Dict] = []
        bar_ts   = rows[0]["ts"]
        bar_open = bar_high = bar_low = bar_close = rows[0]["price"]
        bar_vol  = bar_buy = bar_sell = 0.0

        for r in rows:
            if r["ts"] < bar_ts + bar_seconds:
                bar_high  = max(bar_high, r["price"])
                bar_low   = min(bar_low,  r["price"])
                bar_close = r["price"]
                bar_vol  += r["qty"]
                if r["side"] == 1:
                    bar_buy  += r["qty"]
                else:
                    bar_sell += r["qty"]
            else:
                delta    = bar_buy - bar_sell
                delta_pct = delta / (bar_vol + 1e-9)
                bars.append({
                    "ts":        bar_ts,
                    "open":      bar_open,
                    "high":      bar_high,
                    "low":       bar_low,
                    "close":     bar_close,
                    "vol":       bar_vol,
                    "buy":       bar_buy,
                    "sell":      bar_sell,
                    "delta":     delta,
                    "delta_pct": delta_pct,
                })
                steps    = max(1, int((r["ts"] - bar_ts) / bar_seconds))
                bar_ts  += bar_seconds * steps
                bar_open = bar_high = bar_low = bar_close = r["price"]
                bar_vol  = r["qty"]
                bar_buy  = r["qty"] if r["side"] ==  1 else 0.0
                bar_sell = r["qty"] if r["side"] == -1 else 0.0

        if bar_vol > 0:
            delta     = bar_buy - bar_sell
            delta_pct = delta / (bar_vol + 1e-9)
            bars.append({
                "ts": bar_ts, "open": bar_open, "high": bar_high,
                "low": bar_low, "close": bar_close, "vol": bar_vol,
                "buy": bar_buy, "sell": bar_sell,
                "delta": delta, "delta_pct": delta_pct,
            })

        logger.info(f"{len(bars):,} barras generadas.")

        # ── Umbral de delta dinámico (percentil 70) ───────────────────────────
        all_deltas = np.array([abs(b["delta"]) for b in bars])
        delta_threshold = float(np.percentile(all_deltas, 70))
        logger.info(f"Umbral de delta dinámico (p70): {delta_threshold:.4f} ETH")

        # ── Loop principal ────────────────────────────────────────────────────
        self.reset()
        signals_generated = 0

        for i, bar in enumerate(bars):
            price = bar["close"]

            # Usar signal_fn externo si se proporcionó
            if signal_fn is not None:
                sig = signal_fn(bars, i)
            else:
                # Lógica simplificada interna: 2 capas (Context + Setup)
                sig = self._default_signal(
                    bars, i, delta_threshold
                )

            if sig is not None and self._open_trade is None:
                risk   = abs(price - sig["stop_loss"])
                reward = abs(sig["take_profit_2"] - price)
                rr     = reward / risk if risk > 0 else 0.0

                if rr >= self.min_rr and risk > 0:
                    size_eth = (self.capital * self.risk_per_trade) / risk
                    size_eth = min(max(0.001, size_eth), 10.0)

                    self.process_candle(
                        timestamp=bar["ts"],
                        open_=bar["open"],
                        high=bar["high"],
                        low=bar["low"],
                        close=price,
                        signal_direction=sig["direction"],
                        stop_loss=sig["stop_loss"],
                        take_profit_1=sig.get("take_profit_1"),
                        take_profit_2=sig["take_profit_2"],
                        size_eth=size_eth,
                        zone=sig.get("zone", "UNKNOWN"),
                        delta_pct=bar["delta_pct"],
                        macro_bias=sig.get("macro_bias", "NEUTRAL"),
                    )
                    signals_generated += 1
                    continue

            # Sin señal nueva: actualizar posición abierta con la barra
            self.process_candle(
                timestamp=bar["ts"],
                open_=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=price,
            )

        logger.info(
            f"Señales generadas: {signals_generated} | "
            f"Trades ejecutados: {len(self._trades)}"
        )
        return self.get_results()

    # ─── Estrategia interna simplificada (Context + Setup, sin absorción) ─────

    def _default_signal(
        self,
        bars: List[Dict],
        idx:  int,
        delta_threshold: float,
    ) -> Optional[Dict]:
        """
        Evaluación de señal simplificada para backtest.

        CAPA 1 — CONTEXTO (Volume Profile rolling de ~1h):
          Precio fuera del Value Area → hay ventaja posicional.
          Precio dentro → no operar.

        CAPA 2 — SETUP (Delta dinámico):
          El delta de la barra debe superar el percentil 70 del período
          y apuntar en la misma dirección que el contexto.

        NO evalúa absorción: imposible reconstruir microestructura desde barras.

        Parámetros ajustados vs versión anterior:
          VP_WINDOW = 360 barras (~1 hora a 10s/barra)
          tick      = TICK_SIZE * 2  (tolerancia 2 ticks, antes era 5)
          threshold = percentil 70 dinámico (antes MIN_DELTA_PCT = 0.10 fijo)
        """
        VP_WINDOW = 360
        TICK_SIZE = 1.0
        VA_PCT    = 0.70
        tick      = TICK_SIZE * 2   # tolerancia al VAL/VAH (2 ticks)

        vp_bars     = bars[max(0, idx - VP_WINDOW): idx + 1]
        bar         = bars[idx]
        price       = bar["close"]
        delta       = bar["delta"]
        delta_pct   = bar["delta_pct"]

        if len(vp_bars) < 20:
            return None

        all_prices  = np.array([b["close"] for b in vp_bars])
        all_volumes = np.array([b["vol"]   for b in vp_bars])

        p_min = all_prices.min()
        p_max = all_prices.max()
        if p_max <= p_min:
            return None

        # Construir VP
        n_bins    = max(int((p_max - p_min) / TICK_SIZE), 1)
        bin_edges = np.linspace(p_min, p_max + TICK_SIZE, n_bins + 1)
        bin_ctrs  = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_vols  = np.zeros(n_bins)

        idxs = np.clip(
            np.searchsorted(bin_edges, all_prices, side="right") - 1,
            0, n_bins - 1,
        )
        for j, vol in zip(idxs, all_volumes):
            bin_vols[j] += vol

        poc_idx = int(np.argmax(bin_vols))
        target  = bin_vols.sum() * VA_PCT
        acc     = bin_vols[poc_idx]
        hi_idx  = lo_idx = poc_idx

        while acc < target:
            can_up   = hi_idx + 1 < n_bins
            can_down = lo_idx - 1 >= 0
            if not can_up and not can_down:
                break
            v_up   = bin_vols[hi_idx + 1] if can_up   else -1.0
            v_down = bin_vols[lo_idx - 1] if can_down else -1.0
            if v_up >= v_down:
                hi_idx += 1; acc += v_up
            else:
                lo_idx -= 1; acc += v_down

        poc = float(bin_ctrs[poc_idx])
        vah = float(bin_ctrs[hi_idx])
        val = float(bin_ctrs[lo_idx])

        # ── CAPA 1: CONTEXTO ──────────────────────────────────────────────────
        near_val = abs(price - val) <= tick
        near_vah = abs(price - vah) <= tick
        below_va = price < val
        above_va = price > vah

        if not (near_val or near_vah or below_va or above_va):
            return None   # dentro del VA → no operar

        # ── CAPA 2: SETUP (delta dinámico) ───────────────────────────────────
        # LONG: zona baja + delta positivo y fuerte
        if (near_val or below_va) and delta > delta_threshold:
            zone = "BELOW_VALUE_AREA" if below_va else "AT_VAL"
            return {
                "direction":    "LONG",
                "stop_loss":    val - tick * 2,
                "take_profit_1": poc,
                "take_profit_2": vah,
                "zone":         zone,
                "macro_bias":   "NEUTRAL",
            }

        # SHORT: zona alta + delta negativo y fuerte
        if (near_vah or above_va) and delta < -delta_threshold:
            zone = "ABOVE_VALUE_AREA" if above_va else "AT_VAH"
            return {
                "direction":    "SHORT",
                "stop_loss":    vah + tick * 2,
                "take_profit_1": poc,
                "take_profit_2": val,
                "zone":         zone,
                "macro_bias":   "NEUTRAL",
            }

        return None
