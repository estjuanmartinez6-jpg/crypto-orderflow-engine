"""
main.py — Orquestador principal del sistema ETH Order Flow.

APIs reales verificadas:
  DataBuffer(max_trades)
  BinanceWebSocketClient(buffer) → .run() / .stop()
  OrderFlowEngine(buffer) → .snapshot()
  VolumeProfileEngine(buffer) → .compute_session_profile()
  SignalEngine(buffer, vp_engine) → .evaluate(snapshot, profile) → .recent_signals(n)
  TradeSetup → .signal_type (str), .quality, .score, .direction
  BacktestEngine() → .run_from_csv(path) → BacktestResults
  BacktestResults → .total_pnl, .win_rate, .profit_factor, .summary()

Modos:
  generate  — genera CSV de trades sinteticos
  backtest  — backtest sobre ese CSV
  live      — conecta a Binance (paper trading por defecto)
"""
from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import threading
import time
from pathlib import Path
from typing import List, Optional

from data.data_buffer import DataBuffer
from data.websocket_client import BinanceWebSocketClient
from processing.order_flow import OrderFlowEngine, OrderFlowSnapshot
from processing.volume_profile import VolumeProfileEngine, VolumeProfileResult
from strategy.signals import SignalEngine, TradeSetup
from execution.trader import FuturesTrader
from visualization.dashboard import TradingDashboard
from utils.config import CONFIG
from utils.logger import get_logger

log = get_logger("Main", CONFIG.log_file, CONFIG.log_level)


class TradingSystem:

    def __init__(self, paper_mode: bool = True):
        log.info("=" * 55)
        log.info("  ETH ORDER FLOW + VOLUME PROFILE  v2.0")
        log.info("=" * 55)

        self.buffer     = DataBuffer(CONFIG.buffer.max_trades)
        self.ws         = BinanceWebSocketClient(self.buffer)
        self.of_engine  = OrderFlowEngine(self.buffer)
        self.vp_engine  = VolumeProfileEngine(self.buffer)
        self.sig_engine = SignalEngine(self.buffer, self.vp_engine)
        self.trader     = FuturesTrader(
            cfg=CONFIG.exchange,
            paper_mode=paper_mode,
            initial_capital=CONFIG.backtest.initial_capital,
            leverage=CONFIG.backtest.leverage,
        )
        self.dashboard  = TradingDashboard(CONFIG.dashboard)

        self._running        = False
        self._price_history: List[float] = []
        self._cvd_history:   List[float] = []
        self._session_start  = time.time()

    # ── Loop de calculo (modo live) ───────────────────────────────────────────

    async def _calc_loop(self) -> None:
        INTERVAL = 0.5
        log.info("Loop de calculo iniciado.")

        while self._running:
            t0 = time.time()
            try:
                if self.buffer.trade_count < 50:
                    await asyncio.sleep(INTERVAL)
                    continue

                # 1. Order Flow
                of: Optional[OrderFlowSnapshot] = self.of_engine.snapshot()
                if of is None:
                    await asyncio.sleep(INTERVAL)
                    continue

                price = self.buffer.last_price or of.delta.price_close

                # 2. Historial
                self._price_history.append(price)
                self._cvd_history.append(of.cvd.current_cvd)
                if len(self._price_history) > 2000:
                    self._price_history = self._price_history[-2000:]
                    self._cvd_history   = self._cvd_history[-2000:]

                # 3. Volume Profile
                vp: Optional[VolumeProfileResult] = self.vp_engine.compute_session_profile()

                # 4. Posiciones abiertas
                await self.trader.update(price)

                # 5. Señales
                if vp and not self.trader.has_open_position:
                    setup: Optional[TradeSetup] = self.sig_engine.evaluate(of, vp)
                    if setup:
                        self._log_setup(setup)
                        await self.trader.execute_signal(setup, price)

                # 6. Dashboard
                stats   = self.buffer.get_stats()
                elapsed = max(time.time() - self._session_start, 1)
                self.dashboard.update_state(
                    of_snapshot=of,
                    vp_result=vp,
                    signals=self.sig_engine.recent_signals(20),
                    price_history=self._price_history,
                    cvd_history=self._cvd_history,
                    time_history=list(range(len(self._price_history))),
                    system_stats={
                        **stats,
                        "trades_per_second": stats["total_trades_received"] / elapsed,
                        "open_positions": len(self.trader.open_positions),
                    },
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Error: {type(e).__name__}: {e}", exc_info=True)

            await asyncio.sleep(max(0.0, INTERVAL - (time.time() - t0)))

        log.info("Loop detenido.")

    def _log_setup(self, setup: TradeSetup) -> None:
        arrow = "=>" if setup.direction.value == "LONG" else "<="
        log.info(
            f"{arrow} [{setup.quality.value}] {setup.direction.value} "
            f"@ ${setup.entry_price:.2f} | SL ${setup.stop_loss:.2f} | "
            f"TP ${setup.take_profit_1:.2f} | RR {setup.risk_reward:.2f} | "
            f"Score {setup.score:.1f}/10"
        )

    # ── Modo live ─────────────────────────────────────────────────────────────

    async def run_live(self) -> None:
        self._running = True

        dash_thread = threading.Thread(
            target=self.dashboard.run, kwargs={"debug": False}, daemon=True
        )
        dash_thread.start()
        log.info(f"Dashboard => http://{CONFIG.dashboard.host}:{CONFIG.dashboard.port}")

        await self.trader.initialize()

        tasks = [
            asyncio.create_task(self._calc_loop(), name="calc"),
            asyncio.create_task(self.ws.run(),     name="ws"),
        ]
        log.info("Sistema activo. Ctrl+C para detener.")
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            self.ws.stop()
            await self.trader.close()
            log.info("Sistema detenido.")

    # ── Modo backtest ─────────────────────────────────────────────────────────

    def run_backtest(self, data_path: str) -> None:
        from backtesting.engine import BacktestEngine

        log.info(f"Backtesting: {data_path}")
        engine  = BacktestEngine(
            initial_capital=CONFIG.backtest.initial_capital,
            commission_pct=CONFIG.backtest.commission_pct,
            slippage_pct=CONFIG.backtest.slippage_pct,
        )
        results = engine.run_from_csv(data_path)

        # Imprimir resumen en consola
        print(results.summary())

        # Guardar trades a CSV
        Path("logs").mkdir(exist_ok=True)
        results.export_csv("logs/backtest_trades.csv")
        log.info("Trades exportados => logs/backtest_trades.csv")

        # Grafica de equity (requiere plotly)
        self._plot_equity(results)

    def _plot_equity(self, results) -> None:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                subplot_titles=["Equity Curve", "Drawdown %"],
                row_heights=[0.7, 0.3],
            )

            eq = results.equity_curve
            running_max = [max(eq[:i+1]) for i in range(len(eq))]
            dd = [(e - m) / m * 100 for e, m in zip(eq, running_max)]

            fig.add_trace(go.Scatter(
                y=eq, mode="lines",
                line=dict(color="#00e676", width=2), name="Equity",
                fill="tozeroy", fillcolor="rgba(0,230,118,0.07)",
            ), row=1, col=1)
            fig.add_hline(
                y=results.initial_capital,
                line_dash="dash", line_color="#ffd600", row=1, col=1,
            )
            fig.add_trace(go.Scatter(
                y=dd, mode="lines", fill="tozeroy",
                line=dict(color="#ff1744", width=1),
                fillcolor="rgba(255,23,68,0.2)", name="Drawdown %",
            ), row=2, col=1)

            ret_pct = (results.final_capital / results.initial_capital - 1) * 100
            fig.update_layout(
                title=(
                    f"ETH Order Flow Backtest | "
                    f"Retorno: {ret_pct:+.2f}% | "
                    f"Win Rate: {results.win_rate:.1%} | "
                    f"PF: {results.profit_factor:.2f} | "
                    f"Sharpe: {results.sharpe_ratio:.2f} | "
                    f"Max DD: {results.max_drawdown_pct*100:.1f}%"
                ),
                plot_bgcolor="#11111a",
                paper_bgcolor="#0a0a0f",
                font=dict(color="#e0e0f0", family="monospace"),
                height=600,
            )
            fig.update_yaxes(gridcolor="#1e1e2e")
            fig.update_xaxes(gridcolor="#1e1e2e")

            fig.write_html("logs/backtest_equity.html")
            log.info("Equity curve => logs/backtest_equity.html")
            fig.show()

        except ImportError:
            log.warning("Plotly no instalado. Sin grafica de equity.")
        except Exception as e:
            log.warning(f"No se pudo generar grafica: {e}")

    # ── Modo generate ─────────────────────────────────────────────────────────

    def run_generate(self, n_trades: int, output_path: str) -> None:
        from tools.generate_test_data import generate_synthetic_dataset
        generate_synthetic_dataset(n_trades=n_trades, output_path=output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="ETH Order Flow + Volume Profile System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py --mode generate            genera datos sinteticos
  python main.py --mode backtest            backtest sobre esos datos
  python main.py --mode live                live paper trading + dashboard
  python main.py --mode live --real         live trading real (requiere .env)
""",
    )
    p.add_argument("--mode", choices=["live", "backtest", "generate"], default="live")
    p.add_argument("--data", default="data/historical/eth_backtest.csv",
                   help="CSV de datos (para backtest/generate)")
    p.add_argument("--real", action="store_true",
                   help="Trading real en Binance (requiere BINANCE_API_KEY en .env)")
    p.add_argument("--trades", type=int, default=100_000,
                   help="Trades sinteticos a generar (modo generate)")
    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING"], default="INFO")
    return p.parse_args()


async def _async_main(args) -> None:
    system = TradingSystem(paper_mode=not args.real)
    loop   = asyncio.get_event_loop()

    def _stop():
        for t in asyncio.all_tasks(loop):
            t.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except (NotImplementedError, OSError):
            pass  # Windows no soporta add_signal_handler

    await system.run_live()


def main() -> None:
    args = parse_args()
    CONFIG.log_level = args.log_level

    if args.mode == "generate":
        log.info(f"Generando {args.trades:,} trades sinteticos => {args.data}")
        Path(args.data).parent.mkdir(parents=True, exist_ok=True)
        TradingSystem().run_generate(args.trades, args.data)
        log.info("Listo. Ahora corre:  python main.py --mode backtest")
        return

    if args.mode == "backtest":
        if not Path(args.data).exists():
            log.error(f"Archivo no encontrado: {args.data}")
            log.info("Primero genera los datos:  python main.py --mode generate")
            sys.exit(1)
        TradingSystem().run_backtest(args.data)
        return

    # Modo live
    try:
        asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        log.info("Detenido por el usuario.")


if __name__ == "__main__":
    main()