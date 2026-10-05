"""
tools/quickstart.py

Script de inicio rápido para verificar que todo el sistema funciona
sin necesitar conexión a Binance.

1. Genera datos sintéticos
2. Ejecuta un backtest completo
3. Muestra las métricas
4. Opcionalmente abre el dashboard con datos simulados

Uso:
  python tools/quickstart.py
  python tools/quickstart.py --trades 50000
  python tools/quickstart.py --show-dashboard
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_quickstart(n_trades: int = 50_000, show_dashboard: bool = False) -> None:
    print("\n" + "═" * 60)
    print("  ETH ORDER FLOW SYSTEM — QUICK START TEST")
    print("═" * 60)

    # ─ Paso 1: Generar datos
    print("\n📊 PASO 1: Generando datos sintéticos...")
    from tools.generate_test_data import generate_synthetic_dataset
    data_path = "data/historical/quickstart_test.csv"
    generate_synthetic_dataset(
        n_trades=n_trades,
        start_price=3500.0,
        output_path=data_path,
        seed=2024,
        verbose=True,
    )

    # ─ Paso 2: Backtest
    print("\n🔄 PASO 2: Ejecutando backtest...")
    from utils.config import CONFIG
    from backtesting.engine import BacktestEngine
    import pandas as pd

    engine = BacktestEngine(
        bt_cfg=CONFIG.backtest,
        vp_cfg=CONFIG.volume_profile,
        of_cfg=CONFIG.order_flow,
        strategy_cfg=CONFIG.strategy,
    )
    df = engine.load_data(data_path)
    t0 = time.time()
    results = engine.run(df)
    elapsed = time.time() - t0
    print(f"\n  ⏱  Completado en {elapsed:.1f}s")

    # ─ Paso 3: Guardar resultados
    print("\n💾 PASO 3: Guardando resultados...")
    Path("logs").mkdir(exist_ok=True)
    if results.trades:
        trades_df = pd.DataFrame([vars(t) for t in results.trades])
        trades_df.to_csv("logs/quickstart_trades.csv", index=False)
        print(f"  Trades guardados: logs/quickstart_trades.csv ({len(results.trades)} trades)")

    # ─ Paso 4: Equity curve
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=["Equity Curve", "Drawdown %", "PnL por Trade"],
            row_heights=[0.5, 0.25, 0.25],
        )

        # Equity
        fig.add_trace(go.Scatter(
            y=results.equity_curve, mode="lines",
            line=dict(color="#00e676", width=2), name="Equity",
            fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
        ), row=1, col=1)
        fig.add_hline(y=results.initial_capital, line_dash="dash",
                      line_color="#ffd600", row=1, col=1,
                      annotation_text=f"Capital inicial ${results.initial_capital:,.0f}")

        # Drawdown
        fig.add_trace(go.Scatter(
            y=results.drawdown_curve, mode="lines", fill="tozeroy",
            line=dict(color="#ff1744", width=1),
            fillcolor="rgba(255,23,68,0.2)", name="Drawdown",
        ), row=2, col=1)

        # PnL por trade
        if results.trades:
            trade_pnls = [t.net_pnl for t in results.trades]
            colors = ["#00e676" if p > 0 else "#ff1744" for p in trade_pnls]
            fig.add_trace(go.Bar(
                y=trade_pnls, marker_color=colors, name="PnL Trade",
            ), row=3, col=1)

        title = (
            f"Quickstart Backtest ETH/USDT Futures | "
            f"Retorno: {results.total_return_pct:+.2f}% | "
            f"Sharpe: {results.sharpe_ratio:.2f} | "
            f"Win Rate: {results.win_rate:.1%} | "
            f"PF: {results.profit_factor:.2f} | "
            f"Max DD: {results.max_drawdown_pct:.2f}%"
        )

        fig.update_layout(
            title=dict(text=title, font=dict(size=12)),
            plot_bgcolor="#11111a", paper_bgcolor="#0a0a0f",
            font=dict(color="#e0e0f0", family="monospace"),
            showlegend=True,
            height=700,
        )
        fig.update_yaxes(gridcolor="#1e1e2e")
        fig.update_xaxes(gridcolor="#1e1e2e")

        equity_path = "logs/quickstart_equity.html"
        fig.write_html(equity_path)
        print(f"  Equity curve: {equity_path}")
        fig.show()

    except ImportError:
        print("  (plotly no disponible para equity curve)")

    # ─ Paso 5: Validación del sistema
    print("\n✅ PASO 4: Validando módulos del sistema...")
    tests = [
        ("DataBuffer", _test_buffer),
        ("OrderFlowEngine", _test_order_flow),
        ("VolumeProfileEngine", _test_vp),
        ("FootprintEngine", _test_footprint),
        ("RiskManager", _test_risk_manager),
        ("SignalEngine", _test_signals),
        ("FuturesTrader (paper)", _test_trader),
    ]

    all_passed = True
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_passed = False

    # ─ Resumen final
    print("\n" + "═" * 60)
    print("  RESUMEN")
    print("═" * 60)
    print(f"  Trades generados:  {n_trades:,}")
    print(f"  Trades ejecutados: {results.total_trades}")
    print(f"  Win Rate:          {results.win_rate:.1%}")
    print(f"  Profit Factor:     {results.profit_factor:.2f}")
    print(f"  Sharpe Ratio:      {results.sharpe_ratio:.2f}")
    print(f"  Max Drawdown:      {results.max_drawdown_pct:.2f}%")
    print(f"  Retorno total:     {results.total_return_pct:+.2f}%")
    print(f"  Módulos OK:        {'✅ TODOS' if all_passed else '⚠️ ALGUNOS FALLARON'}")
    print("═" * 60)
    print("\n  Para iniciar en live:")
    print("  python main.py --mode live")
    print("\n  Para descargar datos reales:")
    print("  python data/data_collector.py --hours 48 --output data/historical/eth_real.csv")
    print()


# ─── Tests de módulos ─────────────────────────────────────────────────────────

def _test_buffer():
    import asyncio
    from data.data_buffer import DataBuffer, Trade
    from utils.config import BufferConfig
    buf = DataBuffer(BufferConfig(max_trades=100))
    trade = Trade(timestamp=time.time(), price=3500.0, quantity=1.0, side=1)
    asyncio.run(buf.add_trade(trade))
    assert buf.size() == 1


def _test_order_flow():
    import asyncio, numpy as np
    from processing.order_flow import OrderFlowEngine
    from utils.config import OrderFlowConfig
    engine = OrderFlowEngine(OrderFlowConfig())
    delta = engine.compute_delta(np.array([
        [time.time(), 3500.0, 1.0, 1],
        [time.time(), 3501.0, 0.5, -1],
    ]))
    assert delta.buy_volume == 1.0
    assert delta.sell_volume == 0.5


def _test_vp():
    from processing.volume_profile import VolumeProfileEngine
    from utils.config import VolumeProfileConfig
    engine = VolumeProfileEngine(VolumeProfileConfig(tick_size=1.0))
    bin_price = engine._price_to_bin(3500.5)
    assert bin_price == 3501.0 or bin_price == 3500.0


def _test_footprint():
    import numpy as np
    from processing.footprint import FootprintEngine
    engine = FootprintEngine(tick_size=1.0, bar_duration_seconds=60)
    arr = np.array([
        [time.time(), 3500.0, 1.0, 1],
        [time.time() + 1, 3501.0, 0.5, -1],
    ])
    engine.process_trades_array(arr)
    summary = engine.get_current_bar_summary()
    assert summary is not None


def _test_risk_manager():
    from strategy.risk_manager import RiskManager, TradeRecord
    rm = RiskManager(initial_capital=10_000, max_consecutive_losses=3)
    rm.update_capital(10_000)
    status = rm.get_status()
    assert status.is_trading_allowed


def _test_signals():
    from strategy.signals import SignalEngine
    from utils.config import StrategyConfig
    engine = SignalEngine(StrategyConfig())
    assert engine is not None
    assert len(engine.active_signals) == 0


def _test_trader():
    import asyncio
    from execution.trader import FuturesTrader
    from utils.config import ExchangeConfig
    trader = FuturesTrader(ExchangeConfig(), paper_mode=True, initial_capital=10_000)
    asyncio.run(trader.initialize())
    assert not trader.has_open_position


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETH Order Flow Quick Start")
    parser.add_argument("--trades", type=int, default=50_000)
    parser.add_argument("--show-dashboard", action="store_true")
    args = parser.parse_args()
    run_quickstart(n_trades=args.trades, show_dashboard=args.show_dashboard)
