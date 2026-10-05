"""
charts.py — Visualización interactiva con Plotly.

Genera un dashboard completo de 4 paneles:
  1. Precio + Volume Profile lateral + Niveles clave
  2. CVD (Cumulative Volume Delta)
  3. Delta por período
  4. Imbalance del order book

Diseñado para análisis institucional: claro, preciso y sin ruido visual.
"""

import logging
from typing import Optional, List
import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
except ImportError:
    raise ImportError("pip install plotly")

from processing.order_flow import OrderFlowSnapshot, FootprintLevel
from processing.volume_profile import VolumeProfileResult
from strategy.signals import TradeSetup, SignalDirection
from backtesting.engine import BacktestResults
from utils.config import CONFIG

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paleta de colores (tema dark institucional)
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "background": "#0d1117",
    "panel": "#161b22",
    "grid": "#21262d",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "green": "#3fb950",
    "red": "#f85149",
    "yellow": "#d29922",
    "blue": "#58a6ff",
    "purple": "#bc8cff",
    "orange": "#ffa657",
    "poc": "#ffd700",          # Dorado para POC
    "vah": "#00bfff",          # Azul claro para VAH
    "val": "#00bfff",          # Azul claro para VAL
    "hvn": "#2ea043",          # Verde oscuro para HVN
    "lvn": "#da3633",          # Rojo oscuro para LVN
    "long_signal": "#3fb950",
    "short_signal": "#f85149",
    "buy_volume": "#1a472a",
    "sell_volume": "#7f1d1d",
}


def create_main_dashboard(
    snapshot: Optional[OrderFlowSnapshot],
    profile: Optional[VolumeProfileResult],
    price_history: Optional[List[dict]] = None,
    trades: Optional[List[TradeSetup]] = None,
    title: str = "ETH/USDT — Order Flow Dashboard",
) -> go.Figure:
    """
    Dashboard principal de 4 paneles.

    price_history: lista de dicts con {timestamp, open, high, low, close, volume}
    """
    fig = make_subplots(
        rows=4, cols=2,
        column_widths=[0.82, 0.18],
        row_heights=[0.45, 0.20, 0.20, 0.15],
        shared_xaxes=False,
        specs=[
            [{"type": "candlestick"}, {"type": "bar"}],
            [{"type": "scatter", "colspan": 2}, None],
            [{"type": "bar", "colspan": 2}, None],
            [{"type": "scatter", "colspan": 2}, None],
        ],
        subplot_titles=[
            "Precio + Volume Profile", "Profile",
            "CVD (Cumulative Volume Delta)", "",
            "Delta por Período", "",
            "Order Book Imbalance", "",
        ],
        vertical_spacing=0.04,
        horizontal_spacing=0.01,
    )

    # ─── Panel 1: Precio (Candlestick) ────────────────────────────────────────
    if price_history:
        ts = [p["timestamp"] for p in price_history]
        opens = [p["open"] for p in price_history]
        highs = [p["high"] for p in price_history]
        lows = [p["low"] for p in price_history]
        closes = [p["close"] for p in price_history]

        fig.add_trace(go.Candlestick(
            x=ts, open=opens, high=highs, low=lows, close=closes,
            name="ETH/USDT",
            increasing_line_color=COLORS["green"],
            decreasing_line_color=COLORS["red"],
            increasing_fillcolor=COLORS["green"],
            decreasing_fillcolor=COLORS["red"],
        ), row=1, col=1)

    # ─── Niveles del Volume Profile ───────────────────────────────────────────
    if profile and price_history:
        ts_range = [price_history[0]["timestamp"], price_history[-1]["timestamp"]]

        # POC
        fig.add_shape(
            type="line", x0=ts_range[0], x1=ts_range[1],
            y0=profile.poc, y1=profile.poc,
            line=dict(color=COLORS["poc"], width=2, dash="dot"),
            row=1, col=1
        )
        fig.add_annotation(
            x=ts_range[1], y=profile.poc,
            text=f"POC {profile.poc:.2f}",
            font=dict(color=COLORS["poc"], size=11),
            showarrow=False, xanchor="right", row=1, col=1
        )

        # VAH
        fig.add_shape(
            type="line", x0=ts_range[0], x1=ts_range[1],
            y0=profile.vah, y1=profile.vah,
            line=dict(color=COLORS["vah"], width=1.5, dash="dashdot"),
            row=1, col=1
        )
        fig.add_annotation(
            x=ts_range[1], y=profile.vah,
            text=f"VAH {profile.vah:.2f}",
            font=dict(color=COLORS["vah"], size=10),
            showarrow=False, xanchor="right", row=1, col=1
        )

        # VAL
        fig.add_shape(
            type="line", x0=ts_range[0], x1=ts_range[1],
            y0=profile.val, y1=profile.val,
            line=dict(color=COLORS["val"], width=1.5, dash="dashdot"),
            row=1, col=1
        )
        fig.add_annotation(
            x=ts_range[1], y=profile.val,
            text=f"VAL {profile.val:.2f}",
            font=dict(color=COLORS["val"], size=10),
            showarrow=False, xanchor="right", row=1, col=1
        )

        # Value Area shading
        fig.add_shape(
            type="rect",
            x0=ts_range[0], x1=ts_range[1],
            y0=profile.val, y1=profile.vah,
            fillcolor=COLORS["blue"], opacity=0.05,
            line=dict(width=0),
            row=1, col=1
        )

        # HVN
        if CONFIG.visualization.show_hvn_lvn:
            for hvn in profile.hvn_levels:
                fig.add_shape(
                    type="line", x0=ts_range[0], x1=ts_range[1],
                    y0=hvn, y1=hvn,
                    line=dict(color=COLORS["hvn"], width=1, dash="dot"),
                    opacity=0.6, row=1, col=1
                )
            for lvn in profile.lvn_levels:
                fig.add_shape(
                    type="line", x0=ts_range[0], x1=ts_range[1],
                    y0=lvn, y1=lvn,
                    line=dict(color=COLORS["lvn"], width=1, dash="dot"),
                    opacity=0.6, row=1, col=1
                )

    # ─── Señales de trading ───────────────────────────────────────────────────
    if trades and price_history:
        long_trades = [t for t in trades if t.direction == SignalDirection.LONG]
        short_trades = [t for t in trades if t.direction == SignalDirection.SHORT]

        if long_trades:
            fig.add_trace(go.Scatter(
                x=[t.timestamp for t in long_trades],
                y=[t.entry_price for t in long_trades],
                mode="markers",
                marker=dict(
                    symbol="triangle-up", size=14,
                    color=COLORS["long_signal"],
                    line=dict(color="white", width=1)
                ),
                name="Señal LONG",
                text=[f"Score: {t.score:.1f}" for t in long_trades],
            ), row=1, col=1)

        if short_trades:
            fig.add_trace(go.Scatter(
                x=[t.timestamp for t in short_trades],
                y=[t.entry_price for t in short_trades],
                mode="markers",
                marker=dict(
                    symbol="triangle-down", size=14,
                    color=COLORS["short_signal"],
                    line=dict(color="white", width=1)
                ),
                name="Señal SHORT",
                text=[f"Score: {t.score:.1f}" for t in short_trades],
            ), row=1, col=1)

    # ─── Panel 1B: Volume Profile horizontal ─────────────────────────────────
    if profile is not None:
        total = profile.total_volume or 1.0
        norm_vols = profile.volumes / total

        # Colores: más oscuro/claro según intensidad
        colors = []
        max_vol = float(np.max(profile.volumes)) if len(profile.volumes) > 0 else 1.0
        for i, (price, vol) in enumerate(zip(profile.price_levels, profile.volumes)):
            intensity = vol / max_vol
            if price == profile.poc:
                colors.append(COLORS["poc"])
            elif profile.val <= price <= profile.vah:
                g = int(40 + intensity * 60)
                colors.append(f"rgb(30, {g}, 90)")
            else:
                g = int(30 + intensity * 40)
                colors.append(f"rgb({g}, {g}, {g+20})")

        fig.add_trace(go.Bar(
            x=norm_vols,
            y=profile.price_levels,
            orientation="h",
            marker_color=colors,
            name="Volume Profile",
            showlegend=False,
            hovertemplate="Precio: %{y:.2f}<br>Vol: %{x:.4f}<extra></extra>",
        ), row=1, col=2)

    # ─── Panel 2: CVD ─────────────────────────────────────────────────────────
    if snapshot and CONFIG.visualization.show_cvd:
        cvd = snapshot.cvd
        if len(cvd.cvd_values) > 0:
            cvd_colors = [
                COLORS["green"] if v >= 0 else COLORS["red"]
                for v in cvd.cvd_values
            ]
            fig.add_trace(go.Scatter(
                x=cvd.timestamps,
                y=cvd.cvd_values,
                mode="lines",
                line=dict(color=COLORS["blue"], width=2),
                fill="tozeroy",
                fillcolor=f"rgba(88, 166, 255, 0.15)",
                name="CVD",
            ), row=2, col=1)

            # Línea cero
            fig.add_hline(y=0, line_color=COLORS["grid"], line_width=1, row=2, col=1)

    # ─── Panel 3: Delta por período ───────────────────────────────────────────
    if snapshot and CONFIG.visualization.show_delta:
        cvd = snapshot.cvd
        if len(cvd.delta_series) > 0:
            delta_colors = [
                COLORS["green"] if d >= 0 else COLORS["red"]
                for d in cvd.delta_series
            ]
            fig.add_trace(go.Bar(
                x=cvd.timestamps,
                y=cvd.delta_series,
                marker_color=delta_colors,
                name="Delta",
            ), row=3, col=1)
            fig.add_hline(y=0, line_color=COLORS["grid"], line_width=1, row=3, col=1)

    # ─── Panel 4: Imbalance ───────────────────────────────────────────────────
    if snapshot and CONFIG.visualization.show_imbalance and snapshot.imbalance:
        imb = snapshot.imbalance.imbalance
        imb_color = COLORS["green"] if imb > 0 else COLORS["red"]
        fig.add_trace(go.Scatter(
            x=[snapshot.timestamp],
            y=[imb],
            mode="markers+text",
            marker=dict(size=10, color=imb_color),
            text=[f"{imb:.3f}"],
            textposition="top center",
            name="Imbalance",
        ), row=4, col=1)
        fig.add_hline(y=0, line_color=COLORS["grid"], row=4, col=1)
        fig.add_hline(
            y=CONFIG.order_flow.imbalance_threshold,
            line_color=COLORS["green"], line_dash="dot", opacity=0.5, row=4, col=1
        )
        fig.add_hline(
            y=-CONFIG.order_flow.imbalance_threshold,
            line_color=COLORS["red"], line_dash="dot", opacity=0.5, row=4, col=1
        )

    # ─── Estilo global ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=16)),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"], family="JetBrains Mono, monospace"),
        height=CONFIG.visualization.chart_height,
        showlegend=True,
        legend=dict(
            bgcolor=COLORS["panel"],
            bordercolor=COLORS["grid"],
            font=dict(size=10),
        ),
        margin=dict(l=60, r=20, t=60, b=40),
        xaxis_rangeslider_visible=False,
    )

    # Grid styling para todos los subplots
    for i in range(1, 5):
        fig.update_xaxes(
            gridcolor=COLORS["grid"], gridwidth=0.5,
            zeroline=False, tickfont=dict(size=9),
            row=i, col=1
        )
        fig.update_yaxes(
            gridcolor=COLORS["grid"], gridwidth=0.5,
            zeroline=False, tickfont=dict(size=9),
            row=i, col=1
        )

    return fig


def create_backtest_dashboard(results: BacktestResults) -> go.Figure:
    """
    Dashboard de resultados del backtest.
    Equity curve, distribución de PnL, win/loss por calidad de señal.
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Equity Curve", "Distribución de PnL por Trade",
            "PnL por Calidad de Señal", "Drawdown",
        ],
        vertical_spacing=0.12, horizontal_spacing=0.1,
    )

    closed = [t for t in results.trades if t.status.value != "OPEN"]
    trade_nums = list(range(len(results.equity_curve)))

    # Equity curve
    equity_colors = [
        COLORS["green"] if eq >= results.initial_capital else COLORS["red"]
        for eq in results.equity_curve
    ]
    fig.add_trace(go.Scatter(
        x=trade_nums, y=results.equity_curve,
        mode="lines",
        line=dict(color=COLORS["blue"], width=2),
        fill="tozeroy",
        fillcolor="rgba(88, 166, 255, 0.1)",
        name="Capital",
    ), row=1, col=1)
    fig.add_hline(
        y=results.initial_capital,
        line_color=COLORS["yellow"], line_dash="dot",
        row=1, col=1
    )

    # Distribución PnL
    pnls = [t.pnl_usd for t in closed]
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p <= 0]

    if win_pnls:
        fig.add_trace(go.Histogram(
            x=win_pnls, name="Wins",
            marker_color=COLORS["green"], opacity=0.7,
            nbinsx=20,
        ), row=1, col=2)
    if loss_pnls:
        fig.add_trace(go.Histogram(
            x=loss_pnls, name="Losses",
            marker_color=COLORS["red"], opacity=0.7,
            nbinsx=20,
        ), row=1, col=2)

    # PnL por calidad de señal
    quality_groups: dict = {}
    for t in closed:
        q = t.signal_quality
        if q not in quality_groups:
            quality_groups[q] = []
        quality_groups[q].append(t.pnl_usd)

    qualities = sorted(quality_groups.keys())
    avg_pnl_by_quality = [np.mean(quality_groups[q]) for q in qualities]
    colors_quality = [COLORS["green"] if v > 0 else COLORS["red"] for v in avg_pnl_by_quality]

    fig.add_trace(go.Bar(
        x=qualities, y=avg_pnl_by_quality,
        marker_color=colors_quality,
        name="PnL Promedio por Calidad",
    ), row=2, col=1)

    # Drawdown
    eq_arr = np.array(results.equity_curve)
    running_max = np.maximum.accumulate(eq_arr)
    drawdown = (eq_arr - running_max) / running_max * 100

    fig.add_trace(go.Scatter(
        x=trade_nums, y=drawdown,
        mode="lines",
        line=dict(color=COLORS["red"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(248, 81, 73, 0.2)",
        name="Drawdown %",
    ), row=2, col=2)

    fig.update_layout(
        title=dict(
            text=f"Backtest Results — {results.symbol} | "
                 f"WR={results.win_rate*100:.1f}% | "
                 f"PF={results.profit_factor:.2f} | "
                 f"Sharpe={results.sharpe_ratio:.2f}",
            font=dict(color=COLORS["text"], size=14),
        ),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=700,
        showlegend=True,
    )

    for row in range(1, 3):
        for col in range(1, 3):
            fig.update_xaxes(gridcolor=COLORS["grid"], row=row, col=col)
            fig.update_yaxes(gridcolor=COLORS["grid"], row=row, col=col)

    return fig


def create_footprint_chart(
    footprint: List[FootprintLevel],
    title: str = "Footprint Chart — ETH/USDT",
) -> go.Figure:
    """
    Footprint chart: visualización de delta por nivel de precio.
    Permite ver la "huella" de compradores y vendedores.
    """
    if not footprint:
        logger.warning("No hay datos de footprint para visualizar")
        return go.Figure()

    prices = [f.price for f in footprint]
    buy_vols = [f.buy_volume for f in footprint]
    sell_vols = [-f.sell_volume for f in footprint]  # Negativo para mostrar izquierda
    deltas = [f.delta for f in footprint]
    delta_colors = [COLORS["green"] if d > 0 else COLORS["red"] for d in deltas]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=["Volumen Comprador/Vendedor por Precio", "Delta por Nivel"],
    )

    fig.add_trace(go.Bar(
        y=prices, x=buy_vols,
        orientation="h", name="Compras",
        marker_color=COLORS["green"], opacity=0.8,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=prices, x=sell_vols,
        orientation="h", name="Ventas",
        marker_color=COLORS["red"], opacity=0.8,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=prices, x=deltas,
        orientation="h", name="Delta",
        marker_color=delta_colors,
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"])),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["panel"],
        font=dict(color=COLORS["text"]),
        height=600,
        barmode="relative",
    )

    return fig
