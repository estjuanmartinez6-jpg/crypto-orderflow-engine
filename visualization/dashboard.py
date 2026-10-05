"""
visualization/dashboard.py

Dashboard en tiempo real con Dash + Plotly.
Usa los campos reales de OrderFlowSnapshot:
  of.delta       → DeltaResult  (delta, buy_volume, sell_volume, delta_pct, price_close)
  of.cvd         → CVDResult    (current_cvd, cvd_slope)
  of.imbalance   → ImbalanceResult | None  (imbalance, bid_volume, ask_volume, spread)
  of.absorption  → AbsorptionSignal | None
  vp             → VolumeProfileResult (poc, vah, val, hvn_levels, lvn_levels)
"""
from __future__ import annotations

import threading
from typing import Optional

import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from utils.config import DashboardConfig
from utils.logger import get_logger

log = get_logger("Dashboard")

# ── Paleta oscura estilo trading ──────────────────────────────────────────────
C = {
    "bg":      "#0a0a0f",
    "panel":   "#11111a",
    "border":  "#1e1e2e",
    "text":    "#e0e0f0",
    "dim":     "#6e6e8e",
    "green":   "#00e676",
    "red":     "#ff1744",
    "blue":    "#2979ff",
    "yellow":  "#ffd600",
    "cyan":    "#00e5ff",
    "poc":     "#ffd600",
    "vah":     "#00e676",
    "val":     "#ff6d00",
}


class TradingDashboard:

    def __init__(self, cfg: DashboardConfig):
        self.cfg = cfg
        self._lock = threading.Lock()

        # Estado compartido — actualizado desde main.py
        self._of        = None   # OrderFlowSnapshot
        self._vp        = None   # VolumeProfileResult
        self._signals   = []
        self._prices    = []
        self._cvds      = []
        self._stats     = {}

        self.app = self._build_app()

    # ── Actualización de estado (llamado desde main.py) ───────────────────────

    def update_state(self, of_snapshot=None, vp_result=None, signals=None,
                     price_history=None, cvd_history=None, time_history=None,
                     system_stats=None, **kwargs):
        with self._lock:
            self._of      = of_snapshot
            self._vp      = vp_result
            self._signals = (signals or [])[-20:]
            self._prices  = (price_history or [])[-500:]
            self._cvds    = (cvd_history   or [])[-500:]
            self._stats   = system_stats or {}

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_app(self) -> dash.Dash:
        app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.CYBORG],
            title="ETH Order Flow Monitor",
        )

        GRAPH_STYLE = {"height": "420px"}
        SMALL_STYLE = {"height": "210px"}

        app.layout = dbc.Container(fluid=True, style={"backgroundColor": C["bg"], "padding": "10px"}, children=[

            # ── Header ────────────────────────────────────────────────────────
            dbc.Row([
                dbc.Col(html.H5("⚡ ETH/USDT FUTURES — ORDER FLOW MONITOR",
                                style={"color": C["cyan"], "fontFamily": "monospace",
                                       "letterSpacing": "2px", "marginBottom": "0"}), width=5),
                dbc.Col(html.Div(id="stats-bar"), width=7),
            ], style={"borderBottom": f"1px solid {C['border']}", "paddingBottom": "8px",
                      "marginBottom": "10px", "alignItems": "center"}),

            # ── Fila principal ─────────────────────────────────────────────
            dbc.Row([
                # Volume Profile
                dbc.Col(dcc.Graph(id="vp-chart",    style=GRAPH_STYLE,
                                  config={"displayModeBar": False}), width=3),
                # Precio
                dbc.Col(dcc.Graph(id="price-chart", style=GRAPH_STYLE,
                                  config={"displayModeBar": True, "scrollZoom": True}), width=6),
                # Imbalance + señales
                dbc.Col([
                    dcc.Graph(id="imbalance-chart", style=SMALL_STYLE,
                              config={"displayModeBar": False}),
                    html.Div(id="signals-panel", style={"marginTop": "6px"}),
                ], width=3),
            ]),

            # ── Fila inferior: CVD + Delta gauge ──────────────────────────
            dbc.Row([
                dbc.Col(dcc.Graph(id="cvd-chart",   style={"height": "200px"},
                                  config={"displayModeBar": False}), width=8),
                dbc.Col(dcc.Graph(id="delta-gauge", style={"height": "200px"},
                                  config={"displayModeBar": False}), width=4),
            ], style={"marginTop": "8px"}),

            dcc.Interval(id="tick", interval=self.cfg.update_interval_ms, n_intervals=0),
        ])

        self._register_callbacks(app)
        return app

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _register_callbacks(self, app):

        @app.callback(
            [Output("price-chart",    "figure"),
             Output("vp-chart",       "figure"),
             Output("cvd-chart",      "figure"),
             Output("delta-gauge",    "figure"),
             Output("imbalance-chart","figure"),
             Output("signals-panel",  "children"),
             Output("stats-bar",      "children")],
            Input("tick", "n_intervals"),
        )
        def refresh(_):
            with self._lock:
                of      = self._of
                vp      = self._vp
                sigs    = list(self._signals)
                prices  = list(self._prices)
                cvds    = list(self._cvds)
                stats   = dict(self._stats)

            return (
                self._price_chart(prices, vp, of, sigs),
                self._vp_chart(vp, prices[-1] if prices else 0),
                self._cvd_chart(cvds),
                self._delta_gauge(of),
                self._imbalance_chart(of),
                self._signals_panel(sigs),
                self._stats_bar(of, stats),
            )

    # ── Gráfica de precio ─────────────────────────────────────────────────────

    def _price_chart(self, prices, vp, of, signals=None):
        fig = go.Figure()
        if not prices:
            return self._empty("Sin datos de precio")

        x = list(range(len(prices)))
        fig.add_trace(go.Scatter(
            x=x, y=prices, mode="lines", name="ETH/USDT",
            line=dict(color=C["cyan"], width=1.5),
        ))

        # VWAP manual
        if len(prices) > 1:
            vwap = sum(prices) / len(prices)
            fig.add_hline(y=vwap, line_color=C["yellow"], line_dash="dash",
                          line_width=1, annotation_text=f"VWAP ${vwap:.1f}",
                          annotation_position="right")

        # Value Area del VP
        if vp:
            fig.add_hline(y=vp.poc, line_color=C["poc"], line_dash="dot", line_width=1.5,
                          annotation_text=f"POC ${vp.poc:.1f}", annotation_position="right")
            fig.add_hline(y=vp.vah, line_color=C["vah"], line_dash="dash", line_width=1,
                          annotation_text=f"VAH ${vp.vah:.1f}", annotation_position="right")
            fig.add_hline(y=vp.val, line_color=C["val"], line_dash="dash", line_width=1,
                          annotation_text=f"VAL ${vp.val:.1f}", annotation_position="right")
            fig.add_hrect(y0=vp.val, y1=vp.vah,
                          fillcolor="rgba(41,121,255,0.06)", line_width=0)

        # ── Niveles de señales activas: Entry, SL, TP1, TP2 ──────────────────
        if signals:
            # Mostrar solo la ultima señal (la mas reciente)
            last = signals[-1]
            try:
                is_long   = last.direction.value == "LONG"
                entry     = last.entry_price
                sl        = last.stop_loss
                tp1       = last.take_profit_1
                tp2       = last.take_profit_2
                quality   = last.quality.value
                score     = last.score

                # Zona de riesgo: entre entry y SL (rojo translucido)
                fig.add_hrect(
                    y0=min(entry, sl), y1=max(entry, sl),
                    fillcolor="rgba(255,23,68,0.12)",
                    line_width=0,
                )

                # Zona de beneficio: entre entry y TP2 (verde translucido)
                fig.add_hrect(
                    y0=min(entry, tp2), y1=max(entry, tp2),
                    fillcolor="rgba(0,230,118,0.08)",
                    line_width=0,
                )

                # Linea de entrada
                arrow = "▲" if is_long else "▼"
                fig.add_hline(
                    y=entry,
                    line_color="#ffffff",
                    line_dash="solid",
                    line_width=1.5,
                    annotation_text=f"{arrow} ENTRY [{quality}] ${entry:.2f}  Score:{score:.1f}",
                    annotation_position="left",
                    annotation_font=dict(color="#ffffff", size=10),
                )

                # Stop Loss — rojo solido
                fig.add_hline(
                    y=sl,
                    line_color="#ff1744",
                    line_dash="solid",
                    line_width=1.5,
                    annotation_text=f"SL ${sl:.2f}",
                    annotation_position="left",
                    annotation_font=dict(color="#ff1744", size=10),
                )

                # TP1 — verde punteado (objetivo parcial)
                fig.add_hline(
                    y=tp1,
                    line_color="#00e676",
                    line_dash="dot",
                    line_width=1.5,
                    annotation_text=f"TP1 ${tp1:.2f}",
                    annotation_position="left",
                    annotation_font=dict(color="#00e676", size=10),
                )

                # TP2 — verde solido (objetivo final)
                fig.add_hline(
                    y=tp2,
                    line_color="#00e676",
                    line_dash="solid",
                    line_width=2,
                    annotation_text=f"TP2 ${tp2:.2f}  RR:{last.risk_reward:.1f}x",
                    annotation_position="left",
                    annotation_font=dict(color="#00e676", size=10),
                )

                # Marcador de entrada en el chart (triangulo en el precio actual)
                if prices:
                    marker_color = "#00e676" if is_long else "#ff1744"
                    marker_symbol = "triangle-up" if is_long else "triangle-down"
                    fig.add_trace(go.Scatter(
                        x=[len(prices) - 1],
                        y=[entry],
                        mode="markers",
                        marker=dict(
                            symbol=marker_symbol,
                            size=14,
                            color=marker_color,
                            line=dict(color="#ffffff", width=1),
                        ),
                        name=f"{arrow} {last.direction.value}",
                        showlegend=True,
                    ))

            except AttributeError:
                pass   # señal con formato inesperado — ignorar silenciosamente

        self._theme(fig, "Precio ETH/USDT Futures")
        return fig

    # ── Volume Profile horizontal ─────────────────────────────────────────────

    def _vp_chart(self, vp, current_price):
        if vp is None or len(vp.price_levels) == 0:
            return self._empty("Calculando Volume Profile...")

        prices     = vp.price_levels
        buy_vols   = vp.buy_volumes
        sell_vols  = vp.sell_volumes

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=prices, x=buy_vols, orientation="h", name="Buy",
            marker_color=C["green"], opacity=0.75,
        ))
        fig.add_trace(go.Bar(
            y=prices, x=[-v for v in sell_vols], orientation="h", name="Sell",
            marker_color=C["red"], opacity=0.75,
        ))

        fig.add_hline(y=vp.poc, line_color=C["poc"],  line_width=2,
                      annotation_text="POC")
        fig.add_hline(y=vp.vah, line_color=C["vah"],  line_width=1, line_dash="dash")
        fig.add_hline(y=vp.val, line_color=C["val"],  line_width=1, line_dash="dash")
        if current_price:
            fig.add_hline(y=current_price, line_color=C["cyan"], line_width=1.5,
                          annotation_text=f"${current_price:.1f}")

        fig.update_layout(barmode="overlay", showlegend=False)
        self._theme(fig, "Volume Profile")
        return fig

    # ── CVD ───────────────────────────────────────────────────────────────────

    def _cvd_chart(self, cvds):
        if not cvds:
            return self._empty("Sin datos CVD")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=cvds, mode="lines", name="CVD",
            line=dict(color=C["blue"], width=1.5),
            fill="tozeroy", fillcolor="rgba(41,121,255,0.12)",
        ))
        fig.add_hline(y=0, line_color=C["dim"], line_width=0.5)
        self._theme(fig, "Cumulative Volume Delta (CVD)")
        return fig

    # ── Delta gauge ───────────────────────────────────────────────────────────

    def _delta_gauge(self, of):
        if of is None:
            return self._empty("Sin datos")

        d        = of.delta
        val_pct  = d.delta_pct          # ya es porcentaje (delta/total*100)
        buy_pct  = d.buy_volume  / (d.total_volume + 1e-9) * 100
        sell_pct = d.sell_volume / (d.total_volume + 1e-9) * 100
        color    = C["green"] if val_pct >= 0 else C["red"]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val_pct,
            title={"text": f"Delta % | Buy {buy_pct:.1f}% / Sell {sell_pct:.1f}%",
                   "font": {"color": C["text"], "size": 10}},
            number={"suffix": "%", "font": {"color": color}},
            gauge={
                "axis":  {"range": [-100, 100], "tickcolor": C["dim"]},
                "bar":   {"color": color},
                "steps": [{"range": [-100, 0], "color": "rgba(255,23,68,0.08)"},
                           {"range": [0, 100],  "color": "rgba(0,230,118,0.08)"}],
            },
        ))
        self._theme(fig, "")
        return fig

    # ── Book Imbalance ────────────────────────────────────────────────────────

    def _imbalance_chart(self, of):
        if of is None or of.imbalance is None:
            return self._empty("Sin datos de libro")

        imb    = of.imbalance
        val    = imb.imbalance          # -1 a +1
        color  = C["green"] if val > 0 else C["red"]
        spread = imb.spread or 0

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val * 100,
            title={"text": (f"Book Imbalance | "
                            f"Bid {imb.bid_volume:.1f} / Ask {imb.ask_volume:.1f} | "
                            f"Spread ${spread:.2f}"),
                   "font": {"color": C["text"], "size": 9}},
            number={"suffix": "%", "font": {"color": color}},
            gauge={
                "axis": {"range": [-100, 100]},
                "bar":  {"color": color},
            },
        ))
        self._theme(fig, "")
        return fig

    # ── Panel de señales ──────────────────────────────────────────────────────

    def _signals_panel(self, signals):
        if not signals:
            return dbc.Alert("Sin señales activas", color="secondary",
                             style={"fontSize": "11px", "padding": "4px 8px", "margin": "0"})

        # Mostrar la señal mas reciente con todos sus niveles
        cards = []
        for s in signals[-3:]:
            try:
                direction  = s.direction.value
                is_long    = direction == "LONG"
                dir_color  = C["green"] if is_long else C["red"]
                arrow      = "▲" if is_long else "▼"
                risk       = abs(s.entry_price - s.stop_loss)
                reward2    = abs(s.take_profit_2 - s.entry_price)

                card = html.Div([
                    # Cabecera: direccion + calidad
                    html.Div([
                        html.Span(f"{arrow} {direction}",
                                  style={"color": dir_color, "fontWeight": "bold",
                                         "fontSize": "12px"}),
                        html.Span(f"  [{s.quality.value}]  Score {s.score:.1f}/10",
                                  style={"color": C["yellow"], "fontSize": "10px"}),
                    ]),
                    # Niveles
                    html.Div([
                        html.Span("ENTRY ", style={"color": C["dim"], "fontSize": "9px"}),
                        html.Span(f"${s.entry_price:.2f}",
                                  style={"color": "#ffffff", "fontWeight": "bold", "fontSize": "11px"}),
                    ], style={"marginTop": "4px"}),
                    html.Div([
                        html.Span("SL    ", style={"color": C["dim"], "fontSize": "9px"}),
                        html.Span(f"${s.stop_loss:.2f}",
                                  style={"color": C["red"], "fontWeight": "bold", "fontSize": "11px"}),
                        html.Span(f"  (-${risk:.2f})",
                                  style={"color": C["red"], "fontSize": "9px"}),
                    ]),
                    html.Div([
                        html.Span("TP1   ", style={"color": C["dim"], "fontSize": "9px"}),
                        html.Span(f"${s.take_profit_1:.2f}",
                                  style={"color": C["green"], "fontSize": "11px"}),
                    ]),
                    html.Div([
                        html.Span("TP2   ", style={"color": C["dim"], "fontSize": "9px"}),
                        html.Span(f"${s.take_profit_2:.2f}",
                                  style={"color": C["green"], "fontWeight": "bold", "fontSize": "11px"}),
                        html.Span(f"  RR {s.risk_reward:.1f}x",
                                  style={"color": C["cyan"], "fontSize": "10px"}),
                    ]),
                    # Confirmaciones
                    html.Div(
                        "✓ " + "  ✓ ".join(s.confirmations),
                        style={"color": C["dim"], "fontSize": "9px", "marginTop": "3px",
                               "borderTop": f"1px solid {C['border']}", "paddingTop": "3px"},
                    ),
                ], style={
                    "backgroundColor": C["panel"],
                    "border": f"1px solid {dir_color}33",
                    "borderLeft": f"3px solid {dir_color}",
                    "borderRadius": "4px",
                    "padding": "6px 8px",
                    "marginBottom": "6px",
                    "fontFamily": "monospace",
                })
                cards.append(card)
            except Exception:
                continue

        return html.Div(cards) if cards else dbc.Alert(
            "Sin señales activas", color="secondary",
            style={"fontSize": "11px", "padding": "4px 8px", "margin": "0"}
        )

    # ── Barra de estadísticas ─────────────────────────────────────────────────

    def _stats_bar(self, of, stats):
        if of is None:
            return html.Span("Conectando a Binance...",
                             style={"color": C["dim"], "fontFamily": "monospace"})

        d       = of.delta
        price   = d.price_close
        cvd_now = of.cvd.current_cvd
        cvd_col = C["green"] if cvd_now >= 0 else C["red"]

        # Spread del libro (si disponible)
        spread_str = ""
        if of.imbalance:
            spread_str = f"${of.imbalance.spread:.2f}"

        # Absorción activa
        abs_str = ""
        if of.absorption:
            abs_str = f"🔔 {of.absorption.direction}"

        items = [
            ("PRECIO",    f"${price:,.2f}",           C["cyan"]),
            ("CVD",       f"{cvd_now:+,.1f}",          cvd_col),
            ("DELTA%",    f"{d.delta_pct:+.1f}%",     C["green"] if d.delta_pct >= 0 else C["red"]),
            ("SPREAD",    spread_str or "—",           C["dim"]),
            ("TRADES/s",  f"{stats.get('trades_per_second', 0):.1f}", C["text"]),
            ("BUFFER",    f"{stats.get('trades_in_buffer', 0):,}", C["dim"]),
        ]

        spans = []
        for label, value, color in items:
            spans.append(html.Span([
                html.Span(label + ": ", style={"color": C["dim"],  "fontSize": "10px"}),
                html.Span(value + "   ", style={"color": color, "fontWeight": "bold",
                                                "fontSize": "12px", "fontFamily": "monospace"}),
            ]))

        if abs_str:
            spans.append(html.Span(abs_str, style={"color": C["yellow"], "fontSize": "11px",
                                                   "fontFamily": "monospace"}))

        return html.Div(spans, style={"display": "flex", "alignItems": "center",
                                      "flexWrap": "wrap", "justifyContent": "flex-end"})

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _empty(self, msg: str) -> go.Figure:
        fig = go.Figure()
        fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                           font=dict(color=C["dim"], size=13), showarrow=False)
        self._theme(fig, "")
        return fig

    def _theme(self, fig: go.Figure, title: str) -> None:
        fig.update_layout(
            plot_bgcolor=C["panel"], paper_bgcolor=C["bg"],
            font=dict(color=C["text"], size=10, family="monospace"),
            title=dict(text=title, font=dict(size=11, color=C["dim"])),
            margin=dict(l=40, r=60, t=28, b=28),
            xaxis=dict(gridcolor=C["border"], showgrid=True, zeroline=False),
            yaxis=dict(gridcolor=C["border"], showgrid=True, zeroline=False),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        )

    def run(self, debug: bool = False) -> None:
        log.info(f"Dashboard iniciado en http://{self.cfg.host}:{self.cfg.port}")
        self.app.run(
            host=self.cfg.host,
            port=self.cfg.port,
            debug=debug,
            use_reloader=False,
        )
