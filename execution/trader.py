"""
execution/trader.py

Motor de ejecución completo para Binance Futures.

Cambios respecto a la versión anterior:

  1. PULLBACK ENTRY — no se entra inmediatamente al detectar una señal.
     Se espera un retroceso del 0.15% en la dirección contraria antes
     de ejecutar. Mejor precio, menos trades impulsivos.

  2. TP PARCIAL — al alcanzar TP1 se cierra el 50% de la posición y
     el SL se mueve a break-even. El resto corre hacia TP2.
     Position ahora tiene tp1_hit: bool y take_profit_2.

  3. FILTRO DE PRECIO — si el precio ya se movió > 0.2% desde la señal,
     se descarta. Evita perseguir el precio.

  4. SIZING AJUSTADO POR VOLATILIDAD — si el rango SL/TP > 1% del precio,
     el tamaño se reduce a la mitad.

  5. SLIPPAGE + SPREAD REALISTA — slippage 0.03%, spread 0.02%.
     Antes era 0.005% (0.5 bps), demasiado optimista.

  6. NO SOBREOPERAR — execute_signal retorna None si ya hay posición abierta.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import aiohttp

from utils.config import ExchangeConfig
from utils.logger import get_logger

log = get_logger("Trader")


# ──────────────────────────────────────────────────────────────────────────────
# Parámetros de ejecución
# ──────────────────────────────────────────────────────────────────────────────

PULLBACK_PCT    = 0.0015   # 0.15% de retroceso mínimo antes de entrar
MAX_MOVE_PCT    = 0.0020   # 0.2%  máximo movimiento desde la señal (filtro)
SLIP_PCT        = 0.0003   # 0.03% slippage base (antes 0.005%)
SPREAD_PCT      = 0.0002   # 0.02% spread mid-to-touch
VOL_THRESHOLD   = 0.01     # si rango SL/TP > 1% → mercado volátil → qty * 0.5
TP1_RATIO       = 0.50     # fracción de posición que se cierra en TP1


# ──────────────────────────────────────────────────────────────────────────────
# Enums y estructuras base
# ──────────────────────────────────────────────────────────────────────────────

class OrderSide(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET             = "MARKET"
    LIMIT              = "LIMIT"
    STOP_MARKET        = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(Enum):
    FILLED   = "FILLED"
    CANCELED = "CANCELED"
    NEW      = "NEW"


@dataclass
class OrderResult:
    order_id:   int
    symbol:     str
    side:       str
    quantity:   float
    avg_price:  float
    status:     str
    timestamp:  float
    commission: float = 0.0
    is_paper:   bool  = False

    @property
    def fill_cost(self) -> float:
        return self.avg_price * self.quantity


@dataclass
class Position:
    symbol:         str
    side:           str
    entry_price:    float
    quantity:       float
    stop_loss:      float
    take_profit_1:  float
    take_profit_2:  Optional[float] = None
    sl_order_id:    Optional[int]   = None
    tp_order_id:    Optional[int]   = None
    open_time:      float           = field(default_factory=time.time)
    signal_type:    str             = ""
    unrealized_pnl: float           = 0.0

    # Estado de cierre parcial (TP1)
    tp1_hit:        bool  = False   # True cuando ya se ejecutó el parcial

    @property
    def is_long(self) -> bool:
        return self.side == "LONG"

    def pnl_at_price(self, price: float) -> float:
        if self.is_long:
            return (price - self.entry_price) * self.quantity
        return (self.entry_price - price) * self.quantity

    def pnl_pct_at_price(self, price: float) -> float:
        base = self.entry_price * self.quantity
        return self.pnl_at_price(price) / base * 100 if base > 0 else 0.0

    def update_unrealized(self, price: float) -> None:
        self.unrealized_pnl = self.pnl_at_price(price)


# ──────────────────────────────────────────────────────────────────────────────
# Position Sizer
# ──────────────────────────────────────────────────────────────────────────────

class PositionSizer:
    """
    Calcula el tamaño óptimo de posición.
    Estrategias: Fixed Fractional y Half-Kelly Criterion.
    """

    def __init__(self, risk_per_trade: float = 0.01, max_exposure_pct: float = 0.06):
        self.risk_per_trade    = risk_per_trade
        self.max_exposure_pct  = max_exposure_pct

    def fixed_fractional(
        self,
        capital:   float,
        entry:     float,
        stop:      float,
        leverage:  int = 1,
        volatility_pct: float = 0.0,
    ) -> float:
        """
        Qty = (capital * risk_pct * leverage) / |entry - stop|

        Si volatility_pct > VOL_THRESHOLD (mercado muy volátil),
        el tamaño se reduce al 50% para proteger el capital.
        """
        price_risk = abs(entry - stop)
        if price_risk < 0.01:
            return 0.0

        qty = (capital * self.risk_per_trade * leverage) / price_risk

        # Ajuste por volatilidad: si el rango SL/TP es > 1% del precio,
        # el mercado está errático y reducimos exposición
        if volatility_pct > VOL_THRESHOLD:
            qty *= 0.5
            log.debug(f"Volatilidad alta ({volatility_pct:.2%}) → qty reducida al 50%")

        return max(0.0, round(qty, 3))

    def half_kelly(
        self,
        capital:      float,
        entry:        float,
        stop:         float,
        win_rate:     float,
        avg_win_usd:  float,
        avg_loss_usd: float,
        leverage:     int = 1,
        volatility_pct: float = 0.0,
    ) -> float:
        """
        Half-Kelly: f* = 0.5 * (W - (1-W)/R)
        W = win_rate, R = avg_win / avg_loss
        """
        if avg_loss_usd <= 0 or win_rate <= 0 or win_rate >= 1:
            return self.fixed_fractional(capital, entry, stop, leverage, volatility_pct)

        R     = avg_win_usd / avg_loss_usd
        kelly = win_rate - (1 - win_rate) / R
        half  = max(0.0, min(kelly * 0.5, 0.20))

        price_risk = abs(entry - stop)
        if price_risk < 0.01:
            return 0.0

        qty = (capital * half * leverage) / price_risk

        if volatility_pct > VOL_THRESHOLD:
            qty *= 0.5

        return max(0.0, round(qty, 3))


# ──────────────────────────────────────────────────────────────────────────────
# Binance REST client
# ──────────────────────────────────────────────────────────────────────────────

class BinanceFuturesREST:
    """
    Cliente REST para Binance Futures USDM.
    Maneja autenticación HMAC-SHA256, rate limits y reintentos.
    """

    BASE = "https://fapi.binance.com"

    def __init__(self, api_key: str, secret: str):
        self.api_key = api_key
        self.secret  = secret
        self._session: Optional[aiohttp.ClientSession] = None

    async def _session_(self) -> aiohttp.ClientSession:
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self.api_key}
            )
        return self._session

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        q = urllib.parse.urlencode(sorted(params.items()))
        params["signature"] = hmac.new(
            self.secret.encode(), q.encode(), hashlib.sha256
        ).hexdigest()
        return params

    async def _req(self, method: str, path: str, params: dict, retries: int = 3) -> dict:
        url    = self.BASE + path
        params = self._sign(params)
        delay  = 0.5

        for attempt in range(retries):
            try:
                session = await self._session_()
                if method == "GET":
                    async with session.get(url, params=params) as r:
                        data = await r.json()
                        if r.status == 200:
                            return data
                        if r.status == 429:
                            wait = int(r.headers.get("Retry-After", 30))
                            log.warning(f"Rate limit — esperando {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        raise ValueError(f"HTTP {r.status}: {data}")
                else:
                    fn   = session.post if method == "POST" else session.delete
                    body = urllib.parse.urlencode(params)
                    async with fn(url, data=body) as r:
                        data = await r.json()
                        if r.status in (200, 201):
                            return data
                        if r.status == 429:
                            await asyncio.sleep(30)
                            continue
                        raise ValueError(f"HTTP {r.status}: {data}")

            except aiohttp.ClientError as e:
                log.warning(f"Intento {attempt+1}/{retries}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2

        raise ConnectionError(f"Todos los reintentos fallaron: {path}")

    async def get_balance(self) -> float:
        data = await self._req("GET", "/fapi/v2/balance", {})
        for a in data:
            if a["asset"] == "USDT":
                return float(a["availableBalance"])
        return 0.0

    async def set_leverage(self, symbol: str, lev: int) -> None:
        await self._req("POST", "/fapi/v1/leverage",
                        {"symbol": symbol, "leverage": lev})

    async def set_margin_type(self, symbol: str) -> None:
        try:
            await self._req("POST", "/fapi/v1/marginType",
                            {"symbol": symbol, "marginType": "ISOLATED"})
        except ValueError:
            pass  # Ya es ISOLATED

    async def market_order(self, symbol: str, side: OrderSide, qty: float,
                           reduce_only: bool = False) -> dict:
        params: dict = {"symbol": symbol, "side": side.value,
                        "type": "MARKET", "quantity": f"{qty:.3f}"}
        if reduce_only:
            params["reduceOnly"] = "true"
        return await self._req("POST", "/fapi/v1/order", params)

    async def stop_market(self, symbol: str, side: OrderSide, qty: float,
                          stop_price: float) -> dict:
        return await self._req("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side.value,
            "type": "STOP_MARKET", "quantity": f"{qty:.3f}",
            "stopPrice": f"{stop_price:.2f}", "reduceOnly": "true",
        })

    async def take_profit_market(self, symbol: str, side: OrderSide, qty: float,
                                 tp_price: float) -> dict:
        return await self._req("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side.value,
            "type": "TAKE_PROFIT_MARKET", "quantity": f"{qty:.3f}",
            "stopPrice": f"{tp_price:.2f}", "reduceOnly": "true",
        })

    async def cancel_all(self, symbol: str) -> None:
        await self._req("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol})

    async def get_position(self, symbol: str) -> Optional[dict]:
        data = await self._req("GET", "/fapi/v2/positionRisk", {"symbol": symbol})
        for p in data:
            if abs(float(p["positionAmt"])) > 0:
                return p
        return None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Paper Trader
# ──────────────────────────────────────────────────────────────────────────────

class PaperTrader:
    """
    Simulador de paper trading con capital virtual.

    Cambios vs versión anterior:
      - _fill() usa SLIP_PCT (0.03%) + SPREAD_PCT (0.02%) en lugar de 0.005%
      - check_exits() implementa cierre parcial en TP1 (50%) con SL a break-even
      - Position ahora tiene tp1_hit y take_profit_2
    """

    def __init__(self, capital: float = 10_000.0, commission_rate: float = 0.0004):
        self.capital          = capital
        self._initial         = capital
        self.commission_rate  = commission_rate
        self._oid             = 9000
        self._positions:      Dict[str, Position] = {}
        self._closed_trades:  List[dict]          = []

    def _fill(self, side: OrderSide, price: float, qty: float, symbol: str) -> OrderResult:
        """
        Simula un fill con slippage y spread realistas.

        Versión anterior: 0.5 bps de slippage, sin spread → demasiado optimista
        Versión actual:
          - spread de 2 bps (±0.02% mid-to-touch)
          - slippage de 3 bps (0.03%)

        BUY:  fill_price = price + spread + slip  (pagamos más)
        SELL: fill_price = price - spread - slip  (recibimos menos)
        """
        slip   = price * SLIP_PCT
        spread = price * SPREAD_PCT

        if side == OrderSide.BUY:
            fp = price + spread + slip
        else:
            fp = price - spread - slip

        comm = fp * qty * self.commission_rate
        self.capital -= comm
        self._oid += 1

        return OrderResult(
            order_id=self._oid, symbol=symbol, side=side.value,
            quantity=qty, avg_price=fp, status="FILLED",
            timestamp=time.time(), commission=comm, is_paper=True,
        )

    async def open_long(
        self, symbol: str, price: float, qty: float,
        sl: float, tp1: float, signal_type: str = "",
        tp2: Optional[float] = None,
    ) -> Optional[Position]:
        fill = self._fill(OrderSide.BUY, price, qty, symbol)
        pos  = Position(
            symbol=symbol, side="LONG", entry_price=fill.avg_price,
            quantity=qty, stop_loss=sl,
            take_profit_1=tp1, take_profit_2=tp2,
            signal_type=signal_type,
        )
        self._positions[f"{symbol}_LONG"] = pos
        log.info(
            f"📗 PAPER LONG  {qty:.3f} ETH @ ${fill.avg_price:.2f} | "
            f"SL ${sl:.2f} | TP1 ${tp1:.2f}"
            + (f" | TP2 ${tp2:.2f}" if tp2 else "")
        )
        return pos

    async def open_short(
        self, symbol: str, price: float, qty: float,
        sl: float, tp1: float, signal_type: str = "",
        tp2: Optional[float] = None,
    ) -> Optional[Position]:
        fill = self._fill(OrderSide.SELL, price, qty, symbol)
        pos  = Position(
            symbol=symbol, side="SHORT", entry_price=fill.avg_price,
            quantity=qty, stop_loss=sl,
            take_profit_1=tp1, take_profit_2=tp2,
            signal_type=signal_type,
        )
        self._positions[f"{symbol}_SHORT"] = pos
        log.info(
            f"📕 PAPER SHORT {qty:.3f} ETH @ ${fill.avg_price:.2f} | "
            f"SL ${sl:.2f} | TP1 ${tp1:.2f}"
            + (f" | TP2 ${tp2:.2f}" if tp2 else "")
        )
        return pos

    async def check_exits(self, price: float, symbol: str) -> None:
        """
        Verifica SL, TP1 (parcial) y TP2 para todas las posiciones abiertas.

        Lógica de TP1 parcial:
          1. Se cierra TP1_RATIO (50%) de la qty al precio de TP1
          2. El SL se mueve a break-even (entry_price) → riesgo cero
          3. La posición sigue abierta con la qty restante hacia TP2
          4. tp1_hit = True previene que se re-ejecute el parcial

        Lógica de TP2:
          Si la posición ya hizo TP1 parcial, cierra el restante.
          Si no, cierra todo (comportamiento normal sin TP2 configurado).
        """
        to_close = []

        for key, pos in self._positions.items():
            if symbol.upper() not in key.upper():
                continue

            pos.update_unrealized(price)

            if pos.is_long:
                hit_sl  = price <= pos.stop_loss
                hit_tp1 = price >= pos.take_profit_1 and not pos.tp1_hit
                hit_tp2 = pos.take_profit_2 is not None and price >= pos.take_profit_2
            else:
                hit_sl  = price >= pos.stop_loss
                hit_tp1 = price <= pos.take_profit_1 and not pos.tp1_hit
                hit_tp2 = pos.take_profit_2 is not None and price <= pos.take_profit_2

            # ── TP2: cierre final del restante ────────────────────────────
            if hit_tp2:
                close_side = OrderSide.SELL if pos.is_long else OrderSide.BUY
                fill = self._fill(close_side, pos.take_profit_2, pos.quantity, symbol)
                pnl  = pos.pnl_at_price(fill.avg_price) - fill.commission
                self.capital += pnl
                log.info(
                    f"🏆 PAPER CLOSE TP2 [{pos.side}]: "
                    f"${pos.entry_price:.2f}→${fill.avg_price:.2f} | "
                    f"PnL ${pnl:+.2f} | Capital ${self.capital:,.2f}"
                )
                self._closed_trades.append({
                    "direction": pos.side, "signal_type": pos.signal_type,
                    "entry": pos.entry_price, "exit": fill.avg_price,
                    "qty": pos.quantity, "pnl": pnl, "reason": "TP2",
                    "holding": time.time() - pos.open_time,
                })
                to_close.append(key)
                continue

            # ── TP1: cierre parcial 50% + SL a break-even ─────────────────
            if hit_tp1:
                close_qty  = pos.quantity * TP1_RATIO
                close_side = OrderSide.SELL if pos.is_long else OrderSide.BUY
                fill = self._fill(close_side, pos.take_profit_1, close_qty, symbol)

                if pos.is_long:
                    pnl = (fill.avg_price - pos.entry_price) * close_qty
                else:
                    pnl = (pos.entry_price - fill.avg_price) * close_qty
                pnl -= fill.commission

                self.capital    += pnl
                pos.quantity    -= close_qty
                pos.tp1_hit      = True
                pos.stop_loss    = pos.entry_price   # SL a break-even

                log.info(
                    f"🎯 PAPER TP1 PARCIAL [{pos.side}]: "
                    f"cerrado {close_qty:.3f} ETH @ ${fill.avg_price:.2f} | "
                    f"PnL ${pnl:+.2f} | SL movido a break-even ${pos.entry_price:.2f} | "
                    f"Qty restante {pos.quantity:.3f}"
                )
                # La posición sigue abierta, no se añade a to_close
                continue

            # ── SL ────────────────────────────────────────────────────────
            if hit_sl:
                close_side = OrderSide.SELL if pos.is_long else OrderSide.BUY
                fill = self._fill(close_side, pos.stop_loss, pos.quantity, symbol)
                pnl  = pos.pnl_at_price(fill.avg_price) - fill.commission
                self.capital += pnl
                emoji = "✅" if pnl > 0 else "❌"
                log.info(
                    f"{emoji} PAPER CLOSE SL [{pos.side}]: "
                    f"${pos.entry_price:.2f}→${fill.avg_price:.2f} | "
                    f"PnL ${pnl:+.2f} | Capital ${self.capital:,.2f}"
                )
                self._closed_trades.append({
                    "direction": pos.side, "signal_type": pos.signal_type,
                    "entry": pos.entry_price, "exit": fill.avg_price,
                    "qty": pos.quantity, "pnl": pnl, "reason": "SL",
                    "holding": time.time() - pos.open_time,
                })
                to_close.append(key)

        for k in to_close:
            del self._positions[k]

    @property
    def open_positions(self) -> List[Position]:
        return list(self._positions.values())

    @property
    def total_pnl(self) -> float:
        return self.capital - self._initial

    def print_summary(self) -> None:
        total = len(self._closed_trades)
        wins  = [t for t in self._closed_trades if t["pnl"] > 0]
        print(f"\n{'='*52}\n  PAPER TRADING SUMMARY\n{'='*52}")
        print(f"  Capital:    ${self._initial:,.2f} → ${self.capital:,.2f}")
        print(f"  PnL total:  ${self.total_pnl:+,.2f}")
        if total > 0:
            print(f"  Trades:     {total} | Win rate: {len(wins)/total:.1%}")
        print(f"{'='*52}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Futures Trader (orquestador principal)
# ──────────────────────────────────────────────────────────────────────────────

class FuturesTrader:
    """
    Trader principal: orquesta paper trading o real trading.
    Único punto de entrada para ejecutar señales de la estrategia.

    Flujo de una señal:
      1. execute_signal() recibe la señal → NO entra inmediatamente.
         Guarda la señal y activa _waiting_entry = True.
      2. update() se llama en cada tick de precio (cada ~500ms).
         Si hay señal en espera, verifica si el precio retrocedió el
         pullback mínimo. Si sí → ejecuta. Si no → espera.
      3. La señal expira si el precio ya se movió demasiado (>0.2%).
    """

    def __init__(
        self,
        cfg:             ExchangeConfig,
        paper_mode:      bool  = True,
        initial_capital: float = 10_000.0,
        leverage:        int   = 3,
    ):
        self.cfg        = cfg
        self.paper_mode = paper_mode
        self.leverage   = leverage
        self.sizer      = PositionSizer(risk_per_trade=0.01)

        self._active_positions: Dict[str, Position] = {}
        self._initialized = False

        # Estado de espera de pullback
        self._pending_signal        = None
        self._pending_price: Optional[float] = None
        self._waiting_entry: bool   = False

        if paper_mode:
            self.paper = PaperTrader(initial_capital)
            self.rest: Optional[BinanceFuturesREST] = None
            log.info("🟡 Trader: modo PAPER (sin riesgo real)")
        else:
            if not cfg.api_key or not cfg.api_secret:
                raise ValueError("API key y secret son requeridos para trading real.")
            self.paper = None
            self.rest  = BinanceFuturesREST(cfg.api_key, cfg.api_secret)
            log.info("🟢 Trader: modo REAL (Binance Futures)")

    async def initialize(self) -> None:
        if self.paper_mode:
            self._initialized = True
            return
        sym = self.cfg.symbol.upper()
        await self.rest.set_leverage(sym, self.leverage)
        await self.rest.set_margin_type(sym)
        bal = await self.rest.get_balance()
        log.info(f"✅ Trader inicializado | {self.leverage}x leverage | Balance: ${bal:,.2f}")
        self._initialized = True

    # ─── execute_signal ───────────────────────────────────────────────────────

    async def execute_signal(self, signal, current_price: float) -> Optional[Position]:
        """
        Recibe una señal y activa el modo de espera de pullback.

        NO entra inmediatamente. Guarda la señal y espera a que update()
        detecte el retroceso mínimo (PULLBACK_PCT) antes de ejecutar.

        Filtros aplicados ANTES de guardar la señal:
          1. Solo una posición abierta a la vez (no sobreoperar)
          2. Si el precio ya se movió > MAX_MOVE_PCT desde signal.entry_price,
             la señal se descarta (precio perseguido)
        """
        if not self._initialized:
            await self.initialize()

        # Filtro 6: no sobreoperar
        if self.has_open_position:
            log.debug("Ya hay posición abierta, ignorando señal.")
            return None

        # Filtro 3: precio ya se movió demasiado desde que se generó la señal
        if hasattr(signal, "entry_price") and signal.entry_price > 0:
            move_pct = abs(current_price - signal.entry_price) / signal.entry_price
            if move_pct > MAX_MOVE_PCT:
                log.warning(
                    f"❌ Precio ya se movió {move_pct:.2%} desde la señal "
                    f"(máx {MAX_MOVE_PCT:.2%}). Ignorando."
                )
                return None

        # Guardar señal y activar espera de pullback
        self._pending_signal  = signal
        self._pending_price   = current_price
        self._waiting_entry   = True

        log.info(
            f"⏳ Señal en espera de pullback ({PULLBACK_PCT:.2%}): "
            f"{signal.direction.value} @ ${current_price:.2f}"
        )
        return None

    # ─── _do_execute ─────────────────────────────────────────────────────────

    async def _do_execute(self, signal, current_price: float) -> Optional[Position]:
        """
        Ejecuta el trade real una vez confirmado el pullback.
        Separado de execute_signal para claridad del flujo.
        """
        direction = signal.direction.value
        sym       = self.cfg.symbol.upper()
        pos_key   = f"{sym}_{direction}"

        if pos_key in self._active_positions:
            return None

        sl  = signal.stop_loss
        tp1 = signal.take_profit_1
        tp2 = getattr(signal, "take_profit_2", None)

        # Capital disponible
        capital = self.paper.capital if self.paper_mode else await self.rest.get_balance()

        # Volatilidad de la señal: rango SL/TP como % del precio
        vol_range = abs(tp1 - sl) / (current_price + 1e-9)
        qty = self.sizer.fixed_fractional(
            capital, current_price, sl, self.leverage, volatility_pct=vol_range
        )

        if qty < 0.001:
            log.warning(f"Qty demasiado pequeño ({qty:.4f}). Skip.")
            return None

        log.info(
            f"🎯 EJECUTANDO {direction} | {qty:.3f} ETH @ ~${current_price:.2f} | "
            f"SL ${sl:.2f} | TP1 ${tp1:.2f}"
            + (f" | TP2 ${tp2:.2f}" if tp2 else "")
            + f" | Vol {vol_range:.2%}"
        )

        if self.paper_mode:
            if direction == "LONG":
                pos = await self.paper.open_long(
                    sym, current_price, qty, sl, tp1,
                    getattr(signal, "signal_type", ""),
                    tp2=tp2,
                )
            else:
                pos = await self.paper.open_short(
                    sym, current_price, qty, sl, tp1,
                    getattr(signal, "signal_type", ""),
                    tp2=tp2,
                )
            if pos:
                self._active_positions[pos_key] = pos
            return pos
        else:
            return await self._real_bracket(direction, sym, qty, sl, tp1, signal)

    # ─── update ──────────────────────────────────────────────────────────────

    async def update(self, current_price: float) -> None:
        """
        Llamado en cada tick de precio (~500ms).

        1. Si hay señal pendiente de pullback:
           - Verifica si el precio retrocedió lo suficiente → ejecuta
           - Si el precio se alejó demasiado → descarta la señal
        2. Actualiza posiciones abiertas (exits, PnL no realizado)
        """
        # ── Lógica de pullback ────────────────────────────────────────────
        if self._waiting_entry and self._pending_signal is not None:
            signal       = self._pending_signal
            entry_ref    = self._pending_price
            direction    = signal.direction.value

            # Comprobar si el precio se alejó demasiado → señal obsoleta
            drift_pct = abs(current_price - entry_ref) / (entry_ref + 1e-9)
            if drift_pct > MAX_MOVE_PCT * 3:
                log.warning(
                    f"⚠️ Señal cancelada: precio derivó {drift_pct:.2%} desde ${entry_ref:.2f}"
                )
                self._waiting_entry   = False
                self._pending_signal  = None
                self._pending_price   = None
            else:
                # Verificar pullback: precio bajó (LONG) o subió (SHORT)
                ejecutar = False
                if direction == "LONG":
                    ejecutar = current_price <= entry_ref * (1 - PULLBACK_PCT)
                else:
                    ejecutar = current_price >= entry_ref * (1 + PULLBACK_PCT)

                if ejecutar:
                    log.info(
                        f"✅ Pullback confirmado ({PULLBACK_PCT:.2%}): "
                        f"ref=${entry_ref:.2f} → actual=${current_price:.2f}"
                    )
                    self._waiting_entry  = False
                    self._pending_signal = None
                    self._pending_price  = None
                    await self._do_execute(signal, current_price)

        # ── Actualizar posiciones abiertas ────────────────────────────────
        sym = self.cfg.symbol.upper()
        if self.paper_mode and self.paper:
            await self.paper.check_exits(current_price, sym)
            self._active_positions = {
                f"{p.symbol}_{p.side}": p for p in self.paper.open_positions
            }
        else:
            for pos in self._active_positions.values():
                pos.update_unrealized(current_price)

    # ─── Utilidades ───────────────────────────────────────────────────────────

    async def _real_bracket(
        self, direction: str, sym: str, qty: float,
        sl: float, tp: float, signal
    ) -> Optional[Position]:
        side       = OrderSide.BUY  if direction == "LONG" else OrderSide.SELL
        close_side = OrderSide.SELL if direction == "LONG" else OrderSide.BUY
        try:
            entry = await self.rest.market_order(sym, side, qty)
            avg   = float(entry.get("avgPrice", 0))
            log.info(f"✅ Entrada real: {direction} {qty} {sym} @ ${avg:.2f}")

            sl_res = await self.rest.stop_market(sym, close_side, qty, sl)
            tp_res = await self.rest.take_profit_market(sym, close_side, qty, tp)

            pos = Position(
                symbol=sym, side=direction, entry_price=avg, quantity=qty,
                stop_loss=sl, take_profit_1=tp,
                take_profit_2=getattr(signal, "take_profit_2", None),
                sl_order_id=int(sl_res.get("orderId", 0)),
                tp_order_id=int(tp_res.get("orderId", 0)),
                signal_type=getattr(signal, "signal_type", ""),
            )
            self._active_positions[f"{sym}_{direction}"] = pos
            return pos
        except Exception as e:
            log.error(f"Error en bracket order: {e}")
            try:
                await self.rest.cancel_all(sym)
            except Exception:
                pass
            return None

    @property
    def has_open_position(self) -> bool:
        return len(self._active_positions) > 0

    @property
    def open_positions(self) -> List[Position]:
        return list(self._active_positions.values())

    @property
    def is_waiting_entry(self) -> bool:
        """True si hay una señal pendiente de pullback."""
        return self._waiting_entry

    async def cancel_pending(self) -> None:
        """Cancela una señal en espera de pullback sin ejecutarla."""
        if self._waiting_entry:
            log.info("Señal pendiente cancelada manualmente.")
        self._waiting_entry  = False
        self._pending_signal = None
        self._pending_price  = None

    async def emergency_close_all(self) -> None:
        log.warning("🚨 CIERRE DE EMERGENCIA")
        await self.cancel_pending()
        if not self.paper_mode and self.rest:
            sym = self.cfg.symbol.upper()
            await self.rest.cancel_all(sym)
            pos = await self.rest.get_position(sym)
            if pos:
                amt  = float(pos["positionAmt"])
                if abs(amt) > 0:
                    side = OrderSide.SELL if amt > 0 else OrderSide.BUY
                    await self.rest.market_order(sym, side, abs(amt), reduce_only=True)
        self._active_positions.clear()

    async def close(self) -> None:
        if self.rest:
            await self.rest.close()
        if self.paper_mode and self.paper:
            self.paper.print_summary()
