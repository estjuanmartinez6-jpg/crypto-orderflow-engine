from backtesting.engine import BacktestEngine

print("🚀 Iniciando backtest con datos reales...")

engine = BacktestEngine(
    initial_capital=10_000,
    commission_pct=0.0004,
    slippage_pct=0.0003,
)

results = engine.run_from_csv(
    "eth_trades_real.csv",
    bar_seconds=5
)

print("\n📊 RESULTADOS:")
print(results.summary())

results.export_csv("resultados_real.csv")

print("\n✅ Backtest terminado")
print("📁 Resultados guardados en resultados_real.csv")