import pandas as pd
import glob
from pathlib import Path

# Carpeta donde están los CSV descomprimidos de Binance
DATA_DIR = Path(r"C:\Users\juanm\Documents\TRADING\ETH\Data")

files = sorted(DATA_DIR.glob("ETHUSDT-aggTrades-*.csv"))

if not files:
    raise FileNotFoundError(f"No se encontraron archivos en {DATA_DIR}")

dfs = []

for f in files:
    print(f"Procesando {f}")

    # Leer todo como texto para evitar problemas de inferencia de tipos
    df = pd.read_csv(
        f,
        header=None,
        names=[
            "agg_trade_id",
            "price",
            "quantity",
            "first_trade_id",
            "last_trade_id",
            "transact_time",
            "is_buyer_maker",
        ],
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )

    # Convertir columnas numéricas de forma segura
    df["agg_trade_id"] = pd.to_numeric(df["agg_trade_id"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["first_trade_id"] = pd.to_numeric(df["first_trade_id"], errors="coerce")
    df["last_trade_id"] = pd.to_numeric(df["last_trade_id"], errors="coerce")
    df["transact_time"] = pd.to_numeric(df["transact_time"], errors="coerce")

    # is_buyer_maker puede venir como True/False, 1/0, o texto
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )

    # Eliminar filas inválidas
    df = df.dropna(subset=["price", "quantity", "transact_time", "is_buyer_maker"])

    # Timestamp en segundos
    df["timestamp"] = df["transact_time"] / 1000.0

    # Side: +1 compra agresiva, -1 venta agresiva
    df["side"] = df["is_buyer_maker"].apply(lambda x: -1 if x else 1)

    dfs.append(df[["timestamp", "price", "quantity", "side"]])

if not dfs:
    raise RuntimeError("No se pudo procesar ningún archivo válido.")

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.sort_values("timestamp")

output_file = DATA_DIR.parent / "eth_trades_real.csv"
df_all.to_csv(output_file, index=False)

print(f"✅ Dataset listo: {output_file}")
print(f"Filas finales: {len(df_all):,}")