import pandas as pd
import numpy as np

def analizar_ventas(df):
    """
    Función que limpia y analiza un DataFrame de ventas con los siguientes parámetros: df; DataFrame con columnas: fecha, producto, cantida y precio_unitario

    Se espera:
        DataFrame agrupado por mes y producto con: ingreso_total y cantidad_promedio
    """
    #copiamos el dataframe
    df_clean = df.copy()

    #eliminamos nulos
    df_clean.dropna(
        subset=["fecha", "producto", "cantidad", "precio_unitario"],
        inplace=True
    )

    #conversión de fecha a datetime
    df_clean["fecha"] = pd.to_datetime(df_clean["fecha"])
    df_clean["mes"] = df_clean["fecha"].dt.to_period("M")

    #calculo dle ingreso total
    df_clean["ingreso_total"] = df_clean["cantidad"] * df_clean["precio_unitario"]

    #agrupamos por mes y producto
    resultado = (
        df_clean.groupby(["mes", "producto"], sort=True)
        .agg(
            ingreso_total=("ingreso_total", "sum"),
            cantidad_promedio=("cantidad", "mean")
        )
        .reset_index()
    )

    resultado["cantidad_promedio"] = resultado["cantidad_promedio"].round(2)

    return resultado
