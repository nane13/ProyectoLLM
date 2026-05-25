import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest

def detectar_anomalias_energeticas(df):
    """
    La función se encarga de detectar anomalías en un dataframe con registros de consumo energético

    Contamos con los siguientes parametros:
        df: DataFrame con las columnas: consumo_kwh, temperatura_ambiente y tipo_tarifa

    Se espera:
        DataFrame limpio con una nueva columna 'es_anomalia', donde 1 indica registro normal y -1 indica anomalía
    """
    #eliminamos filas donde consumo_kwh sea nulo
    df_clean = df.dropna(subset=["consumo_kwh"]).copy()

    #definimos columnas
    num_col = ["temperatura_ambiente"]
    cat_col = ["tipo_tarifa"]

    #creamos el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="mean"), num_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_col)
        ]
    )

    #transformamos los datos
    X = preprocessor.fit_transform(df_clean)

    #entreamos el modelo Isolation Forest
    modelo = IsolationForest(random_state=42)

    #predecimos
    preds = modelo.fit_predict(X)

    #agragamos la columna resultado finalmente
    df_clean["es_anomalia"] = preds

    return df_clean
