import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

def predecir_resistencia_material(df_historico, configuracion_nueva):
    """
    Función que entrena un modelo de regresión para predecir el límite elástico de una nueva aleación experimental

    Parámetros: DataFrame con las columnas Al_porcentaje, Mg_porcentaje, Zn_porcentaje, Temp_Tratamiento y Limite_Elastico_MPa

        - configuracion_nueva: lista con los valores [Al_porcentaje, Mg_porcentaje, Zn_porcentaje, Temp_Tratamiento]

    Se espera:
        Diccionario con resistencia_predicha_mpa
    """

    #definimos columnas predictoras y el target
    columnas_x = [
        "Al_porcentaje",
        "Mg_porcentaje",
        "Zn_porcentaje",
        "Temp_Tratamiento"
    ]

    X = df_historico[columnas_x]
    y = df_historico["Limite_Elastico_MPa"]

    #escalamos las características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #entrenamos al modelo
    modelo = GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )

    modelo.fit(X_scaled, y)

    #preparamos nueva configuración
    input_df = pd.DataFrame(
        [configuracion_nueva],
        columns=columnas_x
    )

    #escalamos la nueva configuración
    input_scaled = scaler.transform(input_df)

    #predecimos resistencia
    prediccion = modelo.predict(input_scaled)[0]

    return {
        "resistencia_predicha_mpa": round(float(prediccion), 2)
    }
