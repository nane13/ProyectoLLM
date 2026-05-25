import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def evaluar_segmentos_regresion(df, target_col, threshold):
    """
    Función que entrena un modelo de regresión lineal y evalúa el MSE separado en dos segmentos según el valor real de la variable objetivo

    Parámetros:
        - df: DataFrame completo con columnas de features y target
        - target_col: nombre de la columna objetivo
        - threshold: valor de corte para separar grupo bajo y grupo alto

    Se espera: numpy.ndarray con [mse_grupo_bajo, mse_grupo_alto]
    """

    #separamos variables predictoras y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]

    #entrenamiento/prueba 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    #entrenamos al modelo
    model = LinearRegression()
    model.fit(X_train, y_train)

    #predecimos
    y_pred = model.predict(X_test)

    #creamos la segmentacion
    grupo_bajo = y_test <= threshold
    grupo_alto = y_test > threshold

    #calculo del MSE para cada grupo
    mse_bajo = (
        mean_squared_error(y_test[grupo_bajo], y_pred[grupo_bajo])
        if np.sum(grupo_bajo) > 0
        else 0.0
    )

    mse_alto = (
        mean_squared_error(y_test[grupo_alto], y_pred[grupo_alto])
        if np.sum(grupo_alto) > 0
        else 0.0
    )

    return np.array([float(mse_bajo), float(mse_alto)])
