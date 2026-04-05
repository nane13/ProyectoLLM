import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import VarianceThreshold

def generar_caso_de_uso_detectar_columnas_baja_cardinalidad():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función detectar_columnas_baja_cardinalidad.
    """

    n_rows = random.randint(6, 12)

    n_cols = random.randint(3, 5)

    data = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        
        tipo = random.choice(["baja", "media", "alta"])
        
        if tipo == "baja":
            valores = np.random.choice(range(3), size=n_rows)
        
        elif tipo == "media":
            valores = np.random.choice(range(10), size=n_rows)
        
        else:  # alta cardinalidad
            valores = np.random.randint(0, 100, size=n_rows)
        
        data[col_name] = valores

    df = pd.DataFrame(data)

    umbral = random.randint(2, 4)

    if all(df[col].nunique() >= umbral for col in df.columns):
        col_random = random.choice(df.columns)
        df[col_random] = np.random.choice([1, 1, 2], size=n_rows)

    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "umbral": umbral
    }

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------
    cols_baja_card = [
        col for col in df.columns
        if df[col].nunique() < umbral
    ]

    # (Paso sklearn adicional, aunque no afecta output final)
    df_num = df.select_dtypes(include=[np.number])
    if not df_num.empty:
        selector = VarianceThreshold()
        selector.fit(df_num)

    output_data = cols_baja_card

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_detectar_columnas_baja_cardinalidad()

    print("=== INPUT ===")
    print(entrada["df"])
    print("Umbral:", entrada["umbral"])

    print("\n=== OUTPUT ===")
    print(salida)
