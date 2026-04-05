import pandas as pd
import numpy as np
import random
from sklearn.feature_selection import VarianceThreshold

def generar_caso_de_uso_columna_mayor_varianza():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función columna_mayor_varianza.
    """

    n_rows = random.randint(6, 12)
    n_cols = random.randint(3, 5)

    data = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        
        std = random.uniform(0.5, 5.0)  # desviación aleatoria
        valores = np.random.normal(0, std, n_rows)
        
        data[col_name] = valores

    df = pd.DataFrame(data)

    while True:
        stds = [random.uniform(0.5, 5.0) for _ in range(n_cols)]
        if len(set(round(s, 2) for s in stds)) == n_cols:
            break

    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy()
    }

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------
    df_num = df.select_dtypes(include=[np.number])

    selector = VarianceThreshold()
    selector.fit(df_num)

    varianzas = df_num.var()
    col_max = varianzas.idxmax()

    output_data = col_max

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_columna_mayor_varianza()

    print("=== INPUT ===")
    print(entrada["df"])

    print("\n=== OUTPUT ===")
    print("Columna con mayor varianza:", salida)
