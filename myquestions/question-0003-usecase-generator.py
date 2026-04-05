import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_identificar_filas_con_nan():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función identificar_filas_con_nan.
    """

    n_rows = random.randint(6, 12)
    n_cols = random.randint(2, 4)

    data = np.random.randn(n_rows, n_cols)
    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])

    # Introducir NaNs
    mask = np.random.choice([True, False], size=df.shape, p=[0.2, 0.8])
    df[mask] = np.nan

    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy()
    }

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------
    indices = df[df.isna().any(axis=1)].index.tolist()

    output_data = indices

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_identificar_filas_con_nan()

    print("=== INPUT ===")
    print(entrada["df"])

    print("\n=== OUTPUT ===")
    print(salida)
