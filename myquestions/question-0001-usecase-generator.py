import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_detectar_filas_constantes():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función detectar_filas_constantes.
    """

    # 1. Configuración aleatoria
    n_rows = random.randint(6, 12)
    n_cols = random.randint(2, 4)

    data = np.random.randint(0, 10, size=(n_rows, n_cols))
    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(n_cols)])

    # Forzar algunas filas constantes
    n_const = random.randint(1, 3)
    for i in random.sample(range(n_rows), n_const):
        val = random.randint(0, 5)
        df.loc[i] = val

    # ---------------------------------------------------------
    # INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy()
    }

    # ---------------------------------------------------------
    # OUTPUT (Ground Truth)
    # ---------------------------------------------------------
    df_num = df.select_dtypes(include=[np.number])

    indices = [
        i for i, row in df_num.iterrows()
        if len(set(row.values)) == 1
    ]

    output_data = indices

    return input_data, output_data


if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_detectar_filas_constantes()

    print("=== INPUT ===")
    print(entrada["df"])

    print("\n=== OUTPUT ===")
    print(salida)
