import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_detectar_columnas_baja_cardinalidad():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función detectar_columnas_baja_cardinalidad.
    """

    n_rows = random.randint(6, 12)
    n_cols = random.randint(3, 5)

    # ---------------------------------------------------------
    # 1. Generar DataFrame aleatorio
    # ---------------------------------------------------------
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

    # Asegurar que haya al menos una columna de baja cardinalidad
    if all(df[col].nunique() >= umbral for col in df.columns):
        col_random = random.choice(df.columns)
        df[col_random] = np.random.choice([1, 1, 2], size=n_rows)

    # ---------------------------------------------------------
    # 2. INPUT
    # ---------------------------------------------------------
    input_data = {
        "df": df.copy(),
        "umbral": umbral
    }

    # ---------------------------------------------------------
    # 3. OUTPUT (Ground Truth)
    # ---------------------------------------------------------

    # A. Columnas con baja cardinalidad
    cols_baja_card = [
        col for col in df.columns
        if df[col].nunique() < umbral
    ]

    # B. Columnas numéricas con varianza cero
    df_num = df.select_dtypes(include=[np.number])

    cols_var_cero = []
    if not df_num.empty:
        varianzas = df_num.var()
        cols_var_cero = varianzas[varianzas == 0].index.tolist()

    # C. Unión de ambas condiciones
    output_data = list(set(cols_baja_card + cols_var_cero))

    return input_data, output_data


# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_detectar_columnas_baja_cardinalidad()

    print("=== INPUT ===")
    print(entrada["df"])
    print("Umbral:", entrada["umbral"])

    print("\n=== OUTPUT ===")
    print(salida)
