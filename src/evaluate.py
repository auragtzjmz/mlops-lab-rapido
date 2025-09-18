import json
import sys

# Umbral mínimo de precisión aceptable.
# Si la precisión del modelo es menor, el script fallará.
MIN_ACC = 0.90


def main():
    """
    Lee las métricas de un archivo JSON y verifica si la precisión (accuracy)
    supera el umbral mínimo definido.
    """
    try:
        with open("artifacts/metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("FAIL: No se encontró el archivo 'artifacts/metrics.json'.")
        sys.exit(1) # Termina el script con un código de error

    # Obtiene el valor de 'accuracy', si no existe, devuelve 0.0
    acc = metrics.get("accuracy", 0.0)

    print(f"Evaluación -> Accuracy encontrado: {acc:.4f}")

    # Compara la precisión del modelo con el umbral mínimo
    if acc < MIN_ACC:
        print(f"FAIL: La precisión ({acc:.4f}) está por debajo del umbral ({MIN_ACC}).")
        sys.exit(1) # Termina el script con un código de error

    print(f"Evaluación OK 👍: La precisión supera el umbral de {MIN_ACC}.")


if __name__ == "__main__":
    main()