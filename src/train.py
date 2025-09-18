from pathlib import Path
import json
import joblib

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    """
    Carga el dataset Iris, entrena un modelo de Regresión Logística,
    guarda el modelo y sus métricas de rendimiento.
    """
    # Carga y prepara los datos
    X, y = load_iris(return_X_y=True, as_frame=True)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Inicializa y entrena el modelo
    model = LogisticRegression(max_iter=400)
    model.fit(Xtr, ytr)

    # Crea el directorio para guardar los resultados si no existe
    Path("artifacts").mkdir(exist_ok=True)

    # Guarda el modelo entrenado
    joblib.dump(model, "artifacts/model.pkl")

    # Evalúa el modelo y guarda la métrica de precisión (accuracy)
    acc = accuracy_score(yte, model.predict(Xte))
    with open("artifacts/metrics.json", "w") as f:
        json.dump({"accuracy": acc}, f, indent=2)

    print(f"Entrenamiento completado. Accuracy: {acc}")


if __name__ == "__main__":
    main()