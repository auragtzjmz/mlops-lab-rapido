from sklearn.datasets import load_iris

def test_iris_has_150_rows_and_4_features():
    """
    Verifica que el dataset Iris se cargue con la forma correcta.
    """
    X, y = load_iris(return_X_y=True, as_frame=True)

    # El DataFrame de características (X) debe tener 150 filas y 4 columnas
    assert X.shape == (150, 4)

    # El vector de etiquetas (y) debe tener 150 elementos
    assert len(y) == 150