import os
import numpy as np

# Creamos un modelo robusto usando scikit-learn DummyRegressor
os.makedirs("models", exist_ok=True)
path = os.path.join("models", "modelo_final.joblib")

try:
    from sklearn.dummy import DummyRegressor
    model = DummyRegressor(strategy="constant", constant=0.0)
    # Ajustar con un ejemplo mínimo para asegurar atributos n_features_in_
    model.fit(np.zeros((1, 1)), [0.0])
    import joblib
    joblib.dump(model, path)
    print("Saved demo DummyRegressor (joblib) to", path)
except Exception as e:
    # Fallback a pickle si joblib no funciona
    try:
        import joblib
        joblib.dump(model, path)
        print("Saved demo model (joblib fallback) to", path)
    except Exception:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print("Saved demo model (pickle fallback) to", path)
