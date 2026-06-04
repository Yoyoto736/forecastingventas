import os

# Modelo demo simple que acepta cualquier número de features y devuelve ceros
class DemoModel:
    def predict(self, X):
        import numpy as np
        X = np.asarray(X)
        return np.zeros(X.shape[0])

os.makedirs("models", exist_ok=True)
path = os.path.join("models", "modelo_final.joblib")

try:
    import joblib
    joblib.dump(DemoModel(), path)
    print("Saved demo model (joblib) to", path)
except Exception:
    import pickle
    with open(path, "wb") as f:
        pickle.dump(DemoModel(), f)
    print("Saved demo model (pickle) to", path)
