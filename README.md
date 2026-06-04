# Simulador de Ventas — Noviembre 2025

App de Streamlit para simular predicciones de ventas día a día para noviembre 2025.

Estructura:

- `app/app.py` — aplicación Streamlit.
- `data/processed/inferencia_df_transformado.csv` — datos de inferencia (noviembre 2025).
- `models/` — modelos guardados (`modelo_final.joblib`).
- `requirements.txt` — dependencias.

Ejecución local:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Notas:

- El CSV ya contiene lags calculados desde octubre; la app realiza predicciones recursivas día a día.
- Ajusta `requirements.txt` con versiones fijas si lo deseas.

Licencia: MIT (añade tu propia licencia si procede)
