import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import traceback

# Intento de carga segura de joblib (puede venir con sklearn o como paquete independiente)
try:
	import joblib
except Exception:
	try:
		from sklearn.externals import joblib  # type: ignore
	except Exception:
		joblib = None


def load_data(path: Path):
	if not path.exists():
		raise FileNotFoundError(f"No se encuentra el fichero de inferencia: {path}")
	df = pd.read_csv(path, parse_dates=["fecha"]) if "fecha" in pd.read_csv(path, nrows=0).columns else pd.read_csv(path)
	return df


def load_model(candidates):
	for p in candidates:
		p = Path(p)
		if p.exists():
			if joblib is not None:
				try:
					m = joblib.load(p)
					return m, p
				except Exception:
					# intentar pickle como última opción
					try:
						import pickle

						with open(p, "rb") as f:
							m = pickle.load(f)
						return m, p
					except Exception:
						continue
			else:
				try:
					import pickle

					with open(p, "rb") as f:
						m = pickle.load(f)
					return m, p
				except Exception:
					continue
	return None, None


def find_column(df, base_names):
	for b in base_names:
		if b in df.columns:
			return b
	return None


def simulate_product(df_prod, model, discount_pct, comp_multiplier):
	# Copia local
	dfp = df_prod.copy().sort_values("fecha").reset_index(drop=True)

	# Normalizar y elegir columnas de competencia (varian nombres en el CSV)
	for comp in ["Amazon", "Decathlon", "Deporvillage"]:
		candidate = find_column(dfp, [comp, comp + "_x", comp + "_y"])
		if candidate is not None:
			dfp[comp] = pd.to_numeric(dfp[candidate], errors="coerce")
		else:
			dfp[comp] = np.nan

	# Aplicar escenario de competencia
	dfp[["Amazon", "Decathlon", "Deporvillage"]] = dfp[["Amazon", "Decathlon", "Deporvillage"]] * comp_multiplier

	# Recalcular precio_venta según descuento (descuento_pct puede ser negativo)
	dfp["precio_base"] = pd.to_numeric(dfp["precio_base"], errors="coerce").fillna(0.0)
	dfp["precio_venta"] = (dfp["precio_base"] * (1 - discount_pct / 100)).astype(float)
	dfp["descuento_porcentaje"] = discount_pct

	# Recalcular precio_competencia (media de competidores existentes)
	dfp["precio_competencia"] = dfp[["Amazon", "Decathlon", "Deporvillage"]].mean(axis=1, skipna=True)
	# Evitar division por cero
	dfp["ratio_precio"] = dfp["precio_venta"] / dfp["precio_competencia"].replace({0: np.nan})

	# Asegurar columnas de lag
	lag_cols = [f"unidades_vendidas_lag_{i}" for i in range(1, 8)]
	for c in lag_cols:
		if c not in dfp.columns:
			dfp[c] = 0.0
		dfp[c] = pd.to_numeric(dfp[c], errors="coerce").fillna(0.0).astype(float)

	# Media móvil: preferimos 'unidades_vendidas_ma7', si no existe usamos 'unidades_vendidas_roll_7'
	if "unidades_vendidas_ma7" not in dfp.columns and "unidades_vendidas_roll_7" in dfp.columns:
		dfp["unidades_vendidas_ma7"] = pd.to_numeric(dfp["unidades_vendidas_roll_7"], errors="coerce").fillna(0.0)
	elif "unidades_vendidas_ma7" not in dfp.columns:
		# si no hay ninguna, la calculamos a partir de los lags
		dfp["unidades_vendidas_ma7"] = dfp[lag_cols].mean(axis=1)
	else:
		dfp["unidades_vendidas_ma7"] = pd.to_numeric(dfp["unidades_vendidas_ma7"], errors="coerce").fillna(0.0)

	# Obtener lista de features esperadas por el modelo
	model_features = None
	try:
		model_features = list(getattr(model, "feature_names_in_").tolist())
	except Exception:
		try:
			model_features = list(getattr(model, "feature_names_in_"))
		except Exception:
			model_features = []

	# Si no hay feature_names_in_ asumimos que el CSV ya contiene las mismas columnas en orden
	if model_features:
		missing = [f for f in model_features if f not in dfp.columns]
		# Intentos automáticos de alias (ej. unidades_vendidas_roll_7 -> unidades_vendidas_ma7)
		if "unidades_vendidas_ma7" in model_features and "unidades_vendidas_ma7" not in dfp.columns and "unidades_vendidas_roll_7" in dfp.columns:
			dfp["unidades_vendidas_ma7"] = dfp["unidades_vendidas_roll_7"]
			missing = [f for f in model_features if f not in dfp.columns]
		if missing:
			raise KeyError(f"Faltan columnas para el modelo: {missing}")

	# Preparar estructura de predicción recursiva
	# Tomamos los lags del primer día (día 1) tal como vienen en el fichero
	first = dfp.iloc[0]
	last_7 = []
	for c in lag_cols:
		v = first.get(c, 0.0)
		try:
			last_7.append(float(v))
		except Exception:
			last_7.append(0.0)

	results = []

	for idx, row in dfp.iterrows():
		r = row.copy()
		if idx == 0:
			# día 1: usar lags tal cual
			r["unidades_vendidas_ma7"] = float(np.mean(last_7))
		else:
			# Actualizar lags según last_7 (lag_1 = más reciente)
			for j in range(1, 8):
				r[f"unidades_vendidas_lag_{j}"] = float(last_7[j - 1])
			r["unidades_vendidas_ma7"] = float(np.mean(last_7))

		# Preparar array de features para el modelo
		if model_features:
			Xs = r[model_features].copy()
			# coerción segura a numérico (booleans/strings -> 1/0 o NaN)
			Xs = Xs.apply(lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])
			X = np.array([Xs.fillna(0.0).astype(float)]).reshape(1, -1)
		else:
			# si no conocemos las features, usamos todas las columnas numéricas
			numeric_cols = r.index[r.apply(lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").notna().iloc[0])].tolist()
			Xs = r[numeric_cols].apply(lambda v: pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])
			X = np.array([Xs.fillna(0.0).astype(float)]).reshape(1, -1)

		pred = float(model.predict(X)[0])

		# Guardar predicción y actualizar last_7
		results.append({
			"fecha": r["fecha"],
			"dia_mes": int(r.get("dia_mes", pd.to_datetime(r["fecha"]).day)),
			"dia_semana": r.get("dia_semana", ""),
			"precio_venta": float(r.get("precio_venta", 0.0)),
			"precio_competencia": float(r.get("precio_competencia", np.nan)),
			"descuento_porcentaje": float(r.get("descuento_porcentaje", discount_pct)),
			"unidades_pred": pred,
			"ingresos_pred": float(r.get("precio_venta", 0.0)) * pred,
			"black_friday": bool(r.get("black_friday", False)),
		})

		last_7 = [pred] + last_7[:-1]

	out = pd.DataFrame(results)
	return out


def main():
	st.set_page_config(page_title="Simulación Ventas — Nov 2025", layout="wide")
	sns.set_theme(style="whitegrid")
	PRIMARY = "#667eea"
	SECONDARY = "#764ba2"

	st.markdown(f"# 🤖 Simulador de Ventas — Noviembre 2025")

	# Cargar datos y modelo con manejo de errores
	# Resolver ruta del proyecto (un nivel arriba de app/)
	base_dir = Path(__file__).resolve().parents[1]
	data_path = base_dir.joinpath("data", "processed", "inferencia_df_transformado.csv")
	model_candidates = [
		base_dir.joinpath("models", "modelo_final.joblib"),
		base_dir.joinpath("models", "model_full_df.joblib"),
		base_dir.joinpath("models", "modelo_final.pkl"),
		base_dir.joinpath("models", "modelo_full.joblib"),
	]

	try:
		df = load_data(data_path)
	except Exception as e:
		st.error(f"Error al cargar datos: {e}")
		st.stop()

	model, model_path = load_model(model_candidates)
	if model is None:
		st.error("No se ha podido cargar el modelo. Busqué: " + ", ".join([str(p) for p in model_candidates]))
		st.stop()

	# Sidebar - controles
	st.sidebar.header("Controles de Simulación")
	productos = sorted(df["nombre"].dropna().unique().tolist())
	producto = st.sidebar.selectbox("Producto", productos)
	descuento = st.sidebar.slider("Ajuste de descuento (%)", -50, 50, 0, step=5)
	escenario = st.sidebar.radio("Escenario de competencia", ["Actual (0%)", "Competencia -5%", "Competencia +5%"], index=0)
	simular = st.sidebar.button("Simular Ventas")

	st.sidebar.markdown("---")
	st.sidebar.write(f"Modelo cargado: {model_path}")

	if not simular:
		st.info("Elige controles en la barra lateral y pulsa 'Simular Ventas' para generar la predicción.")
		st.stop()

	# Mapear escenario a multiplicador de competencia
	mult = 1.0
	if escenario == "Competencia -5%":
		mult = 0.95
	elif escenario == "Competencia +5%":
		mult = 1.05

	# Filtrar producto
	df_prod = df[df["nombre"] == producto].copy()
	if df_prod.empty:
		st.error("No hay datos de inferencia para el producto seleccionado.")
		st.stop()

	# Ejecutar simulación (y comparativa de escenarios)
	with st.spinner("Simulando ventas día a día (predicciones recursivas)..."):
		try:
			# Escenario elegido
			res_main = simulate_product(df_prod, model, descuento, mult)
			# Comparativa: sin cambio, -5%, +5% (manteniendo descuento)
			res_baseline = simulate_product(df_prod, model, descuento, 1.0)
			res_minus = simulate_product(df_prod, model, descuento, 0.95)
			res_plus = simulate_product(df_prod, model, descuento, 1.05)
		except KeyError as e:
			st.error(f"Faltan columnas requeridas por el modelo: {e}")
			st.stop()
		except Exception as e:
			st.error(f"Error durante la simulación: {e}\n{traceback.format_exc()}")
			st.stop()

	# KPIs
	total_unidades = res_main["unidades_pred"].sum()
	total_ingresos = res_main["ingresos_pred"].sum()
	precio_promedio = (res_main["precio_venta"] * res_main["unidades_pred"]).sum() / max(total_unidades, 1)
	descuento_promedio = res_main["descuento_porcentaje"].mean()

	st.markdown(f"## 📈 Resultados: {producto}")
	k1, k2, k3, k4 = st.columns(4)
	k1.metric("Unidades totales proyectadas", f"{total_unidades:,.0f}")
	k2.metric("Ingresos proyectados", f"€{total_ingresos:,.2f}")
	k3.metric("Precio promedio de venta", f"€{precio_promedio:,.2f}")
	k4.metric("Descuento promedio", f"{descuento_promedio:,.1f}%")

	st.markdown("---")

	# Gráfico de predicción diaria
	fig, ax = plt.subplots(figsize=(12, 4))
	sns.lineplot(data=res_main, x="dia_mes", y="unidades_pred", marker="o", color=PRIMARY, ax=ax)
	ax.set_title(f"Unidades vendidas proyectadas — {producto} (Noviembre 2025)")
	ax.set_xlabel("Día del mes")
	ax.set_ylabel("Unidades vendidas")
	ax.set_xticks(res_main["dia_mes"])
	# Marcar Black Friday (día 28)
	bf_rows = res_main[res_main["black_friday"] == True]
	if bf_rows.shape[0] == 0:
		# intentar día 28
		bf_day = 28
		bf_idx = res_main[res_main["dia_mes"] == bf_day]
	else:
		bf_idx = bf_rows

	if not bf_idx.empty:
		bf_day = int(bf_idx.iloc[0]["dia_mes"])
		bf_val = float(bf_idx.iloc[0]["unidades_pred"])
		ax.axvline(bf_day, color="gray", linestyle="--", linewidth=1)
		ax.scatter([bf_day], [bf_val], color="red", s=80, zorder=5)
		ax.annotate("🛍️ Black Friday", xy=(bf_day, bf_val), xytext=(bf_day + 1, bf_val * 1.05), color="red")

	st.pyplot(fig)

	st.markdown("---")

	# Tabla detallada
	df_table = res_main.copy()
	df_table = df_table[["fecha", "dia_mes", "dia_semana", "precio_venta", "precio_competencia", "descuento_porcentaje", "unidades_pred", "ingresos_pred", "black_friday"]]
	# Añadir nota para Black Friday (emoji)
	df_table["nota"] = df_table["black_friday"].apply(lambda x: "🔥 Black Friday" if x else "")

	# Formatear para visualización (columna nota visible para destacar)
	df_table_display = df_table.copy()
	df_table_display["precio_venta"] = df_table_display["precio_venta"].map(lambda x: f"€{x:,.2f}")
	df_table_display["precio_competencia"] = df_table_display["precio_competencia"].map(lambda x: f"€{x:,.2f}" if pd.notna(x) else "–")
	df_table_display["descuento_porcentaje"] = df_table_display["descuento_porcentaje"].map(lambda x: f"{x:.0f}%")
	df_table_display["unidades_pred"] = df_table_display["unidades_pred"].map(lambda x: f"{x:,.0f}")
	df_table_display["ingresos_pred"] = df_table_display["ingresos_pred"].map(lambda x: f"€{x:,.2f}")

	st.subheader("📋 Detalle diario")
	st.dataframe(df_table_display, height=400)

	st.markdown("---")

	# Comparativa de escenarios
	def summarize(res):
		return {"unidades": res["unidades_pred"].sum(), "ingresos": res["ingresos_pred"].sum()}

	s_base = summarize(res_baseline)
	s_minus = summarize(res_minus)
	s_plus = summarize(res_plus)

	st.subheader("🔁 Comparativa de escenarios (competencia)")
	c1, c2, c3 = st.columns(3)
	c1.metric("Actual (0%) — Unidades", f"{s_base['unidades']:,.0f}")
	c1.metric("Actual (0%) — Ingresos", f"€{s_base['ingresos']:,.2f}")
	c2.metric("Competencia -5% — Unidades", f"{s_minus['unidades']:,.0f}")
	c2.metric("Competencia -5% — Ingresos", f"€{s_minus['ingresos']:,.2f}")
	c3.metric("Competencia +5% — Unidades", f"{s_plus['unidades']:,.0f}")
	c3.metric("Competencia +5% — Ingresos", f"€{s_plus['ingresos']:,.2f}")

	st.markdown("---")
	st.info("Predicción generada con actualización recursiva de lags día a día. Valores redondeados para presentación.")


if __name__ == "__main__":
	main()

