import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Calidad del aire Bogotá", layout="wide")

UMBRAL_OMS = 15     # guía OMS 2021, promedio de 24 h (µg/m³)
UMBRAL_NORMA = 50   # norma diaria nacional citada en el artículo (µg/m³)
LAGS = [1, 2, 3, 7]


# ---------- Datos ----------
@st.cache_data
def listar_estaciones():
    """Busca data/*_pm25.csv. El nombre de la estación sale del nombre del archivo."""
    archivos = sorted(glob.glob("data/*_pm25.csv"))
    return {os.path.basename(a).replace("_pm25.csv", "").replace("_", " "): a for a in archivos}


@st.cache_data
def cargar_serie(ruta):
    df = pd.read_csv(ruta, parse_dates=["Fecha inicial"])
    df = df.rename(columns={"Fecha inicial": "fecha", "PM2.5": "pm25"})
    df = df[["fecha", "pm25"]].sort_values("fecha").set_index("fecha")
    idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(idx)
    faltantes = int(df["pm25"].isna().sum())
    df["pm25"] = df["pm25"].interpolate(method="time")
    return df["pm25"], faltantes


def crear_variables(serie):
    d = pd.DataFrame({"pm25": serie})
    for lag in LAGS:
        d[f"lag_{lag}"] = d["pm25"].shift(lag)
    d["media_7d"] = d["pm25"].shift(1).rolling(7).mean()
    d["dia_semana"] = d.index.dayofweek
    d["mes"] = d.index.month
    return d


def variables_manana(serie):
    manana = serie.index[-1] + pd.Timedelta(days=1)
    fila = {f"lag_{l}": serie.iloc[-l] for l in LAGS}
    fila["media_7d"] = serie.iloc[-7:].mean()
    fila["dia_semana"] = manana.dayofweek
    fila["mes"] = manana.month
    return pd.DataFrame([fila]), manana


# ---------- Modelos ----------
@st.cache_resource
def entrenar_todo(ruta):
    serie, _ = cargar_serie(ruta)
    datos = crear_variables(serie).dropna()
    X, y = datos.drop(columns="pm25"), datos["pm25"]
    corte = int(len(datos) * 0.8)  # división temporal: 80 % pasado, 20 % futuro
    X_tr, X_te, y_tr, y_te = X.iloc[:corte], X.iloc[corte:], y.iloc[:corte], y.iloc[corte:]

    def nuevo(nombre):
        if nombre == "Regresión lineal":
            return LinearRegression()
        return RandomForestRegressor(n_estimators=200, random_state=42)

    filas, finales = [], {"Línea base (valor de ayer)": None}
    pred = X_te["lag_1"].values
    filas.append({"Modelo": "Línea base (valor de ayer)",
                  "MAE": mean_absolute_error(y_te, pred),
                  "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
                  "R²": r2_score(y_te, pred)})
    for nombre in ["Regresión lineal", "Random Forest"]:
        m = nuevo(nombre).fit(X_tr, y_tr)
        pred = m.predict(X_te)
        filas.append({"Modelo": nombre,
                      "MAE": mean_absolute_error(y_te, pred),
                      "RMSE": np.sqrt(mean_squared_error(y_te, pred)),
                      "R²": r2_score(y_te, pred)})
        finales[nombre] = nuevo(nombre).fit(X, y)  # reentrenado con todos los datos
    return pd.DataFrame(filas), finales


def nivel_alerta(valor):
    if valor <= UMBRAL_OMS:
        return "Bueno", "#1E5C42", "#CDEBDD"
    if valor <= UMBRAL_NORMA:
        return "Moderado", "#5C4200", "#F7E3B5"
    return "Alto", "#7A1717", "#F6CFCF"


# ---------- Interfaz ----------
estaciones = listar_estaciones()
if not estaciones:
    st.error("No encontré archivos en la carpeta data/ con el formato NOMBRE_pm25.csv")
    st.stop()

# Escudo de la universidad en la esquina de la barra lateral
import base64
LOGO_B64 = "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"
try:
    st.sidebar.image(base64.b64decode(LOGO_B64), width=90)
except Exception:
    pas  # si el archivo está dañado, el tablero sigue funcionando sin el escudo
st.sidebar.title("Calidad del aire")
st.sidebar.caption("Bogotá · PM2.5 diario")
estacion = st.sidebar.selectbox("Estación", list(estaciones))
modelo_sel = st.sidebar.selectbox("Modelo", ["Random Forest", "Regresión lineal", "Línea base (valor de ayer)"])
dias = st.sidebar.slider("Días de histórico a mostrar", 30, 365, 60)

# Registro de correo para recibir alertas
st.sidebar.divider()
with st.sidebar.form("suscripcion", clear_on_submit=True):
    st.markdown("**Alertas por correo**")
    correo = st.text_input("Tu correo")
    nivel_min = st.selectbox("Avisarme cuando el pronóstico sea", ["Moderado o Alto", "Solo Alto"])
    acepto = st.checkbox("Acepto que se guarde mi correo para enviarme alertas")
    enviar = st.form_submit_button("Suscribirme")
if enviar:
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", correo.strip()):
        st.sidebar.error("Escribe un correo válido.")
    elif not acepto:
        st.sidebar.warning("Debes aceptar para poder suscribirte.")
    else:
        nueva = pd.DataFrame([{"fecha": datetime.now().isoformat(timespec="seconds"),
                               "correo": correo.strip(), "estacion": estacion, "nivel": nivel_min}])
        nueva.to_csv("suscripciones.csv", mode="a", header=not os.path.exists("suscripciones.csv"), index=False)
        st.sidebar.success("¡Listo! Quedaste suscrito.")

st.sidebar.caption("Fuente: RMCAB (Secretaría Distrital de Ambiente). Prueba de concepto académica, "
                   "no es un sistema de alertas oficial.")

ruta = estaciones[estacion]
serie, faltantes = cargar_serie(ruta)
tabla, modelos = entrenar_todo(ruta)

X_man, fecha_man = variables_manana(serie)
modelo = modelos[modelo_sel]
pronostico = float(X_man["lag_1"].iloc[0]) if modelo is None else float(modelo.predict(X_man)[0])
nombre_nivel, color_txt, color_fondo = nivel_alerta(pronostico)

st.title("Pronóstico de PM2.5 para el día siguiente")
st.caption(f"Estación {estacion} · el pronóstico es para el {fecha_man:%d-%m-%Y}, "
           f"el día siguiente al último dato del archivo ({serie.index[-1]:%d-%m-%Y}).")

c1, c2, c3 = st.columns(3)
c1.metric("PM2.5 pronosticado (µg/m³)", f"{pronostico:.1f}")
c2.markdown("**Nivel de alerta**")
c2.markdown(f"<span style='background:{color_fondo};color:{color_txt};padding:6px 14px;"
            f"border-radius:999px;font-weight:700'>{nombre_nivel}</span>", unsafe_allow_html=True)
c2.caption(f"Bueno: hasta {UMBRAL_OMS} · Moderado: hasta {UMBRAL_NORMA} · Alto: más de {UMBRAL_NORMA}")
c3.metric("Último dato observado (µg/m³)", f"{serie.iloc[-1]:.1f}")

# Gráfica: histórico + pronóstico + umbrales
hist = serie.iloc[-dias:]
grafica = pd.DataFrame({"PM2.5 observado": hist})
grafica.loc[fecha_man, "Pronóstico"] = pronostico
grafica.loc[hist.index[-1], "Pronóstico"] = hist.iloc[-1]  # une la línea con el último dato
grafica["Guía OMS (15)"] = UMBRAL_OMS
grafica["Norma (50)"] = UMBRAL_NORMA
st.subheader("Histórico y pronóstico")
st.line_chart(grafica)

izq, der = st.columns(2)
with izq:
    st.subheader("Comparar estaciones")
    if len(estaciones) > 1:
        prom = {n: cargar_serie(r)[0].iloc[-30:].mean() for n, r in estaciones.items()}
        st.bar_chart(pd.Series(prom, name="Promedio últimos 30 días (µg/m³)"))
    else:
        st.info("Agrega más archivos *_pm25.csv en la carpeta data/ para comparar estaciones.")
with der:
    st.subheader("Desempeño de los modelos")
    st.dataframe(tabla.round(2), hide_index=True)
    st.caption("Evaluados en el 20 % más reciente de los datos (división temporal).")

st.caption(f"Días sin dato en el archivo, rellenados por interpolación: {faltantes}")