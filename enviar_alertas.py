"""Envía por correo el pronóstico a quienes se suscribieron en el tablero.

Uso:
    python enviar_alertas.py --simular   (solo muestra qué correos enviaría)
    python enviar_alertas.py             (envía de verdad)

Para enviar de verdad, define antes estas variables de entorno (no escribas la clave en el código):
    CORREO_REMITENTE = tu_correo@gmail.com
    CORREO_CLAVE     = clave de aplicación de Gmail (no tu contraseña normal)
"""
import argparse
import glob
import os
import smtplib
from email.message import EmailMessage

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

UMBRAL_OMS, UMBRAL_NORMA = 15, 50
LAGS = [1, 2, 3, 7]


def cargar_serie(ruta):
    df = pd.read_csv(ruta, parse_dates=["Fecha inicial"])
    df = df.rename(columns={"Fecha inicial": "fecha", "PM2.5": "pm25"})
    df = df[["fecha", "pm25"]].sort_values("fecha").set_index("fecha")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="D"))
    df["pm25"] = df["pm25"].interpolate(method="time")
    return df["pm25"]


def pronosticar(ruta):
    serie = cargar_serie(ruta)
    d = pd.DataFrame({"pm25": serie})
    for lag in LAGS:
        d[f"lag_{lag}"] = d["pm25"].shift(lag)
    d["media_7d"] = d["pm25"].shift(1).rolling(7).mean()
    d["dia_semana"] = d.index.dayofweek
    d["mes"] = d.index.month
    d = d.dropna()
    modelo = RandomForestRegressor(n_estimators=200, random_state=42).fit(d.drop(columns="pm25"), d["pm25"])
    manana = serie.index[-1] + pd.Timedelta(days=1)
    fila = {f"lag_{l}": serie.iloc[-l] for l in LAGS}
    fila["media_7d"] = serie.iloc[-7:].mean()
    fila["dia_semana"], fila["mes"] = manana.dayofweek, manana.month
    return float(modelo.predict(pd.DataFrame([fila]))[0]), manana


def nivel(valor):
    if valor <= UMBRAL_OMS:
        return "Bueno"
    return "Moderado" if valor <= UMBRAL_NORMA else "Alto"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simular", action="store_true")
    simular = ap.parse_args().simular

    if not os.path.exists("suscripciones.csv"):
        print("Todavía no hay suscripciones.")
        return
    subs = pd.read_csv("suscripciones.csv").drop_duplicates(["correo", "estacion"], keep="last")
    archivos = {os.path.basename(a).replace("_pm25.csv", "").replace("_", " "): a
                for a in glob.glob("data/*_pm25.csv")}

    cache, mensajes = {}, []
    for _, s in subs.iterrows():
        if s["estacion"] not in archivos:
            continue
        if s["estacion"] not in cache:
            cache[s["estacion"]] = pronosticar(archivos[s["estacion"]])
        valor, fecha = cache[s["estacion"]]
        n = nivel(valor)
        if s["nivel"] == "Solo Alto" and n != "Alto":
            continue
        if n == "Bueno":
            continue
        msg = EmailMessage()
        msg["Subject"] = f"Alerta de calidad del aire: {n} en {s['estacion']}"
        msg["To"] = s["correo"]
        msg.set_content(f"Pronóstico de PM2.5 para el {fecha:%d-%m-%Y} en {s['estacion']}: {valor:.1f} µg/m³ ({n}).\n\n"
                        "Prueba de concepto académica (ETITC), no es un sistema de alertas oficial.")
        mensajes.append(msg)

    print(f"Correos a enviar: {len(mensajes)}")
    if simular:
        for m in mensajes:
            print("-", m["To"], "|", m["Subject"])
        return
    remitente, clave = os.environ["CORREO_REMITENTE"], os.environ["CORREO_CLAVE"]
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as srv:
        srv.login(remitente, clave)
        for m in mensajes:
            m["From"] = remitente
            srv.send_message(m)
    print("Listo.")


if __name__ == "__main__":
    main()