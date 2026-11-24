CALIDAD DEL AIRE EN BOGOTÁ: PREDICCIÓN DE PM2.5
Electiva Técnica I: Ciencia de Datos - ETITC
Autor: Jhon Alejandro Serrano Solorzano
Profesor: Elias Buitrago Bolivar

DESCRIPCIÓN
Predice el PM2.5 del día siguiente en 3 estaciones de la red de monitoreo de
calidad del aire de Bogotá (RMCAB): Puente Aranda, Tunal y Fontibón. Incluye un
tablero web hecho con Streamlit.

CONTENIDO
- notebooks/analisis_pm25.ipynb : limpieza, exploración, modelos y resultados (Google Colab).
- app.py                        : tablero en Streamlit.
- data/                         : series diarias de PM2.5 de las 3 estaciones (NOMBRE_pm25.csv).
- requirements.txt              : librerías necesarias.

REQUISITOS
Python 3.10 o superior.

CÓMO REPRODUCIR
1. Instalar las librerías:
   pip install -r requirements.txt
2. Análisis y modelos: abrir el notebook en Google Colab, subir los 3 CSV de la
   carpeta data/ y ejecutar las celdas en orden.
3. Tablero: desde esta carpeta ejecutar
   streamlit run app.py
   Se abre en el navegador (normalmente http://localhost:8501).

DATOS
Series diarias de PM2.5 (µg/m³) de la RMCAB, Secretaría Distrital de Ambiente.
Periodos: Puente Aranda sep-2016 a feb-2021, Tunal ene-2015 a feb-2021,
Fontibón ene-2019 a feb-2021.

NOTAS
Los días sin dato se rellenan por interpolación en el tiempo.
La división de datos es temporal (80 % entrenamiento, 20 % prueba).
Prueba de concepto académica, no es un sistema de alertas oficial.
