import streamlit as st
import pandas as pd
import pickle
from datetime import datetime


# Importar el modelo desde un archivo
with open('modelo_knn.pkl', 'rb') as archivo:
    knn = pickle.load(archivo)

# Importar el diccionario de incumplimiento, que tiene todas las especialidades y su tasa de incumplimiento
with open('incumpl_dicc.pkl', 'rb') as archivo:
    incumpl_dicc = pickle.load(archivo)


# Título WebApp
st.title('Predictor de Asistencia a Citas Médicas')

# Subtítulo
st.subheader("Ingresa los datos del paciente y la aplicación indicará si el paciente asistirá o no a la cita agendada.")

# Definimos las opciones de las otras listas desplegables
generos = ['Masculino', 'Femenino']
edades = [i for i in range(0, 101)]
especialidades = list(incumpl_dicc.keys())
tipos_afiliacion = ['Gold', 'Silver', 'Convenio']
horas = [f'{i}:00' for i in range(6, 23)] # Definimos horas disponibles

# Creamos el DataFrame vacío
df = pd.DataFrame(columns=['genero', 'tipo_afiliacion', 'dia_semana', 'decenas', 'tasa_incumpl', 'bloque_hora'])

# Creamos las listas desplegables en la interfaz
genero_seleccionado = st.selectbox('Género del paciente:', generos)
edad_seleccionada = st.selectbox('Edad del paciente:', edades)
especialidad_seleccionada = st.selectbox('Selecciona especialidad:', especialidades)
tipo_afiliacion_seleccionado = st.selectbox('Selecciona tipo de afiliación:', tipos_afiliacion)
fecha_seleccionada = st.date_input('Selecciona la fecha:', value=datetime.now())
hora_seleccionada = st.selectbox('Selecciona la hora:', horas)

# Actualizamos el DataFrame con las selecciones del usuario
df.loc[len(df)] = [genero_seleccionado, tipo_afiliacion_seleccionado, fecha_seleccionada, edad_seleccionada, especialidad_seleccionada, hora_seleccionada]

# Preparar los datos del usuario para que estén en el mismo formato que los datos utilizados para entrenar el modelo
# Género ================================================
dicc_aux = {
    'Masculino': 0,
    'Femenino': 1
}
df['genero'] = list(map(dicc_aux.get, df.genero))


# tipo_afiliacion (normalizado) ================================================
dicc_aux = {
    'Gold': 0,
    'Silver': 0.5,
    'Convenio': 1
}
df['tipo_afiliacion'] = list(map(dicc_aux.get, df.tipo_afiliacion))


# Día de la semana (dow) ================================================
df['dia_semana'] = pd.to_datetime(df['dia_semana']).dt.weekday

# Cargar el objeto ajustado
with open('scaler_minmax_dow.pkl', 'rb') as f:
    scaler_minmax = pickle.load(f)

# Utilizar el objeto cargado para transformar los datos
df['dia_semana'] = scaler_minmax.transform(df['dia_semana'].values.reshape(-1, 1))


# Decenas (Edad) ================================================
# Cargar el objeto ajustado
with open('scaler_std_decenas.pkl', 'rb') as f:
    scaler_std = pickle.load(f)

df['decenas'] = scaler_std.transform(df['decenas'].values.reshape(-1, 1))


# Bloque hora (Hora) ================================================
df['bloque_hora'] = pd.to_datetime(df['bloque_hora'], format='%H:%M').dt.hour

bloque_hora = []
for i in df.bloque_hora:
    if i > 18:
        bloque_hora.append(3)
    elif i >= 16:
        bloque_hora.append(2)
    elif i >= 15:
        bloque_hora.append(1)
    else:
        bloque_hora.append(0)

df['bloque_hora'] = bloque_hora

# Cargar el objeto ajustado
with open('scaler_minmax_bloque_hora.pkl', 'rb') as f:
    scaler_minmax = pickle.load(f)

df['bloque_hora'] = scaler_minmax.transform(df['bloque_hora'].values.reshape(-1, 1))


# Tasa incumplimiento (incumplimiento de cada especialidad) ================================================
df['tasa_incumpl'] = df['tasa_incumpl'].str.upper()
df['tasa_incumpl'] = list(map(incumpl_dicc.get, df.tasa_incumpl))  #Reemplaza la especialidad por la tasa de inclump.

# Cargar el objeto ajustado
with open('scaler_minmax_tasa_incumpl.pkl', 'rb') as f:
    scaler_minmax = pickle.load(f)

df['tasa_incumpl'] = scaler_minmax.transform(df['tasa_incumpl'].values.reshape(-1, 1))


# Agregue un botón "Predecir" a la interfaz de usuario
if st.button('Predecir'):
    try:
        # Ejecute la predicción
        prediccion = knn.predict(df)[0]
        # Mostrar la predicción en la interfaz de usuario
        if df['dia_semana'][0] <= 1.1:  # Para que no entre si el día es domingo (día 1.2 sg la normalización)
            if prediccion == 0:
                st.write('Se predice que el paciente NO asistirá a la cita.')
            elif prediccion == 1:
                st.write('Se predice que el paciente SI asistirá a la cita.')
        else:
            st.write('Domingo sin atención a público.')
    except:
        st.write('Sin datos para este caso.')
