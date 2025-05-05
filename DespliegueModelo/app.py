import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import pickle

filename = 'modelo-clas-hiper.pkl'
best_rf_model,modelNN,labelencoder,variables,min_max_scaler = pickle.load(open(filename, 'rb'))

st.title('Predicción de Riesgo de Depresión en Estudiantes')

# Entradas del usuario
age = st.slider('Edad', min_value=15, max_value=40, value=20)
academic_pressure = st.selectbox('Presión académica', [0, 1])
study_satisfaction = st.selectbox('Satisfacción con el estudio', [0, 1])
work_study_hours = st.slider('Horas de estudio o trabajo por día', min_value=0, max_value=16, value=4)
financial_stress = st.selectbox('Estrés financiero', [0, 1])

# Variables categóricas ya codificadas como dummies
sleep_less_5 = st.checkbox('Duerme menos de 5 horas')
sleep_more_8 = st.checkbox('Duerme más de 8 horas')
diet_healthy = st.checkbox('Dieta saludable')
diet_unhealthy = st.checkbox('Dieta no saludable')
degree_class12 = st.checkbox('Estudia Class 12')
suicidal_thoughts = st.checkbox('Ha tenido pensamientos suicidas')
family_history = st.checkbox('Antecedentes familiares de enfermedad mental')

# Crear DataFrame con los datos en el mismo orden y nombres que el modelo espera
datos = pd.DataFrame([[
    age,
    academic_pressure,
    study_satisfaction,
    work_study_hours,
    financial_stress,
    int(sleep_less_5),
    int(sleep_more_8),
    int(diet_healthy),
    int(diet_unhealthy),
    int(degree_class12),
    int(suicidal_thoughts),
    int(family_history)
]], columns=[
    'Age', 'Academic Pressure', 'Study Satisfaction',
    'Work/Study Hours', 'Financial Stress',
    'Sleep Duration_Less than 5 hours',
    'Sleep Duration_More than 8 hours',
    'Dietary Habits_Healthy', 'Dietary Habits_Unhealthy',
    'Degree_Class 12',
    'Have you ever had suicidal thoughts ?_Yes',
    'Family History of Mental Illness_Yes'
])

# Botón para hacer la predicción
if st.button('Predecir'):
    pred = best_rf_model.predict(datos)[0]
    resultado = 'Con Riesgo de Depresión' if pred == 1 else 'Sin Riesgo de Depresión'
    st.subheader(f'Resultado: {resultado}')