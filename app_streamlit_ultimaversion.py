# -*- coding: utf-8 -*-
"""
Aplicación Streamlit para predecir morosidad de clientes
Modelos: Keras y PyTorch comparados visualmente
Autor: Lucero Manrique
"""

# =============================================
# 📦 1. Importar librerías
# =============================================
import streamlit as st
import numpy as np
import pickle
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

import torch.nn as nn
from skorch import NeuralNetClassifier
import joblib
# =============================================
# 💾 2. Cargar escalador
# =============================================
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =============================================
# 🧠 3. Definir modelo PyTorch
# =============================================
class ClassNN(nn.Module):   
    def __init__(self, input_dim=18, hidden_sizes=[32,16], activation=nn.ReLU, dropout=0.0):
        super(ClassNN, self).__init__()
        layers = []
        in_features = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, X):
        return self.net(X)
    


# =============================================
#   Cargar modelos
# =============================================

model_pytorch = joblib.load("03_modelo_pytorch_app.pkl")
model_keras = load_model("model.h5")


# =============================================
# 🧩 4. Configurar interfaz Streamlit
# =============================================
st.set_page_config(page_title="Predicción de Morosidad", page_icon="💳", layout="centered")
st.title("💳 Predicción de Morosidad de Clientes")
st.write("Ingrese los datos del cliente y seleccione el modelo para la predicción.")

# =============================================
# 🗂 Selector de modelo
# =============================================
modelo_opcion = st.selectbox("Selecciona el modelo a usar:", ("Keras", "PyTorch", "Ambos"))

# =============================================
# 🧾 Entradas del usuario
# =============================================
col1, col2 = st.columns(2)

with col1:
    LIMIT_BAL = st.number_input("Monto de crédito otorgado (NT$)", min_value=0, value=20000, step=1000)
    SEX = st.selectbox("Sexo", options=[1, 2], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
    EDUCATION = st.selectbox("Nivel educativo", options=[1, 2, 3, 4], format_func=lambda x: {
        1: "Postgrado", 2: "Universidad", 3: "Secundaria", 4: "Otros"
    }[x])
    MARRIAGE = st.selectbox("Estado civil", options=[1, 2, 3], format_func=lambda x: {
        1: "Casado", 2: "Soltero", 3: "Otros"
    }[x])
    AGE = st.slider("Edad (años)", 18, 80, 35)

with col2:
    PAY_1 = st.number_input("Estado de pago mes 1 (Sept 2005)", min_value=-1, max_value=9, value=0)
    PAY_2 = st.number_input("Estado de pago mes 2 (Ago 2005)", min_value=-1, max_value=9, value=0)
    PAY_3 = st.number_input("Estado de pago mes 3 (Jul 2005)", min_value=-1, max_value=9, value=0)
    PAY_4 = st.number_input("Estado de pago mes 4 (Jun 2005)", min_value=-1, max_value=9, value=0)
    PAY_5 = st.number_input("Estado de pago mes 5 (May 2005)", min_value=-1, max_value=9, value=0)
    PAY_6 = st.number_input("Estado de pago mes 6 (Abr 2005)", min_value=-1, max_value=9, value=0)

st.write("### 💰 Montos de facturas y pagos anteriores (en NT$)")
col_bill, col_pay = st.columns(2)
BILL_AMT1 = col_bill.number_input("Factura mes 1 (BILL_AMT1)", min_value=0, value=5000, step=1000)
PAY_AMT1 = col_pay.number_input("Pago mes 1 (PAY_AMT1)", min_value=0, value=2000, step=500)
PAY_AMT2 = col_pay.number_input("Pago mes 2 (PAY_AMT2)", min_value=0, value=4000, step=500)
PAY_AMT3 = col_pay.number_input("Pago mes 3 (PAY_AMT3)", min_value=0, value=6000, step=500)
PAY_AMT4 = col_pay.number_input("Pago mes 4 (PAY_AMT4)", min_value=0, value=8000, step=500)
PAY_AMT5 = col_pay.number_input("Pago mes 5 (PAY_AMT5)", min_value=0, value=10000, step=500)
PAY_AMT6 = col_pay.number_input("Pago mes 6 (PAY_AMT6)", min_value=0, value=12000, step=500)

# =============================================
# 🧮 Construir vector de entrada
# =============================================
input_data = np.array([[LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
                        PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
                        BILL_AMT1,
                        PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6]])
input_scaled = scaler.transform(input_data)

# 🔮 Predicción y visualización
# --- Botón de predicción ---
if st.button("🔍 Predecir probabilidad de morosidad"):

    resultados = {}

    # --- Keras ---
    if modelo_opcion in ["Keras", "Ambos"]:
        prob_keras = float(model_keras.predict(input_scaled)[0][0])
        resultados["Keras"] = prob_keras

    # --- PyTorch ---
    if modelo_opcion in ["PyTorch", "Ambos"]:
        input_scaled_2d = input_scaled.astype(np.float32).reshape(1, -1)
        proba = 1-float(model_pytorch.predict_proba(input_scaled_2d)[0, 0])
        resultados["PyTorch"] = proba

    # --- Mostrar resultados individuales ---
    for nombre, prob in resultados.items():
        estado = "Moroso" if prob > 0.5 else "No Moroso"
        color = "red" if prob > 0.5 else "green"
        st.markdown(f"### {nombre}: <span style='color:{color}'>{prob*100:.2f}% - {estado}</span>", unsafe_allow_html=True)
        st.progress(prob)

    # --- Guardar resultados en session_state para usar después ---
    st.session_state["resultados"] = resultados

# --- Checkbox para mostrar gráfico comparativo ---
if modelo_opcion == "Ambos" and "resultados" in st.session_state:
    mostrar_grafico = st.checkbox("Mostrar gráfico comparativo")
    if mostrar_grafico:
        resultados = st.session_state["resultados"]
        modelos = list(resultados.keys())
        probs = [resultados[m]*100 for m in modelos]
        colores = ["green" if p <= 50 else "red" for p in probs]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(modelos, probs, color=colores)
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probabilidad de morosidad (%)')
        ax.set_title('Comparación de predicciones por modelo')

        # Etiquetas encima de las barras
        for bar in bars:
            width = bar.get_width()
            estado = "Moroso" if width > 50 else "No Moroso"
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.2f}% - {estado}', va='center', fontsize=12)

        st.pyplot(fig)
        plt.close(fig)



