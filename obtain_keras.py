import tensorflow as tf
print(tf.config.list_physical_devices())
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pickle
import warnings
warnings.filterwarnings('ignore')

# Manejo de datos
import numpy as np
import pandas as pd
from collections import Counter

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader

# scikeras
from scikeras.wrappers import KerasClassifier, KerasRegressor

# scikit-learn
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, cross_val_score,
    train_test_split, StratifiedKFold
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    ConfusionMatrixDisplay, classification_report, auc
)
from sklearn.utils import resample, compute_class_weight

# Imbalanced-learn
from imblearn.over_sampling import SMOTE

# UCI datasets
from ucimlrepo import fetch_ucirepo
# ==========================
# 1. Cargar dataset
# ==========================
df = pd.read_csv("/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/DOCCC.csv")
df.rename(columns={'default payment next month':'DEFAULT'}, inplace=True)
df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
df = df.drop(df[df['MARRIAGE']==0].index)
df = df.drop(df[df['EDUCATION']==0].index)
df = df.drop(df[df['EDUCATION']==5].index)
df = df.drop(df[df['EDUCATION']==6].index)
cols_to_drop = ['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                'BILL_AMT5', 'BILL_AMT6', 'ID']
df = df.drop(columns=cols_to_drop, errors='ignore')

df.head()
# Ajusta el nombre EXACTO de la variable objetivo
y = df["DEFAULT"].values  
X = df.drop(columns=["DEFAULT"]).values

# ==========================
# 2. Escalar datos
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar scaler
with open("/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ==========================
# 3. Partir datos
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================
# 4. Crear modelo Keras
# ==========================
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==========================
# 5. Entrenar
# ==========================
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# ==========================
# 6. Guardar modelo Keras (.h5)
# ==========================
model.save("/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/model.h5")

print("Modelo guardado como model.h5")
print("Scaler guardado como scaler.pkl")


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import os
# ==========================================
# 4. Escalado
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar scaler
with open("/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✔ scaler.pkl creado")

# ==========================================
# 5. Modelo Keras
# ==========================================
# ==========================================
# 6. Entrenamiento
# ==========================================
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==========================================
# 7. Guardar modelo
# ==========================================
model.save("/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/model.h5")
print("✔ model.h5 creado")

import pickle

# Cargar el scaler existente
with open("/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# IMPORTANTE:
# Debes reemplazar esto por tus columnas REALES del dataset
# Son las mismas columnas que usaste al entrenar X_train
feature_columns = list(["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"])


import pickle

data = {
    "model": model,
    "scaler": scaler,
    "feature_names": feature_columns
}

with open("/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/02_modelo_keras_app.pkl", "wb") as f:
    pickle.dump(data, f)

print("Archivo 02_modelo_keras_app.pkl creado correctamente.")

# Ruta del modelo keras ya entrenado
model_path = "/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/model.h5"   # o "modelo_keras.h5" si lo guardaste así

# Crear diccionario para la app
app_package = {
    "scaler": scaler,
    "feature_columns": feature_columns,
    "model_path": model_path
}

# Guardar archivo final
with open("/home/userrr/Documentos/UNALM/2025-2/CD2/PARCIAL/02_modelo_keras_app.pkl", "wb") as f:
    pickle.dump(app_package, f)

print("02_modelo_keras_app.pkl creado exitosamente.")
