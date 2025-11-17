import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn 
import torch.optim as optim
import random 
import numpy as np

from skorch import NeuralNetClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split



with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv("DOCCC.csv")
df.rename(columns={'default payment next month':'DEFAULT'}, inplace=True)
df.rename(columns={'PAY_0': 'PAY_1'}, inplace=True)
df = df.drop(df[df['MARRIAGE']==0].index)
df = df.drop(df[df['EDUCATION']==0].index)
df = df.drop(df[df['EDUCATION']==5].index)
df = df.drop(df[df['EDUCATION']==6].index)
cols_to_drop = ['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 
                'BILL_AMT5', 'BILL_AMT6', 'ID']
df = df.drop(columns=cols_to_drop, errors='ignore')

y = df["DEFAULT"].values
X = df.drop(columns=["DEFAULT"])

# Escalado
X_scaled = scaler.transform(X.values)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Convertir a float32
X_train_np = X_train.astype('float32')
X_test_np  = X_test.astype('float32')
y_train_np = y_train.astype('float32').reshape(-1,1)  
y_test_np  = y_test.astype('float32').reshape(-1,1)   

# Semillas
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.use_deterministic_algorithms(True)

# Modelo PyTorch
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

net = NeuralNetClassifier(
    module=ClassNN,
    criterion=nn.BCEWithLogitsLoss,
    optimizer=optim.Adam,
    max_epochs=50,
    lr=0.001,
    batch_size=32,
    iterator_train__shuffle=True,
    verbose=1,
    device='cpu'
)

# Param grid
param_dist = {
    'module__hidden_sizes': [[16], [32], [16,8], [32,16], [32,16,8], [64], [64,32], [128,64,32]],
    'module__activation': [nn.ReLU, nn.Tanh, nn.LeakyReLU, nn.ELU],
    'module__dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
    'optimizer': [optim.SGD, optim.Adam, optim.RMSprop],
    'lr': [0.01, 0.005, 0.001, 0.0005],
    'batch_size': [16, 32, 64, 128],
}

rand_search = RandomizedSearchCV(
    estimator=net,
    param_distributions=param_dist,
    n_iter=20, 
    scoring='accuracy',  
    cv=3,
    n_jobs=-1,
    random_state=seed,
    return_train_score=True
)

# Busqueda de los mejores hiperparametros (contenidos en el area de busqueda)
rand_search.fit(X_train_np, y_train_np)

# Resultados
scores = pd.DataFrame(rand_search.cv_results_)
bestMOD_pytorch = rand_search.best_estimator_

#Entrenamiento final con el mejor modelo
bestMOD_pytorch.fit(X_train_np, y_train_np)

#Guardando el mejor modelo pytorch
joblib.dump(bestMOD_pytorch, "03_modelo_pytorch_app.pkl")
