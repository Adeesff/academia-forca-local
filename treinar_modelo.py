import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("🔧 Iniciando o treinamento...")

# Gerando dados fictícios
np.random.seed(42)
dados = pd.DataFrame({
    'idade': np.random.randint(18, 60, 300),
    'frequencia_media': np.random.randint(1, 30, 300),
    'atrasos_pagamento': np.random.randint(0, 5, 300),
    'usou_personal': np.random.choice([0, 1], 300),
    'cancelou': np.random.choice([0, 1], 300, p=[0.7, 0.3])
})

X = dados.drop('cancelou', axis=1)
y = dados['cancelou']

# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Salvando o modelo treinado
with open('modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("🔧 Iniciando o treinamento...")

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Gerando dados fictícios
np.random.seed(42)
dados = pd.DataFrame({
    'idade': np.random.randint(18, 60, 300),
    'frequencia_media': np.random.randint(1, 30, 300),
    'atrasos_pagamento': np.random.randint(0, 5, 300),
    'usou_personal': np.random.choice([0, 1], 300),
    'cancelou': np.random.choice([0, 1], 300, p=[0.7, 0.3])
})

X = dados.drop('cancelou', axis=1)
y = dados['cancelou']

# Separando treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Salvando o modelo treinado
with open('modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print("✅ Modelo treinado e salvo com sucesso!")

