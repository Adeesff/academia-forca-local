import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Carregar dados
df = pd.read_csv('dados_clientes.csv')

# Preprocessamento simples
df['sexo'] = df['sexo'].map({'F': 0, 'M': 1})
df['uso_personal'] = df['uso_personal'].map({'Não': 0, 'Sim': 1})

X = df.drop('cancelou', axis=1)
y = df['cancelou']

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Salvar modelo
with open('modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)
