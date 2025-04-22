import pickle
import pandas as pd

# Carregar modelo
with open('modelo.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Exemplo de entrada
entrada = pd.DataFrame([{
    'idade': 34,
    'sexo': 0,
    'frequencia_mensal': 6,
    'atraso_pagamento': 1,
    'dias_sem_frequencia': 14,
    'uso_personal': 0
}])

# Prever
resultado = modelo.predict(entrada)
print("Vai cancelar?" , "Sim" if resultado[0] == 1 else "Não")
