import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o modelo treinado
def carregar_modelo():
    with open('modelo.pkl', 'rb') as f:
        modelo = pickle.load(f)
    return modelo

# Carregar o modelo quando a aplicação iniciar
model = carregar_modelo()

st.set_page_config(page_title="Previsão de Cancelamento", layout="wide")

# Estilo do título
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧠 Previsão de Cancelamento - Academia Força Local</h1>
    <p style='text-align: center;'>Plataforma inteligente para ajudar a reduzir a evasão de clientes</p>
    <hr>
""", unsafe_allow_html=True)

# Dados fictícios
@st.cache_data
def gerar_dados():
    np.random.seed(42)
    dados = pd.DataFrame({
        'idade': np.random.randint(18, 60, 300),
        'frequencia_media': np.random.randint(1, 30, 300),
        'atrasos_pagamento': np.random.randint(0, 5, 300),
        'usou_personal': np.random.choice([0, 1], 300),
        'cancelou': np.random.choice([0, 1], 300, p=[0.7, 0.3])
    })
    return dados

dados = gerar_dados()

# Visualização dos dados
with st.expander("📊 Visualizar dados"):
    st.dataframe(dados.head(10))

# Gráfico
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots()
    sns.countplot(x='cancelou', data=dados, ax=ax, palette='Set2')
    ax.set_title('Distribuição de Cancelamentos')
    ax.set_xticklabels(['Não Cancelou', 'Cancelou'])
    st.pyplot(fig)

with col2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='cancelou', y='frequencia_media', data=dados, palette='Set3')
    ax2.set_title('Frequência de uso x Cancelamento')
    ax2.set_xticklabels(['Não Cancelou', 'Cancelou'])
    st.pyplot(fig2)

# Separar dados
X = dados.drop('cancelou', axis=1)
y = dados['cancelou']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Avaliação (Utilizando o modelo carregado)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.success(f"✅ Acurácia do modelo: {acc * 100:.2f}%")
with st.expander("🔍 Relatório completo"):
    st.text(classification_report(y_test, y_pred))

# Previsão interativa
st.markdown("### 🧪 Prever se um cliente pode cancelar:")

idade = st.slider("Idade", 18, 60, 30)
frequencia = st.slider("Frequência média (dias/mês)", 0, 30, 15)
atrasos = st.slider("Atrasos no pagamento (últimos 3 meses)", 0, 5, 1)
usou_personal = st.selectbox("Usou serviço de personal trainer?", ["Sim", "Não"])
usou_personal = 1 if usou_personal == "Sim" else 0

entrada = pd.DataFrame([[idade, frequencia, atrasos, usou_personal]],
                       columns=['idade', 'frequencia_media', 'atrasos_pagamento', 'usou_personal'])

if st.button("🔎 Verificar chance de cancelamento"):
    pred = model.predict(entrada)[0]
    if pred == 1:
        st.error("⚠️ Cliente com ALTA chance de cancelar!")
    else:
        st.success("✅ Cliente com BAIXA chance de cancelar.")
