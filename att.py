# Importação das bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Carregar o dataset
url = 'https://www.kaggle.com/code/jurk06/dengue-prediction/data'  # URL do dataset
data = pd.read_csv(url)

# 2. Pré-processamento de Dados
# Preencher valores ausentes com a média
data.fillna(data.mean(), inplace=True)

# Codificando a variável 'City' para valores numéricos
label_encoder = LabelEncoder()
data['City'] = label_encoder.fit_transform(data['City'])

# Normalizando as variáveis climáticas (Temperatura, Umidade, Precipitação)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Temperature', 'Humidity', 'Precipitation']])

# 3. Clusterização com K-Means
# Determinar o número ideal de clusters usando o Método do Cotovelo
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plotando o gráfico do Método do Cotovelo
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('Número de Clusters')
plt.ylabel('Inércia')
plt.show()

# Aplicar KMeans com o número ideal de clusters (supondo k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(scaled_data)

# Visualizando os clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Temperature', y='Humidity', hue='Cluster', palette='viridis')
plt.title('Clusterização K-Means')
plt.show()

# 4. Preparação para a Regressão Logística
# Criando uma variável binária para indicar o surto de dengue
data['Dengue Surto'] = np.where(data['Dengue Cases'] > 0, 1, 0)

# Selecionando as variáveis independentes (features)
X = data[['Temperature', 'Humidity', 'Precipitation', 'City']]
y = data['Dengue Surto']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. Treinando o modelo de Regressão Logística
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)

# 6. Previsões e Avaliação do Modelo de Regressão Logística
y_pred = log_reg.predict(X_test)

# Acurácia do modelo
print("Acurácia:", accuracy_score(y_test, y_pred))

# Relatório de classificação
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de Confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Não Surto', 'Surto'], yticklabels=['Não Surto', 'Surto'])
plt.title('Matriz de Confusão')
plt.xlabel('Previsão')
plt.ylabel('Real')
plt.show()
