# Adicionar importação de bibliotecas do Python.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

#Importar KMeans da biblioteca scikit-learn.
from sklearn.cluster import KMeans

#Importar dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

#Calcular WCSS (o init inicializa o centroide).
wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#PLOTAR O ELBOW PLOT.
plt.figure(figsize=(8,5))
#Definindo o eixo x de 1 a 10 e o eixo y com o wcss.
plt.plot(range(1,11), wcss, marker="o", linestyle = "--")

#Colocando informações textuais do gráfico.
plt.title("Elbow plot")
plt.show()

#ADICIONAR TREINAMENTO DO MODELO.
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 60, c = 'yellow', label = 'Cluster5') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.xlabel('Annual Income (k$)') 
plt.ylabel('Spending Score (1-100)') 
plt.legend() 
plt.show()