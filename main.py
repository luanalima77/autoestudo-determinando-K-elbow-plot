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
    
