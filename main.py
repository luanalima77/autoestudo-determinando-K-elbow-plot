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