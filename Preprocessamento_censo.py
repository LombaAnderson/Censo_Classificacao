# -*- coding: utf-8 -*-
"""
Created on Sat Apr  10 19:20:12 2021

@author: Anderson Lomba
"""
import pandas as pd

base = pd.read_csv('census.csv' )

# Criando matriz previsores e matriz classe 
previsores = base.iloc[:, 0:14].values
classe = base.iloc[:,14].values


# Transformando atributos categóricos de previsores em atributos numéricos

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()  # Instanciamento da variável labelencoder
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,5] = labelencoder.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder.fit_transform(previsores[:,9])
previsores[:,13] = labelencoder.fit_transform(previsores[:,13])


# Transformando atributos categóricos da matriz classe em atributos numéricos
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)


# Divisão das bases de dados em teste e treinamento(Aprendizagem de máquina)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split\
    (previsores,classe,test_size = 0.25, random_state= 0)


# Utilização do algoritmo Naïve Bayes para aprendizagem de máquina (geração da tabela de probabilidade)

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# Códigos para verificar os acertos e erros do algoritmo Naïve Bayes

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)



# Acurácia do algoritmo Naïve Bayes























