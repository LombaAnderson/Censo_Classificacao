# Censo_Classificacao
[![NPM](https://img.shields.io/npm/l/react)](https://github.com/LombaAnderson/Censo_Classificacao/blob/main/LICENSE)

# Sobre o projeto

Base de dados censo retirada do site UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/index.php). Nesse conjunto de dados a precisão de acertos
do algoritmo Naïve Bayes foi de 79,4 %.  Nessa análise os atributos categóricos foram transformados em numéricos para que a classificação pudesse ser feita pelo algoritmo.


# Tecnologias utilizadas

- Anaconda
- Jupyter Notebook
- Pandas
- Sklearn
- Naïve Bayes 


# Como executar o projeto

Pré-requisitos: Possuir instalado Python, Anaconda e acessar o Jubyter Notebook 

```bash
# clonar repositório
git clone https://github.com/LombaAnderson/Censo_Classificacao

# entrar na pasta do projeto com o Jupyter Notebook

# Acessar file e escolher o nome do projeto Censo_Classificacao

# executar o projeto

Preprocessamento_censo.py 



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


# Divisão das bases de dados em teste e treinamento(Aprendizagem supervisionada)

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split\
    (previsores,classe,test_size = 0.25, random_state= 0)


# Utilização do algoritmo Naïve Bayes para geração da tabela de probabilidade

from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# Códigos para verificar os acertos e erros do algoritmo Naïve Bayes

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)



# Resultado da acurácia do algoritmo Naïve Bayes

precisao * 100  # resultado 79,4 % 

```

# Autor

Anderson Lomba de Oliveira

Linkedin

https://www.linkedin.com/in/anderson-lomba-140279142/



"""
Created on Sat Apr  10 19:20:12 2021





