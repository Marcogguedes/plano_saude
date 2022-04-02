Prevendo o Custo do Plano de Saúde
O Brasil alcançou o número recorde de beneficiários de planos de saúde desde 2016, com 48,4 milhões de usuários. O balanço é da Agência Nacional de Saúde Suplementar(ANS), confirmando os dados que tinham sidos estimados no boletim prelimar da agência. Ainda assim mais de 60 % dos brasileiros, aproximadamente, não possuem plano de saúde.

1º Plobrema de Negócio
A empresa GSH - Guedes Science Health que é uma operadora de plano de saúde, que quer desenvolver um trabalho com uma metodologia para previsão do valor do plano de saúde para seus benificiários.

2º Carregando Bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o dataset
train_data = pd.read_csv('Train_Data.csv')

# Visualizando as primeiras (5) linhas do dataset
train_data.head()

# Verificando a forma do dataset
train_data.shape

# Trazendo as informações do dataset
train_data.info()

3º Análise de Dados

# Checando se há valores nulos (Missing Values)
train_data.isnull().sum()

# Verificando as estatísticas do dataset
train_data.describe()

# Decrição do dataset (categorias)
train_data.describe(include=['O'])

Taxas de Seguro Médico

# Histograma da taxa de seguro médico
plt.figure(figsize=(15,5))
sns.histplot(train_data['charges'], kde=True, color='seagreen')
plt.title('Despesas com Plano de Saúde', fontsize=20)
plt.show()

observações:

Os valores medianos do plano de saúde estão entre 8.000,00 a 9.000,00 reais.

A distribuição é assimétrica à direita do gráfico de histograma.

# Boxplot da taxa de seguro médico
plt.figure(figsize=(15,5))
sns.boxplot(train_data['charges'], color='tomato')
plt.title('Despesas com Plano de Saúde (Boxplot)', fontsize=20)
plt.show()

observações:

No gráfico boxplot observa-se vários out layers em relação aos valores com despesas devido a assimetria dos valores maiores do que a média.

Idades

# Histograma das idades dos beneficiários do plano
plt.figure(figsize=(15,5))
sns.histplot(train_data['age'], kde=True, color='seagreen')
plt.title('Idade', fontsize=20)
plt.show()

observações:

As idades dos beneficiários em média estão em torno de 40 anos.

# Boxplot das idades dos beneficiários do plano
plt.figure(figsize=(15,5))
sns.boxplot(train_data['age'], color='tomato')
plt.title('Idade (Boxplot)', fontsize=20)
plt.show()

observações:

Não há out layers em relação das idades dos beneficiários.

# Histograma do índice de massa corporal
plt.figure(figsize=(15,5))
sns.histplot(train_data['bmi'], kde=True, color='seagreen')
plt.title('Índice de Massa Corporal', fontsize=20)
plt.show()

# Boxplot do índice da massa corporal
plt.figure(figsize=(15,5))
sns.boxplot(train_data['bmi'], color='tomato')
plt.title('Índice de Massa Corporal (Boxplot)', fontsize=20)
plt.show()

observações:

Observa-se que há out layers com obsidade mórbida e com o índice de massa corporal acima dos cinquenta por cento.

# Histograma do número de filhos
plt.figure(figsize=(15,5))
sns.histplot(train_data['children'], kde=True, color='seagreen')
plt.title('Número de Filhos', fontsize=20)
plt.show()

observações:

A quantidade de números de filhos dos benificiáriso está em um número de três.

# Boxplot do número de filhos
plt.figure(figsize=(15,5))
sns.boxplot(train_data['children'], color='tomato')
plt.title('Número de Filhos (Boxplot)', fontsize=20)
plt.show()

observações:

A mediana do número de filhos está em três.

Gênero

# value counts
print('Male:', train_data['sex'].value_counts()[0])
print('Female:', train_data['sex'].value_counts()[1])

# visualização
plt.figure(figsize=(15,5))
sns.countplot(train_data['sex'])
plt.title('Gênero', fontsize=20)
plt.show()

observações:

Há um equilíbrio no número de benifeciários por gênero sendo que há mais homens.

Fumantes

# value counts
print("Smokers:", train_data['smoker'].value_counts()[1])
print("Non-Smokers:", train_data['smoker'].value_counts()[0])

# visualização
plt.figure(figsize=(15,5))
sns.countplot(train_data['smoker'])
sns.countplot(train_data['smoker'])
plt.title('Quantidade de Fumantes e Não Fumantes', fontsize=20)
plt.show()

observações:

O número de beneficiários não fumantes está em um número bem maior.

Região

# value counts
print("South-East region:", train_data['region'].value_counts()[0])
print("North-East region:", train_data['region'].value_counts()[1])
print("South-East region:", train_data['region'].value_counts()[2])
print("North-East region:", train_data['region'].value_counts()[3])

# visualização
plt.figure(figsize=(15,4))
sns.countplot(train_data['region'])
sns.countplot(train_data['region'])
plt.title('Regiões', fontsize=20)
plt.show()

train_data.head()

4º Pre-Processamento dos Dados

# Arredondar a variável idade
train_data['age'] = round(train_data['age'])

# Primeiras 5 linhas do dataset depois da realização do arredondamento
train_data.head()

# OHEncoding: Transformando as variáveis para o tipo numérico
train_data = pd.get_dummies(train_data, drop_first=True)

train_data.head()

# colunas do dataset
train_data.columns

# reorganizando as colunas para ver melhor
train_data = train_data[['age','sex_male','smoker_yes','bmi','children','region_northwest','region_southeast','region_southwest','charges']]
train_data.head(2)

# divisão de recurso independente e dependente
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

# dois melhores registros de recursos independentes
X.head(2)

# dois melhores registros de recursos dependentes
y.head(2)

# divisão dos dados em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

4º e 5º Construção e Avaliação de Máquina Preditiva

Criação da metodologia de previsão do valor do custo do plano de saúde que é a própria máquina preditiva.

# Importando as métricas de avaliação
from sklearn.metrics import mean_squared_error, r2_score

MP com Regressão Linear

# Linear Regression:
from sklearn.linear_model import LinearRegression
LinearRegression = LinearRegression()
LinearRegression = LinearRegression.fit(X_train, y_train)

# Prediction:
y_pred = LinearRegression.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

observação:

Com a regressão linear a acuracidade alcançou setenta e quatro por cento.

MP com Regressão Ridge

# Ridge
from sklearn.linear_model import Ridge
Ridge = Ridge()
Ridge = Ridge.fit(X_train, y_train)

# Prediction:
y_pred = Ridge.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

observação:

Com a regressão ridge a acuracidade alcançou setenta e quatro por cento ligeiramento inferior a regressão linear.

MP com Regressão Lasso

# Lasso
from sklearn.linear_model import Lasso
Lasso = Lasso()
Lasso = Lasso.fit(X_train, y_train)

# Prediction:
y_pred = Lasso.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

observações:

Com a regressão Lasso a acuracidade alcançou setenta e quatro por cento.

MP com Random Forest

from sklearn.ensemble import RandomForestRegressor
RandomForestRegressor = RandomForestRegressor()
RandomForestRegressor = RandomForestRegressor.fit(X_train, y_train)

# Prediction:
y_pred = RandomForestRegressor.predict(X_test)

# Scores:
print(r2_score(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))

observações:

Com o Randon Forest a acuracidade alcançou noventa por cento.

Salvando máquina preditiva para o Deploy ou a Imprementação

# Criando um Pickle File para o classificador
import pickle
filename = 'MedicalInsuranceCost.pkl'
pickle.dump(RandomForestRegressor, open(filename, 'wb'))

FIM