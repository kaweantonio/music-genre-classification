import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

import pickle

from unicodedata import normalize

df_bossa = pd.read_csv('./data/bossa_nova.csv')

df_funk = pd.read_csv('./data/funk.csv')

df_gospel = pd.read_csv('./data/gospel.csv')

df_sertanejo = pd.read_csv('./data/sertanejo.csv')

# cria base de dados única

df_bossa['genre'] = 'bossa'
df_funk['genre'] = 'funk'
df_gospel['genre'] = 'gospel'
df_sertanejo['genre'] = 'sertanejo'

frames = [df_bossa, df_funk, df_gospel, df_sertanejo]

base_dados = pd.concat(frames, ignore_index = True)

def preprocessamento(df):
  # converte letras para lowercase
  df['lyric'] = df['lyric'].apply(lambda x: x.lower())

  # remove pontuação
  df['lyric'] = df['lyric'].str.replace(r'[^\w\s]', '')
  
  # remove acentos
  df['lyric'] = df['lyric'].apply(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore').decode('ASCII'))

  # remove \n|\r|\n\r
  df['lyric'] = df['lyric'].str.replace(r'\n|\r|\n\r', ' ')

  # remove stopwords
  stop = set(stopwords.words('portuguese'))
  df['lyric'] = df['lyric'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  
  # convertendo label 'genre' para números
  le = LabelEncoder()
  df['genre'] = le.fit(df['genre']).transform(df['genre'])
  
  return df

df = preprocessamento(base_dados)

# normalizacao 
tfidf_vec = TfidfVectorizer()
tfidf_vec.set_params(stop_words=None, max_features=30000, min_df=4, ngram_range=(1, 2))
tfidf = tfidf_vec.fit(df.lyric)
X = tfidf_vec.transform(df.lyric)

pickle.dump(tfidf, open("tfidf_vec.pickle", "wb"))

scale = MinMaxScaler()
X = scale.fit_transform(X.toarray())

X = pd.DataFrame(X)
y = df.genre

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = .1, random_state=200)

resultados = []

lr = LogisticRegression()
fit = lr.fit(x_train, y_train)
pred = fit.predict(x_test)
accuracy = accuracy_score(y_test, pred)

resultados.append({ "Classificador" : "Logistic Regression", 
                        "Acurácia" :   "%.4f" % accuracy, 
                      })

nb = MultinomialNB()
fit = nb.fit(x_train, y_train)
pred = fit.predict(x_test)
accuracy = accuracy_score(y_test, pred)

resultados.append({ "Classificador" : "Naive Bayes", 
                        "Acurácia" :   "%.4f" % accuracy, 
                      })

sgdc = SGDClassifier(loss='hinge', penalty='l2',
                     alpha=1e-3, random_state=42)
fit = sgdc.fit(x_train, y_train)
pred = fit.predict(x_test)
accuracy = accuracy_score(y_test, pred)

resultados.append({ "Classificador" : "SGDC", 
                        "Acurácia" :   "%.4f" % accuracy, 
                      })

decision = MLPClassifier(alpha=1)
fit = decision.fit(x_train, y_train)
pred = fit.predict(x_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average=None)

resultados.append({ "Classificador" : "Neural Network", 
                        "Acurácia" :   "%.4f" % accuracy, 
                      })

X = X.loc[:,:].values
y = y.loc[:].values

kf = KFold(n_splits=10)

classificadores = [LogisticRegression(),
                   MultinomialNB(),
                   SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42),
                   MLPClassifier(alpha=1),
                  ]

identificacao = [
                 'Logistic Regression',
                 'Naive Bayes', 
                 'SGDC', 
                 'Neural Network',
                ]

acuracia = {}

for i in identificacao:
  acuracia[i] = []
  
for train, test in kf.split(X):
  X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
  
  for ident, clas in zip(identificacao, classificadores):
    fit = clas.fit(X_train, y_train)
    pred = fit.predict(X_test)
    acuracia[ident].append(accuracy_score(y_test, pred))

resultados2 = []

for i in identificacao:
    resultados2.append({ "Classificador" : i, 
                        "Acurácia (Kfold)" :   "%.4f ± %.4f " % (np.mean(acuracia[i]), np.std(acuracia[i])), 
                      })

pd.DataFrame(resultados)

pd.DataFrame(resultados2)
