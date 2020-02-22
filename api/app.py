import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
from flask.json import jsonify
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
  return flask.render_template('./index.html')


def preprocessamento(df):
  # converte letras para lowercase
  df = df.apply(lambda x: x.lower())

  # remove pontuação
  df = df.str.replace(r'[^\w\s]', '')
  
  # remove acentos
  df = df.apply(lambda x: normalize('NFKD', x).encode('ASCII', 'ignore').decode('ASCII'))

  # remove \n|\r|\n\r
  df = df.str.replace(r'\n|\r|\n\r', ' ')

  # remove stopwords
  stop = set(stopwords.words('portuguese'))
  df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  
  return df

@app.route('/resultado',methods = ['POST'])
def resultado():
    if request.method == 'POST':
      try: 
        letra = request.form.get('letra_musica')
        letra_copy = letra
        letra = [letra]

        tfidf = pickle.load(open("tfidf_vec.pickle", "rb"))
        model = pickle.load(open("model.pickle","rb"))
        label = pickle.load(open("labels.pickle","rb"))

        print(label.classes_)
        vect = tfidf.transform(letra).toarray()
        prediction = model.predict(vect) 
        prediction = label.inverse_transform(prediction)
      except Exception:
        prediction = "Erro"
        letra_copy = ""
      
      return render_template('./resultado.html', prediction=prediction, letra = letra_copy)
