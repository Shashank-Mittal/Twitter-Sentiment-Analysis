from flask import *
import nltk
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import pickle
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/filtered",methods=["GET"])
def filtered():
    q = request.args.get('input') # read data from field
    nltk.download("stopwords") # stopword required
    p = joblib.load('model.pkl')
    p1 = joblib.load('countvectorizer.pkl')
    ps=PorterStemmer()
    # check the sentence
    crt=[]
    revie=re.sub('[^a-zA-Z]'," ",q)
    revie=revie.lower()
    revie=revie.split()
    revie=[ps.stem(word) for word in revie if not word in set(stopwords.words("english"))]
    crt.append(" ".join(revie))
    xt=p1.transform(crt)
    y_predict=p.predict(xt)
    return str(y_predict)

if __name__ == '__main__':
    app.run(debug=True)