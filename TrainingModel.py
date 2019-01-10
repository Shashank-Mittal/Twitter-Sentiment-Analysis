import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
dataset=pd.read_csv("train.csv",encoding='latin1')
X1=dataset.iloc[:,1].values
Y1=dataset.iloc[:,2].values
labelEncoder_X=LabelEncoder()
X1=labelEncoder_X.fit_transform(X1)
cr=[]
for i in range(0,99989):
    lis=Y1[i].split(" ")
    for k in lis:
        try:
            if(k[0]=="@" or k[0]=="&" or k[0]=="#"):
                lis.remove(k)
        except:
            l=1
    Y1[i]=" ".join(lis)
    Emotion=re.sub('[^a-zA-Z]'," ",Y1[i])
    Emotion=Emotion.lower()
    Emotion=Emotion.split()
    ps=PorterStemmer()
    Emotion=[ps.stem(word) for word in Emotion if not word in set(stopwords.words("english"))]
    cr.append(" ".join(Emotion))

cv=CountVectorizer()
x=cv.fit_transform(cr).toarray()
sc = StandardScaler()
xP = sc.fit_transform(x)
x_train,x_test,X1_train,X1_test= train_test_split(X1,xp,test_size=0.15,random_state=0)
classifier = RandomForestClassifier(n_estimators = 90, criterion = 'entropy', random_state = 0)
classifier.fit(X1_train,x_train)
X1_predict=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(X1_test,X1_predict)
print(cm)

#joblib.dump(classifier,"modelllll.pkl")
#joblib.dump(cv,"countvectorizer.pkl")
#joblib.dump(sc,"feature.pkl")

