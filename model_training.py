import warnings
warnings.filterwarnings('ignore')
import re
import nltk
import matplotlib.pyplot as plt

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup

import pandas as pd
df=pd.read_csv('employee_reviews_real.csv', encoding='ISO-8859–1')
test_data=pd.read_csv('amazon.csv', encoding='ISO-8859–1')

df.head()

df.groupby("Liked").count().plot.bar()

print('~> Number of neagtive reviews:\n   {}%'.format(df[df['Liked']==0].shape[0]/df['Liked'].shape[0]*100))
print('\n~> Number of positive reviews\n   {}%'.format(df[df['Liked']==1].shape[0]/df['Liked'].shape[0]*100))

# To get the results in 4 decemal points
SAFE_DIV = 0.0001 

STOP_WORDS = stopwords.words("english")


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    porter = PorterStemmer()
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    
    
    if type(x) == type(''):
        x = porter.stem(x)
        example1 = BeautifulSoup(x)
        x = example1.get_text()
        
        return x
               
df['Review'].apply(preprocess) #higher accuracy when stopwords are not removed

from sklearn.model_selection import train_test_split
X = df['Review']
Y = df['Liked']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
tvec = TfidfVectorizer()
lr = LogisticRegression()

from sklearn.pipeline import Pipeline
# from sklearn.externals import joblib
import joblib

model = Pipeline([('Vectorizer',tvec),('classifer',lr)])
model.fit(X_train,Y_train)
joblib.dump(model, 'model_predict.joblib') #Model and pipeline save


from sklearn.metrics import confusion_matrix
prediction = model.predict(X_test)
print(confusion_matrix(prediction,Y_test))

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,prediction))
    
