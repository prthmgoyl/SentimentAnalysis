import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

df= pd.read_csv('IMDBDataset.csv')
print(df)

vector = CountVectorizer()
tfidf = TfidfVectorizer( 
    use_idf=True, norm='l2' , smooth_idf=True
)
df['boolean'] = (df['sentiment'] == 'positive').astype(int)
y = df['boolean']
x= tfidf.fit_transform(df['review'].values.astype('U'))

from sklearn.model_selection import train_test_split
xtrain , xtest , ytrain , ytest = train_test_split(x,y,random_state=1, test_size=0.2)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(xtrain,ytrain)

print("TrainAccuracy :",clf.score(xtrain,ytrain))
print("TestAccuracy :",clf.score(xtest,ytest))