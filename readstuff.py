## read text and preprocessing

import numpy as np
import re
import nltk

# download when needed
#nltk.download('stopwords')
#nltk.download('wordnet')

from sklearn.datasets import load_files
import pickle
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()
tfile = open('imdb_train.txt','r',encoding='utf-8')
Y = [] #label
x = [] #initial feature

#what the file looks like:
# 1 this movie is very good .......
for line in tfile:
    Y.append(line[0]) #take the 1
    x.append(line[2:])#take the text

tmp = [] #storing the processed x
for w in range(0, len(x)):
    # remove special characters
    txt = re.sub(r'\W', ' ', str(x[w]))
    
    # remove characters
    txt = re.sub(r'\s+[a-zA-Z]\s+', ' ', txt)
    
    # Remove single characters start
    txt = re.sub(r'\^[a-zA-Z]\s+', ' ', txt) 
    
    # remove multiple spaces with single space
    txt = re.sub(r'\s+', ' ', txt, flags=re.I)
    
    # removing prefixed 'b'
    txt = re.sub(r'^b\s+', '', txt)
    
    # turn lowercase
    txt = txt.lower()
    
    # lemmatization
    txt = txt.split()

    txt = [wnl.lemmatize(word) for word in txt]
    txt = ' '.join(txt)
    
    tmp.append(txt)
    
print(len(tmp))
print(tmp[0])
print(Y[1:5])

## turn text to num
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# initialize vectorizer
# max_features = bag of words method, 1500 as most occuring words
# mind_df = only include features that appear in at least 5 docs
# max_df = like min_df, 0.7 means 70%
# remove stopwords in file from nltk lib
vtrizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
# turn words to nums
# fit_transform coverts text into num features
X = vtrizer.fit_transform(tmp).toarray()

# TFIDF only useful when your text files are separated
# TFIDF assign score across documents
#from sklearn.feature_extraction.text import TfidfTransformer
#X = TfidfTransformer.fit_transform(X1).toarray()

print(len(X))
print(X[2])

## split dataset for training and testing
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y, test_size=0.2, random_state=0)

## model training and testing
from sklearn.ensemble import RandomForestClassifier
bc = RandomForestClassifier(n_estimators=1000, random_state=0)
bc.fit(xtrain, ytrain)
ypred = bc.predict(xtest)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(ytest,ypred))
print(classification_report(ytest,ypred))
print(accuracy_score(ytest, ypred))