import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

from main import sentence_filter, sentiment_transform, print_output, bnb, predict_and_test

sklearn_site_joblib=True

#read data
df = pd.read_csv(sys.stdin, sep='\t', header=None)
df = df.rename(columns={0: 'index', 1: 'rating', 2: 'sentence'} )
#filtering data
df['sentence'] = df['sentence'].apply(lambda x: sentence_filter(x)) #apply lambda filter to column
sentiment_transform(df)
X = df['sentence'].to_numpy()
indices = df['index'].to_numpy()
y = df['rating'].to_numpy()

# split into train and test
split_percentage = 0.8
split_point = int(len(X) * split_percentage)
X_train = X[:split_point]
X_test = X[split_point:]
y_train = y[:split_point]
y_test = y[split_point:]
indices_test = indices[split_point:]

#predict data
result = bnb(X_train, X_test, y_train, y_test, output = 'result')
#output data
print_output(indices_test, result)
