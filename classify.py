import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score

import string
import keras
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from keras.models import Sequential
from keras.initializers import Constant
from keras.optimizers import Adam
from spellchecker import SpellChecker
import gensim


tweet=pd.read_csv(r"C:\Users\sasan\OneDrive\Desktop\Fall 2022 Folder\ALDA\Project\train.csv")
test=pd.read_csv(r"C:\Users\sasan\OneDrive\Desktop\Fall 2022 Folder\ALDA\Project\test.csv")
tweet = tweet.drop(columns=['id'])
test = test.drop(columns=['id'])
print(tweet.shape,test.shape)

#visualize missing data

missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)

sns.barplot(x=tweet[missing_cols].isnull().sum().index, y=tweet[missing_cols].isnull().sum().values, ax=axes[0])
sns.barplot(x=test[missing_cols].isnull().sum().index, y=test[missing_cols].isnull().sum().values, ax=axes[1])

axes[0].set_ylabel('Missing Value Count', size=15, labelpad=20)
axes[0].tick_params(axis='x', labelsize=15)
axes[0].tick_params(axis='y', labelsize=15)
axes[1].tick_params(axis='x', labelsize=15)
axes[1].tick_params(axis='y', labelsize=15)

axes[0].set_title('Training Set', fontsize=13)
axes[1].set_title('Test Set', fontsize=13)

plt.show()

for df in [tweet, test]:
   for col in ['keyword', 'location']:
       df[col] = df[col].fillna(f'no_{col}')

# from the chart - it makes sense to drop location as it is missing in more than 33% of data
# Locations are not automatically generated, they are user inputs. That's why location is very dirty and there are too many unique values in it. It shouldn't be used as a feature.
tweet = tweet.drop(columns=['location'])
test = test.drop(columns=['location'])

#keyword can be dropped as it is already part of the tweet
tweet = tweet.drop(columns=['keyword'])
test = test.drop(columns=['keyword'])

#count of classes - There is a class distribution.There are more tweets with class 0 ( No disaster) than class 1 ( disaster tweets)

x=tweet.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')


def create_corpus(target):
   corpus = []

   for x in tweet[tweet['target'] == target]['text'].str.split():
       for i in x:
           corpus.append(i)
   return corpus


corpus = create_corpus(0)
corpus.extend(create_corpus(1))

#common words
counter=Counter(corpus)
most=counter.most_common()
x=[]
y=[]
for word,count in most[:40]:
   if (word not in stop) :
       x.append(word)
       y.append(count)

sns.barplot(x=y,y=x)

#since most of the common words are stop words - a lot of cleaning is required



