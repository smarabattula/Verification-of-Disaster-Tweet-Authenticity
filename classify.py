import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
from sklearn.decomposition import TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches

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


#visualization of tf-idf and word2vec
X_train, X_test, y_train, y_test = train_test_split(tweet["text"], tweet["target"], test_size=0.2, random_state=2022)

#plotting using latent sentiment analysis - This transformer performs linear dimensionality reduction by means of truncated singular value decomposition
def plot_LSA(test_data, test_labels, plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue', 'blue']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Irrelevant')
        green_patch = mpatches.Patch(color='blue', label='Disaster')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})


#plotting tfidf
def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()

    train = tfidf_vectorizer.fit_transform(data)

    return train, tfidf_vectorizer

X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

fig = plt.figure(figsize=(16, 16))
plot_LSA(X_train_tfidf, y_train)
plt.show()

#plotting word2vec
#fig = plt.figure(figsize=(16, 16))
#plot_LSA(embeddings, list_labels)
#plt.show()
