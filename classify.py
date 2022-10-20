from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import defaultdict
from collections import  Counter
plt.style.use('ggplot')
stop=set(stopwords.words('english'))
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
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
tweet = tweet.drop(columns=['id'])
print(tweet.shape)

#visualize missing data

missing_cols = ['keyword', 'location']

fig, axes = plt.subplots(ncols=1, figsize=(17, 4), dpi=100)

sns.barplot(x=tweet[missing_cols].isnull().sum().index, y=tweet[missing_cols].isnull().sum().values, ax=axes)

axes.set_ylabel('Missing Value Count', size=15, labelpad=20)
axes.tick_params(axis='x', labelsize=15)
axes.tick_params(axis='y', labelsize=15)

axes.set_title('Training Set', fontsize=13)

plt.show()

# from the chart - it makes sense to drop location as it is missing in more than 33% of data
# Locations are not automatically generated, they are user inputs. That's why location is very dirty and there are too many unique values in it. It shouldn't be used as a feature.
tweet = tweet.drop(columns=['location'])

#keyword can be dropped as it is already part of the tweet
tweet = tweet.drop(columns=['keyword'])

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

corpus_merged=create_corpus(1)
corpus_merged.extend(create_corpus(0))
wc = WordCloud(background_color='black')
wc.generate(' '.join(corpus_merged))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()

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

def clean_tweet(text):
    # converting text to lower case
    text = text.lower()
    # removing all mentions and hashtags from the tweet
    temp = re.sub("@[a-z0-9_]+", "", text)
    temp = re.sub("#[a-z0-9_]+", "", temp)
    # removing all websites and urls from the tweet
    temp = re.sub(r"http\S+", "", temp)
    temp = re.sub(r"www.\S+", "", temp)
    # removing punctuations from the tweet
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]', ' ', temp)
    # removing all non-alphanumeric characters from the text
    temp = re.sub("[^a-z0-9]", " ", temp)

    # correcting spellings
    # temp = correct_spellings(temp)

    # removing all stopwords from the text -- #todo check accuracy with and without removing these
    temp = temp.split()
    temp = [w for w in temp if not w in stop]
    temp = " ".join(word for word in temp)

    # not stemming because the stemmed words will not be present in Glove and word2vec databases

    return temp

rawTexData = tweet["text"].head(10)

tweet['text']=tweet['text'].apply(lambda x : clean_tweet(x))
cleanTexData = tweet["text"].head(10)
#visualization of tf-idf and word2vec
X_train, X_test, y_train, y_test = train_test_split(tweet["text"], tweet["target"], test_size=0.2, random_state=2022)

#plotting using latent sentiment analysis - This transformer performs linear dimensionality reduction by means of truncated singular value decomposition
def plot_LSA(test_data, test_labels, plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['red', 'blue', 'green']
    if plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='red', label='Not Disaster')
        green_patch = mpatches.Patch(color='green', label='Disaster')
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

word2vec_path = "~/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(lambda x: get_average_word2vec(x, vectors,
                                                                                generate_missing=generate_missing))
    return list(embeddings)

tokenizer = RegexpTokenizer(r'\w+')
list_labels = tweet["target"].tolist()
tweet["tokens"] = tweet["text"].apply(tokenizer.tokenize)
tweet.head()

embeddings = get_word2vec_embeddings(word2vec, tweet)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels,
                                                                                        test_size=0.2, random_state=2022)
																						#plotting word2vec
fig = plt.figure(figsize=(16, 16))
plot_LSA(embeddings, list_labels)
plt.show()

#Logistic regression
logistic_reg = LogisticRegression(penalty='l2',
                                        solver='saga',
                                        random_state = 2022)

logistic_reg.fit(X_train_word2vec,y_train_word2vec)
print("Logistic Regression model run successfully")

#SVM
SVClassifier = SVC(kernel= 'linear',
                   degree=3,
                   max_iter=10000,
                   C=2,
                   random_state = 2022)

SVClassifier.fit(X_train_word2vec,y_train_word2vec)

print("SVClassifier model run successfully")

models = [logistic_reg, SVClassifier]

models = [logistic_reg, SVClassifier]

for model in models:
    print(type(model).__name__,'Train Score is   : ' ,model.score(X_train_word2vec, y_train_word2vec))
    print(type(model).__name__,'Test Score is    : ' ,model.score(X_test_word2vec, y_test_word2vec))
    y_pred_word2vec = model.predict(X_test_word2vec)
    print(type(model).__name__,'F1 Score is      : ' ,f1_score(y_test_word2vec,y_pred_word2vec))
    print('**************************************************************')
