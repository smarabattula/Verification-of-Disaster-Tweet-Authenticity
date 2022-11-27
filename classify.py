import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import mean
from numpy import std
import nltk
import copy
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
from wordcloud import WordCloud
from itertools import chain
import string
import keras
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from keras.models import Sequential
from keras.initializers import Constant
from keras.optimizers import Adam, Adamax

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.layers import Input, LSTM, Embedding, Dropout, Bidirectional, Dense
from tensorflow.keras import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from spellchecker import SpellChecker
import gensim
from tensorflow.random import set_seed

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

corpus = create_corpus(0)
corpus.extend(create_corpus(1))

corpus_merged=create_corpus(1)
corpus_merged.extend(create_corpus(0))
wc = WordCloud(background_color='black')
wc.generate(' '.join(corpus_merged))
plt.imshow(wc, interpolation="bilinear")
plt.axis('off')
plt.show()

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
abbreviations = {
    "$" : " dollar ",  "€" : " euro ",  "4ao" : "for adults only",  "a.m" : "before midday",  "a3" : "anytime anywhere anyplace",  "aamof" : "as a matter of fact",  "acct" : "account",  "adih" : "another day in hell",  "afaic" : "as far as i am concerned",  "afaict" : "as far as i can tell",  "afaik" : "as far as i know",  "afair" : "as far as i remember",  "afk" : "away from keyboard",  "app" : "application",  "approx" : "approximately",  "apps" : "applications",  "asap" : "as soon as possible",  "asl" : "age, sex, location",  "atk" : "at the keyboard",  "ave." : "avenue",  "aymm" : "are you my mother",  "ayor" : "at your own risk",   
    "b&b" : "bed and breakfast",  "b+b" : "bed and breakfast",  "b.c" : "before christ",  "b2b" : "business to business",  "b2c" : "business to customer",  "b4" : "before",  "b4n" : "bye for now",  "b@u" : "back at you",  "bae" : "before anyone else",  "bak" : "back at keyboard",  "bbbg" : "bye bye be good",  "bbc" : "british broadcasting corporation",  "bbias" : "be back in a second",  "bbl" : "be back later",  "bbs" : "be back soon",  "be4" : "before",  "bfn" : "bye for now",  "blvd" : "boulevard",  "bout" : "about",  "brb" : "be right back",  "bros" : "brothers",  "brt" : "be right there",  "bsaaw" : "big smile and a wink",  "btw" : "by the way",  "bwl" : "bursting with laughter", 
    "c/o" : "care of",  "cet" : "central european time",  "cf" : "compare",  "cia" : "central intelligence agency",  "csl" : "can not stop laughing",  "cu" : "see you",  "cul8r" : "see you later",  "cv" : "curriculum vitae",  "cwot" : "complete waste of time",  "cya" : "see you",  "cyt" : "see you tomorrow",  "dae" : "does anyone else",  "dbmib" : "do not bother me i am busy",  "diy" : "do it yourself",  "dm" : "direct message",  "dwh" : "during work hours",  "e123" : "easy as one two three",  "eet" : "eastern european time",  "eg" : "example",  "embm" : "early morning business meeting",  "encl" : "enclosed",  "encl." : "enclosed",  "etc" : "and so on",  "faq" : "frequently asked questions",  
    "fawc" : "for anyone who cares",  "fb" : "facebook",  "fc" : "fingers crossed",  "fig" : "figure",  "fimh" : "forever in my heart",   "ft." : "feet",  "ft" : "featuring",  "ftl" : "for the loss",  "ftw" : "for the win",  "fwiw" : "for what it is worth",  "fyi" : "for your information",  "g9" : "genius",  "gahoy" : "get a hold of yourself",  "gal" : "get a life",  "gcse" : "general certificate of secondary education",  "gfn" : "gone for now",  "gg" : "good game",  "gl" : "good luck",  "glhf" : "good luck have fun",  "gmt" : "greenwich mean time",  "gmta" : "great minds think alike",  "gn" : "good night",  "g.o.a.t" : "greatest of all time",  "goat" : "greatest of all time",  "goi" : "get over it",  
    "gps" : "global positioning system",  "gr8" : "great",  "gratz" : "congratulations",  "gyal" : "girl",  "h&c" : "hot and cold",  "hp" : "horsepower",  "hr" : "hour",  "hrh" : "his royal highness",  "ht" : "height",  "ibrb" : "i will be right back",  "ic" : "i see",  "icq" : "i seek you",  "icymi" : "in case you missed it",  "idc" : "i do not care",  "idgadf" : "i do not give a damn fuck",  "idgaf" : "i do not give a fuck",  "idk" : "i do not know",  "ie" : "that is",  "i.e" : "that is",  "ifyp" : "i feel your pain",  "IG" : "instagram",  "iirc" : "if i remember correctly",  "ilu" : "i love you",  "ily" : "i love you",  "imho" : "in my humble opinion",  "imo" : "in my opinion", 
    "imu" : "i miss you",  "iow" : "in other words",  "irl" : "in real life",  "j4f" : "just for fun",  "jic" : "just in case",  "jk" : "just kidding",  "jsyk" : "just so you know",  "l8r" : "later",  "lb" : "pound",  "lbs" : "pounds",  "ldr" : "long distance relationship",  "lmao" : "laugh my ass off",  "lmfao" : "laugh my fucking ass off",  "lol" : "laughing out loud",  "ltd" : "limited",  "ltns" : "long time no see",  "m8" : "mate",  "mf" : "motherfucker",  "mfs" : "motherfuckers",  "mfw" : "my face when",  "mofo" : "motherfucker",  "mph" : "miles per hour",  "mr" : "mister",  "mrw" : "my reaction when",  "ms" : "miss",  "mte" : "my thoughts exactly",  "nagi" : "not a good idea",  
    "nbc" : "national broadcasting company",  "nbd" : "not big deal",  "nfs" : "not for sale",  "ngl" : "not going to lie",  "nhs" : "national health service",  "nrn" : "no reply necessary",  "nsfl" : "not safe for life",  "nsfw" : "not safe for work",  "nth" : "nice to have",  "nvr" : "never",  "nyc" : "new york city",  "oc" : "original content",  "og" : "original",  "ohp" : "overhead projector",  "oic" : "oh i see",  "omdb" : "over my dead body",  "omg" : "oh my god",  "omw" : "on my way",  "p.a" : "per annum",  "p.m" : "after midday",  "pm" : "prime minister",  "poc" : "people of color",  "pov" : "point of view",  "pp" : "pages",  "ppl" : "people",  "prw" : "parents are watching",  
    "ps" : "postscript",  "pt" : "point",  "ptb" : "please text back",  "pto" : "please turn over",  "qpsa" : "what happens",  "ratchet" : "rude",  "rbtl" : "read between the lines",  "rlrt" : "real life retweet",   "rofl" : "rolling on the floor laughing",  "roflol" : "rolling on the floor laughing out loud",  "rotflmao" : "rolling on the floor laughing my ass off",  "rt" : "retweet",  "ruok" : "are you ok",  "sfw" : "safe for work",  "sk8" : "skate",  "smh" : "shake my head",  "sq" : "square",  "srsly" : "seriously",   "ssdd" : "same stuff different day",  "tbh" : "to be honest",  "tbs" : "tablespooful",  "tbsp" : "tablespooful",  "tfw" : "that feeling when",  "thks" : "thank you",  
    "tho" : "though",  "thx" : "thank you",  "tia" : "thanks in advance",  "til" : "today i learned",  "tl;dr" : "too long i did not read",  "tldr" : "too long i did not read",  "tmb" : "tweet me back",  "tntl" : "trying not to laugh",  "ttyl" : "talk to you later",  "u" : "you",  "u2" : "you too",  "u4e" : "yours for ever",  "utc" : "coordinated universal time",  "w/" : "with",  "w/o" : "without",  "w8" : "wait",  "wassup" : "what is up",  "wb" : "welcome back",  "wtf" : "what the fuck",  "wtg" : "way to go",  "wtpa" : "where the party at",  "wuf" : "where are you from",  "wuzup" : "what is up",  "wywh" : "wish you were here",  "yd" : "yard",  "ygtr" : "you got that right",  
    "ynk" : "you never know",  "zzz" : "sleeping bored and tired"
}

# Change an abbreviation by its true meaning
def word_abbrev(word):
    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word

# Replace all abbreviations
def replace_abbrev(text):
    string = ""
    for word in text.split():
        string += word_abbrev(word) + " "        
    return string


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
    
    #expand abbrevations
    text = replace_abbrev(text)
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

max_features=3000
tokenizer=Tokenizer(num_words=max_features,split=' ')
tokenizer.fit_on_texts(tweet['tokens'].values)
X = tokenizer.texts_to_sequences(tweet['tokens'].values)
X = pad_sequences(X)
X_train_word2vec1, X_test_word2vec1, y_train_word2vec1, y_test_word2vec1 = train_test_split(X, list_labels, test_size=0.2, random_state=2022)

embeddings = get_word2vec_embeddings(word2vec, tweet)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels,
                                                                                        test_size=0.2, random_state=2022)
																						#plotting word2vec
#plotting word2vec
fig = plt.figure(figsize=(16, 16),dpi=45)
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
precision_sc=[]
recall_sc=[]
f1_sc=[]
test_sc=[]
train_sc=[]
for model in models:
    print(type(model).__name__,'Train Score is   : ' ,model.score(X_train_word2vec, y_train_word2vec))
    print(type(model).__name__,'Test Score is    : ' ,model.score(X_test_word2vec, y_test_word2vec))
    train_sc.append(model.score(X_train_word2vec, y_train_word2vec))
    test_sc.append(model.score(X_test_word2vec, y_test_word2vec))
    y_pred_word2vec = model.predict(X_test_word2vec)

    print(type(model).__name__,'Precision is      : ' ,precision_score(y_test_word2vec,y_pred_word2vec))   
    print(type(model).__name__,'Recall is         : ' ,recall_score(y_test_word2vec,y_pred_word2vec))  
    print(type(model).__name__,'F1 Score is       : ' ,f1_score(y_test_word2vec,y_pred_word2vec))
    precision_sc.append(precision_score(y_test_word2vec,y_pred_word2vec))
    recall_sc.append(recall_score(y_test_word2vec,y_pred_word2vec))
    f1_sc.append(f1_score(y_test_word2vec,y_pred_word2vec))
    print('**************************************************************')

from sklearn.model_selection import GridSearchCV
param_grid = {
    "max_depth": [3, 4, 5, 7],
    "learning_rate": [0.1, 0.15, 0.2],
    "gamma": [0, 0.25, 1],
    "reg_lambda": [0, 1, 10],
    "subsample": [0.8],
    "colsample_bytree": [0.5],   
}

# Init classifier
xgb_cl = xgb.XGBClassifier(objective= "multi:softmax", num_class = 2)

# Init Grid Search
grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=6, scoring="roc_auc")

# Fit
grid_cv.fit(X_train_word2vec, y_train_word2vec)

grid_cv.best_score_

model = xgb.XGBClassifier(
    **grid_cv.best_params_,
    objective="multi:softmax",
    num_class = 2
)

model.fit(X_train_word2vec, y_train_word2vec)

predictions = model.predict(X_test_word2vec)

a= precision_score(y_test_word2vec,predictions)
b=recall_score(y_test_word2vec,predictions)
c=f1_score(y_test_word2vec,predictions)
d=accuracy_score(y_test_word2vec,predictions)
e=grid_cv.best_score_

print('XGboost Train Score is: ',e)
train_sc.append(float(e))

print('XGboost Test Score is: ',d )
test_sc.append(float(d))

print('XGboost Precision is      : ', a)   
print('XGboost Recall is         : ' ,b)  
print('XGboost F1 Score is       : ' ,c)

precision_sc.append(a)
recall_sc.append(b)
f1_sc.append(c)

'''
#XGBoost Approach 1
train = xgb.DMatrix(X_train_word2vec, label = y_train_word2vec)
test = xgb.DMatrix(X_test_word2vec, label = y_test_word2vec)
param = {
        'max_depth': 4,
        'eta': 0.2,
        'objective': 'multi:softmax',
        'num_class':2}
#epochs = [10, 20, 50, 100, 500]
#train_scores = []
#test_scores = []

epochs = 50

xgmodel = xgb.train(param, train, epochs)

predictions = xgmodel.predict(train)
xgb_acc_train = accuracy_score(y_train_word2vec,predictions)

print('XGboost Train Score is: ', xgb_acc_train)
train_sc.append(float(xgb_acc_train))

predictions = xgmodel.predict(test)
xgb_acc_test = accuracy_score(y_test_word2vec,predictions)

print('XGboost Test Score is: ', xgb_acc_test)
test_sc.append(float(xgb_acc_test))

a= precision_score(y_test_word2vec,predictions)
b=recall_score(y_test_word2vec,predictions)
c=f1_score(y_test_word2vec,predictions)
print('XGboost Precision is      : ', a)   
print('XGboost Recall is         : ' ,b)  
print('XGboost F1 Score is       : ' ,c)

precision_sc.append(precision_score(y_test_word2vec,predictions))
recall_sc.append(recall_score(y_test_word2vec,predictions))
f1_sc.append(f1_score(y_test_word2vec,predictions))
'''
#print('train scores',train_scores)
#print('test scores',test_scores)

fig = plt.figure(1)
plt.plot(epochs, train_scores,label="Train Scores XGBoost") # plot first line
plt.plot(epochs,test_scores,label="Test Scores XGBoost") # plot second line
plt.legend(["Train Scores XGBoost", "Test Scores XGBoost"], loc ="lower right")
plt.xlabel("# Epochs")
plt.ylabel("Score")

plt.show()

X = ['train_sc','test_sc','precision_sc','recall_sc', 'f1_sc'] 
scores = list(zip(train_sc,test_sc,precision_sc,recall_sc,f1_sc))
print(scores)
names = ['LogReg', 'SVM' , 'XGB']
X_axis = np.arange(len(X)) 
plt.bar(X_axis - 0.3, scores[0], 0.3, label = 'LogReg') 
plt.bar(X_axis, scores[1], 0.3, label = 'SVM') 
plt.bar(X_axis + 0.3, scores[2], 0.3, label = 'XGB')   
plt.xticks(X_axis, X) 
plt.xlabel("scoring methods") 
plt.ylabel("value") 
plt.legend( loc ="lower right") 
plt.show() 
 
	
cv = KFold(n_splits=6, random_state=1, shuffle=True)

for model in models:
    scores = cross_val_score(model, embeddings, list_labels, scoring='accuracy', cv=cv, n_jobs=-1)
    # report performance
    print(type(model).__name__,' 6-fold cross validation accuracy: %.3f ' % (mean(scores)))
#print('XGBoost 6-fold cross validation accuracy: 0.818 ',xgb_acc)
#print('LSTM 6-fold cross validation accuracy: 0.751',lstm_acc)

#lstm
"""

vocab_len = max([len(i) for i in tweet["tokens"]])

X_train_word2vec1, X_test_word2vec1, y_train_word2vec1, y_test_word2vec1 = train_test_split(tweet['tokens'], list_labels,
                                                                                        test_size=0.2, random_state=2022)

emb = [s for s in X_train_word2vec1]
emb.extend(s for s in X_test_word2vec1)
w2v =  gensim.models.Word2Vec(
    emb, 
    min_count=2,
    workers=4, 
    window =5
)


def get_word2vec_embeddings1(vectors, clean_questions, generate_missing=False):
    embeddings = [get_average_word2vec(x, vectors, generate_missing=generate_missing) for x in clean_questions]
    return list(embeddings)

embeddings1 = get_word2vec_embeddings1(word2vec, X_train_word2vec1)
embeddingstest = get_word2vec_embeddings1(word2vec, X_test_word2vec1)

def BiDirLSTM(vocSize, inpShape, seeds = 2022):
    np.random.seed(seeds)
    set_seed(seeds)
    inp = Input(shape = (inpShape), name = "input")
    emb = Embedding(
        input_dim = vocSize,
        output_dim = 4
    )(inp)
    drop = Dropout(0.4)(emb)#0.3
    biLstm = Bidirectional(
        LSTM(
            units=16,#16
            activation='tanh',
            return_sequences = True,
            stateful=False,
            recurrent_dropout = 0.4,
            dropout=0.4
        )
    )(drop)
    biLstm = Bidirectional(
        LSTM(
            units = 8,
            activation ='tanh',
            return_sequences = False,
            stateful = False,
            recurrent_dropout = 0.3,#0.3
            dropout = 0.3
        )
    )(biLstm)

    out = Dense(units = 1, activation = "sigmoid")(biLstm)

    m = Model(inputs = inp, outputs = out)
    m.summary()
    return m

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor = 'val_loss', 
    verbose = 1, 
    save_best_only = True
)
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.2, 
    verbose = 1, 
    patience = 5,                        
    min_lr = 0.001
)

model = BiDirLSTM(vocab_len, 300)
optimizer = Adamax(
    lr=0.015, 
    decay=0.0002, 
    clipvalue=10
)
loss = BinaryCrossentropy(label_smoothing=0.01)

model.compile(
    optimizer = optimizer, 
    loss = loss, 
    metrics = ["accuracy"]
)
history = model.fit(
    x = np.asarray(embeddings1), 
    y = np.asarray(y_train_word2vec1),
    validation_data = (np.asarray(embeddingstest), np.asarray(y_test_word2vec1)),
    epochs = 7, 
    batch_size = 96, 
    shuffle = True,
    verbose = 1,
    #callbacks = [reduce_lr, checkpoint]
)
"""

embed_dim = 32
lstm_out = 32
model = Sequential()
model.add(Embedding(max_features, embed_dim,input_length = 300))
model.add(Dropout(0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.4))
model.add(Dense(1,activation='sigmoid'))
adam = Adam(learning_rate=0.002)
model.compile(loss = 'binary_crossentropy', optimizer=adam ,metrics = ['accuracy'])
print(model.summary())

model.fit(np.array(X_train_word2vec1), np.array(y_train_word2vec1), epochs = 10, batch_size=32, validation_data=(np.array(X_test_word2vec1), np.array(y_test_word2vec1)))
