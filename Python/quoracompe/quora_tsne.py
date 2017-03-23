
# coding: utf-8

# # Visualizing Word Vectors with t-SNE
# 
# TSNE is pretty useful when it comes to visualizing similarity between objects. It works by taking a group of high-dimensional (100 dimensions via Word2Vec) vocabulary word feature vectors, then compresses them down to 2-dimensional x,y coordinate pairs. The idea is to keep similar words close together on the plane, while maximizing the distance between dissimilar words. 
# 
# ### Steps
# 
# 1. Clean the data
# 2. Build a corpus
# 3. Train a Word2Vec Model
# 4. Visualize t-SNE representations of the most common words 
# 
# Credit: Some of the code was inspired by this awesome [NLP repo][1]. 
# 
# 
# 
# 
#   [1]: https://github.com/rouseguy/DeepLearningNLP_Py

# In[1]:

import pandas as pd
pd.options.mode.chained_assignment = None 
import re
import nltk
from nltk.corpus import stopwords

from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

data = pd.read_csv('D:/Users/pjpan.CN1/Desktop/train.csv').sample(50000, random_state=23)


# In[2]:

# STOP_WORDS = nltk.corpus.stopwords.words()

STOP_WORDS = set(stopwords.words('english'))

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['question1', 'question2']:
        data[col] = data[col].apply(clean_sentence)
    
    return data

data = clean_dataframe(data)
data.head(5)


# In[3]:

def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['question1', 'question2']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(data)        
corpus[0:2]


# # Word 2 Vec
# 
# The Word to Vec model produces a vocabulary, with each word being represented by an n-dimensional numpy array (100 values in this example)

# In[4]:

model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=200, workers=4)
model.wv['trump']


# In[5]:

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


# In[6]:

tsne_plot(model)


# In[7]:

# A more selective model
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=500, workers=4)
tsne_plot(model)


# In[8]:

# A less selective model
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=100, workers=4)
tsne_plot(model)


# # It's Becoming Hard to Read
# 
# With a dataset this large, its difficult to make an easy-to-read TSNE visualization. What you can do is use the model to look up the most similar words from any given point. 

# In[9]:

model.most_similar('trump')


# In[10]:

model.most_similar('universe')


# # The End
# 
# Good luck!
