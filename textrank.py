'''
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
import re


we_df = pd.read_hdf('mini.h5', start = 0, stop = 100) # (362891, 300)

pi(we_df.shape)

words = we_df.index

pi(words)

pi(words[50000])
pi(we_df.iloc[50000])

mes = 'This is some demo text, which has some spe$hial charecters! And numbers 10, also mixed with text, like - numb3r and number34. Just for testing. #peace_out!'

def get_text_vector(text):
    re.findall(r'[a-zA-Z]+', )

'''

# python textrank.py
# textrank (using conceptnet word ventors/embeddings and cosinesimilarity)

import numpy as np
import pandas as pd
'''
import time
from sklearn.metrics import confusion_matrix
import json
import re
'''

cnnb_df = pd.read_hdf('mini.h5')
# cnnb_df = cnnb_df/59 # not req. (takes ~1.3sec)

def pi(a, b = None):
    if b:
        print('\n', b, a, '\n', type(a))
    else:
        print('\n', a, '\n', type(a))



'''
mes = 'This is some demo text, which has some spe$hial characters! And numbers 10, also mixed with text, like - numb3r and number34. Just for testing. #peace_out!'
#words = ['This', 'is', 'some', 'demo', 'text', 'which', 'has', 'some', 'spe', 'hial', 'characters', 'And', 'numbers', '10', 'also', 'mixed', 'with', 'text', 'like', 'numb', 'r', 'and', 'number', 'Just', 'for', 'testing', 'peace_out']
mes2 = 'demo text, which only has plain characters and no numbers, also not mixed with text, like - numb3r and number34. Just for testing.'

#vec = text_to_vec(list(map(lambda x: x.lower(), words)))
words = re.findall(r'[a-zA-Z]+', mes.lower())
words2 = re.findall(r'[a-zA-Z]+', mes2.lower())
#pi(words)
vec = text_to_vec(words)
vec2 = text_to_vec(words2)
sim = get_cosine_similarity(vec, vec2)

pi(sim)
pi(keyerror_list)
'''

# Read data

df = pd.read_csv('demo_articles.csv')
df.head()
df['article_text'][0]


# Form sentences

from nltk.tokenize import sent_tokenize
sentences = []

for s in df['article_text']:
    sentences.append(sent_tokenize(s))

sentences = [y for x in sentences for y in x] # flatten list / 2d to 1d / combine


# Text preprocessing

# remove punctuations, numbers and special characters
clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

# make alphabets lowercase
clean_sentences = [s.lower() for s in clean_sentences]


import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

print(stop_words)
print(len(stop_words))

# function to remove stopwords
def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

# remove stopwords from the sentences
clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]


# Vector Representation of Sentences

# Form vector from text
keyerror_list = []
def word_to_vec(word):
    vec = pd.Series(np.zeros(shape=(300)))
    try:
        wuri = '/c/en/' + word
        vec = cnnb_df.loc[wuri]
    except KeyError:
        keyerror_list.append(wuri)
    return vec

sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        #v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split())+0.001)
        v = sum([word_to_vec(word) for word in i.split()]) / (len(i.split())+0.001)
    else:
        v = pd.Series(np.zeros(shape=(300)))
    sentence_vectors.append(v)


# Similarity Matrix Preparation


# similarity matrix
sim_mat = np.zeros([len(sentences), len(sentences)])

'''
from sklearn.metrics.pairwise import cosine_similarity
'''

# Vector comparision
def get_cosine_similarity(vec1, vec2):
    # =a.b/|a||b| =dot_prod/vec_mag
    try:
        return sum(vec1 * vec2) / ( pow(sum(vec1*vec1), 0.5) * pow(sum(vec2*vec2), 0.5) )
    except ZeroDivisionError:
        return 0


for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            #sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]
            sim_mat[i][j] = get_cosine_similarity(sentence_vectors[i], sentence_vectors[j])

'''
sim_mat[i][j] = get_cosine_similarity(sentence_vectors[i], sentence_vectors[j])
__main__:3: RuntimeWarning: invalid value encountered in double_scalars
'''


# Applying PageRank Algorithm

import networkx as nx

nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph, max_iter=100) # default max_iter is 100



# Summary Extraction

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

# Extract top 10 sentences as the summary
for i in range(10):
    print(ranked_sentences[i][1])


