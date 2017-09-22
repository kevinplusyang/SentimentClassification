
# coding: utf-8

#future is the missing compatibility layer between Python 2 and Python 3. 
#It allows you to use a single, clean Python 3.x-compatible codebase to 
#support both Python 2 and Python 3 with minimal overhead.
from __future__ import absolute_import, division, print_function


#encoding. word encodig
import codecs
#finds all pathnames matching a pattern, like regex
import glob
#log events for libraries
import logging
#concurrency
import multiprocessing
#dealing with operating system , like reading file
import os
#pretty print, human readable
import pprint
#regular expressions
import re


#natural language toolkit
import nltk
#word 2 vec
import gensim.models.word2vec as w2v
#dimensionality reduction
import sklearn.manifold
#math
import numpy as np
#plotting
import matplotlib.pyplot as plt
#parse dataset
import pandas as pd
#visualization
import seaborn as sns
#classification
from sklearn import svm

get_ipython().magic(u'pylab inline')

#stopwords like the at a an, unnecesasry
#tokenization into sentences, punkt 
#http://www.nltk.org/
nltk.download("punkt")
nltk.download("stopwords")


#convert into list of words
#remove unecessary characters, split into words, no hyhens and shit
#split into words
def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    word_u = [word.decode("utf-8") for word in words]
    return word_u
#take 1 file
def tokenizer(file_name1):
    #initialize rawunicode , we'll add all text to this one bigass file in memory
    corpus_raw = u""
    
    with codecs.open(file_name1,"r", "utf-8") as txt:
        corpus_raw += txt.read()

    print ("Corpus is now {0} characters long".format(len(corpus_raw)))
    #tokenizastion! saved the trained model here
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #tokenize into sentences
    raw_sentences = tokenizer.tokenize(corpus_raw)
    print(len(raw_sentences))
    #for each sentece, sentences where each word is tokenized
    sentences = []
    for raw_sentence in raw_sentences:
#         if len(raw_sentence) > 0:
        sentences.append(sentence_to_wordlist(raw_sentence.lower()))
    print(len(sentences))
    return sentences

def read_file(file_name):
    with open(file_name) as f:
        sentences = []

        for line in f:
            sentences.append(sentence_to_wordlist(line))


    return sentences


def read_file_2(file_name1,file_name2):
    sentences = []
    label = []
    with open(file_name1) as f:
        for line in f:
            sentences.append(sentence_to_wordlist(line.lower()))
            label.append(0)
    
    with open(file_name2) as f:
        for line in f:
            sentences.append(sentence_to_wordlist(line.lower()))
            label.append(1)

    return sentences, label


    

#take 2 files
def tokenizer_2(file_name1, file_name2):
    #initialize rawunicode , we'll add all text to this one bigass file in memory
    corpus_raw = u""
    
    with codecs.open(file_name1,"r", "utf-8") as txt:
        corpus_raw += txt.read()
        
    with codecs.open(file_name2,"r", "utf-8") as txt:
        corpus_raw += txt.read()
        

    print ("Corpus is now {0} characters long".format(len(corpus_raw)))
    #tokenizastion! saved the trained model here
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #tokenize into sentences
    raw_sentences = tokenizer.tokenize(corpus_raw)
    
    #for each sentece, sentences where each word is tokenized
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentence_to_wordlist(raw_sentence.lower()))
    return sentences

#generate label
def label(file_name1, file_name2):
    #initialize rawunicode , we'll add all text to this one bigass file in memory
    corpus_raw_0 = u""
    corpus_raw_1 = u""
    label = []
    with codecs.open(file_name1,"r", "utf-8") as txt:
        corpus_raw_0 += txt.read()
        
    with codecs.open(file_name2,"r", "utf-8") as txt:
        corpus_raw_1 += txt.read()
        

#     print ("Corpus is now {0} characters long".format(len(corpus_raw)))
    #tokenizastion! saved the trained model here
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #tokenize into sentences
    raw_sentences_0 = tokenizer.tokenize(corpus_raw_0)
    raw_sentences_1 = tokenizer.tokenize(corpus_raw_1)

    for raw_sentence in raw_sentences_0:
        if len(raw_sentence) > 0:
            label.append(0)
    for raw_sentence in raw_sentences_1:
        if len(raw_sentence) > 0:
            label.append(1)
    label = np.array(label)
    print(label.shape)
    return label


# Dimensionality of the resulting word vectors.
#more dimensions mean more traiig them, but more generalized
num_features = 50

#
# Minimum word count threshold.
min_word_count = 2

# Number of threads to run in parallel.
num_workers = multiprocessing.cpu_count()

# Context window length.
context_size = 5

# Downsample setting for frequent words.
#rate 0 and 1e-5 
#how often to use
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
seed = 1


review2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)



sentences,training_label =  read_file_2("./SentimentDataset/train/pos.txt","./SentimentDataset/train/neg.txt")
print(sentences[0])
print(len(sentences))

review2vec.build_vocab(sentences)



print("Word2Vec vocabulary length:", len(review2vec.wv.vocab))



#get the count of examples
corpus_count = review2vec.corpus_count

#train model on sentneces
review2vec.train(sentences,total_examples=corpus_count,epochs=review2vec.iter)


#save model
if not os.path.exists("trained"):
    os.makedirs("trained")


review2vec.save(os.path.join("trained", "review2vec.w2v"))


#load model
review2vec = w2v.Word2Vec.load(os.path.join("trained", "review2vec.w2v"))



def sen_vector(sentences):
    #set array 
    whole_array = []
    
    #get average vector of pos sentences
    for sentence in sentences:
        vector = np.zeros(50)
        count = 0
        count_noindex = 0

        for word in sentence:
            try:
                vector=review2vec.wv[word]
                count += 1
            except:
                count_noindex += 1

        if count >0:
            avg_vector = vector/count
        else:
            avg_vector = vector
        
        whole_array.append(avg_vector)
        
    whole_array = np.array(whole_array)
    
    print(whole_array.shape)
    return whole_array
           

training_vector_set = sen_vector(sentences)

test_token_set = read_file('./SentimentDataset/test/test.txt')

test_vector_set = sen_vector(test_token_set)



#use SVM to train classification model
clf = svm.SVC()
clf.fit(training_vector_set, training_label)
prediction = clf.predict(training_vector_set)
n = len(prediction)
count_right = 0
for i in range(n):
    if prediction[i]== training_label[i]:
        count_right += 1
print("the accuracy of SVM with training dataset: " + str(float(count_right)/n))


from sklearn import linear_model
rg = linear_model.LogisticRegression()
rg.fit(training_vector_set, training_label)
prediction = rg.predict(training_vector_set)
n = len(prediction)
count_right = 0
for i in range(n):
    if prediction[i]== training_label[i]:
        count_right += 1
print("the accuracy of LogisticRegression with training dataset: " + str(float(count_right)/n))


#predict test using Logistic Regression model
test_label = rg.predict(test_vector_set)
n = len(test_label)
Id = [i+1 for i in range(n)]

#output test
output = pd.DataFrame(Id,columns = ['Id'] )
output['Prediction'] = test_label
output.to_csv('prediction_word_embedding.csv',index = False)

#squash dimensionality to 2
#https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)


#put it all into a giant matrix
all_word_vectors_matrix = review2vec.wv.syn0


#train t sne
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)


#plot point in 2d space
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[review2vec.wv.vocab[word].index])
            for word in review2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)


#plot
sns.set_context("poster")

points.plot.scatter("x", "y", s=10, figsize=(20, 12))


def plot_region(x_bounds, y_bounds):
    slice = points[
        (x_bounds[0] <= points.x) &
        (points.x <= x_bounds[1]) & 
        (y_bounds[0] <= points.y) &
        (points.y <= y_bounds[1])
    ]
    
    ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))
    for i, point in slice.iterrows():
        ax.text(point.x + 0.005, point.y + 0.005, point.word, fontsize=11)


print(points.loc[points['word'] == 'good'])


print(points.loc[points['word'] == 'great'])



plot_region(x_bounds=(-2, -0.5), y_bounds=(2, 3))


review2vec.wv.most_similar("good")


thrones2vec.most_similar("wonderful")


thrones2vec.most_similar("fine")


#distance, similarity, and ranking
def nearest_similarity_cosmul(start1, end1, end2):
    similarities = review2vec.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


nearest_similarity_cosmul('good','best','worst')





