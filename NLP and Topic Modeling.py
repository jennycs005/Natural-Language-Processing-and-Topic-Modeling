###
In this project, we use unsupervised learning models to cluster unlabeled documents into different groups, visualize the results and identify their latent topics/structures.
###

#Contents
#Part 1: Load Data
#Part 2: Tokenizing and Stemming
#Part 3: TF-IDF
#Part 4: K-means clustering
#Part 5: Topic Modeling - Latent Dirichlet Allocation

#####Part 1: Load Data#####
###########################
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import nltk
import re
import os

from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import lda

#Read data from files. In summary, we have 100 titles and 100 synoposes (combined from imdb and wiki).
#import three lists: titles and wikipedia synopses
titles = open('./title_list.txt').read().split('\n')
titles = titles[:100] #ensures that only the first 100 are read in

#The wiki synopses and imdb synopses of each movie is seperated by the keywords "BREAKS HERE". 
#Each synoposes may consist of multiple paragraphs.
synopses_wiki = open('./synopses_list_wiki.txt').read().split('\n BREAKS HERE')
synopses_wiki = synopses_wiki[:100]

synopses_imdb = open('./synopses_list_imdb.txt').read().split('\n BREAKS HERE')
synopses_imdb = synopses_imdb[:100]

#Combine imdb and wiki to get full synoposes for the top 100 movies. 
synopses = []
for i in range(len(synopses_wiki)):
    item = synopses_wiki[i] + synopses_imdb[i]
    synopses.append(item)
    
#Because these synopses have already been ordered in popularity order, 
#we just need to generate a list of ordered numbers for future usage.
ranks = range(len(titles))

#####Part 2: Tokenizing and Stemming#####
#########################################

#Load stopwords and stemmer function from NLTK library. Stop words are words like "a", "the", or "in" which don't convey #significant meaning. Stemming is the process of breaking a word down into its root.

# Use nltk's English stopwords.
stopwords = nltk.corpus.stopwords.words('english')

print "We use " + str(len(stopwords)) + " stop-words from nltk library."
print stopwords[:10]


