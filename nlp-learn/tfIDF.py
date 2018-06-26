import nltk
import string
import csv
import os
import operator
import collections
import urllib.request

from bs4 import BeautifulSoup
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.tokenize import line_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

'''
pre-process text
'''
#retrieve raw text

#test with lines from Letter from birmingham jail article
'''
response = urllib.request.urlopen("http://www.africa.upenn.edu/Articles_Gen/Letter_Birmingham.html")
html = response.read()
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(strip=True)
documents = line_tokenize(text) #tfIdf needs documents to work
'''

#test with Luther season 1 episodes
documents = {}
path = "./luther/season1"
for subdir, dirs, files in os.walk(path):
    for file in files:
        file_loc = subdir + os.path.sep + file
        episode = open(file_loc, 'r')
        document = episode.read()
        #print(document)
        lower_document = document.lower()
        #print(lower_document)
        noPunc_document = lower_document.translate(str.maketrans('','', string.punctuation))
        #print(noPunc_document)
        #print('\n')
        documents[file] = noPunc_document

#print(documents)

#tokenize text
def tokenize(text):
    tokens = word_tokenize(text)
    #remove stop words
    #stop_words = stopwords.words("english")
    #cleanTokens = [t for t in tokens if not t in stop_words]
    #finalTokens = stemTokens(cleanTokens)
    finalTokens = stemTokens(tokens)
    return finalTokens

def stemTokens(tokens):
    #stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    #stemmed_tokens = [stemmer.stem(c) for c in tokens]
    stemmed_tokens = [lemmatizer.lemmatize(c) for c in tokens]
    return stemmed_tokens

#tfIdf with Scikit sklearn
#tfidf = TfidfVectorizer(tokenizer=tokenize)
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words="english")
tfs = tfidf.fit_transform(documents.values())
feature_names = tfidf.get_feature_names()
#print(len(documents))
#print(tfs)
#print('\n')
#print(feature_names)

'''
#send to csv file
with open("./output/tfIdf_result.csv", "w") as file:
    writer = csv.writer(file, delimiter=",")
    writer.writerow(["Document", "Word", "Score"])

    document_id = 1
    for doc in tfs.todense():
        print("Document %d"  %document_id)
        word_id = 0
        for score in doc.tolist()[0]:
            if score > 0:
                word = feature_names[word_id]
                writer.writerow([document_id, word, score])
            word_id += 1
        document_id += 1
'''
allEpisodes = collections.defaultdict(dict)

episode_id = 1
for l_episode in tfs.todense():
    #print("Episode %d" %episode_id)
    word_id = 0
    for tfIdfScore in l_episode.tolist()[0]:
        if tfIdfScore > 0:
            word = feature_names[word_id]
            #print(word + " " + str(tfIdfScore))
            allEpisodes["episode" + str(episode_id)][word] = tfIdfScore
        word_id +=1
    episode_id += 1


#print(allEpisodes["episode1"])

i = 1
for episodE in allEpisodes:
    print("Episode %d" %i)
    sortedEp = sorted(allEpisodes["episode"+str(i)].items(), key= lambda x: x[1], reverse=True)
    #sortedEp = sorted(episodE.items(), key= lambda x: x[1], reverse=True)
    print(sortedEp[:7]) #top 5 words in episodes
    i +=1



#print(sortedEp)
