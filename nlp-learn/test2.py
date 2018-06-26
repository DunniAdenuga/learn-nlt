from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

import urllib.request
import nltk

#Get a page from the internet
response = urllib.request.urlopen('http://php.net/')
html = response.read()

#remove html tags, tokenize
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(strip=True)
tokens = [t for t in text.split()]

#remove english stopwords
stop_w = stopwords.words('english')
cleanTokens = tokens[:]
for token in tokens:
    if token in stop_w:
        cleanTokens.remove(token)

freq = nltk.FreqDist(cleanTokens)
'''
print(freq)
for key,val in freq.items():
    print(str(key) + ':' + str(val))

freq.plot(25, cumulative=False)
'''

#tokenize using nltk
nltkText = "Hello Mr. Alec, how are you ? I hope you love afro-beats! Say hi to Jimi for me. "
print(sent_tokenize(nltkText))
print(word_tokenize(nltkText))

#synonyms and antonyms
syn = wordnet.synsets("small")
print(syn[0].definition())
print(syn[0].examples())
synonyms = []
antonyms = []
for s in syn:
    for lemma in s.lemmas():
        if lemma.antonyms():
            antonyms.append(lemma.antonyms()[0].name())
        else:
            synonyms.append(lemma.name())
print(synonyms)
print(antonyms)

#stemming using nltk
stemmer  = PorterStemmer()
stemmer2 = SnowballStemmer("english")
stemmer3 = LancasterStemmer()
lemmatizer = WordNetLemmatizer()

print(stemmer.stem("increases"))#returns increas for increases
print(stemmer2.stem("increases"))
print(stemmer3.stem("playful"))
print(lemmatizer.lemmatize("increases"))
print(lemmatizer.lemmatize("playful", pos="v"))

print("---------------------------")

#difference between stemming and lemmatizing
print(stemmer.stem("stones"))
print(stemmer.stem("speaking"))
print(stemmer.stem("bedroom"))
print(stemmer.stem("jokes"))
print(stemmer.stem("lisa"))
print(stemmer.stem("purple"))
print("---------------------------")
print(lemmatizer.lemmatize("stones"))
print(lemmatizer.lemmatize("speaking"))
print(lemmatizer.lemmatize("bedroom"))
print(lemmatizer.lemmatize("jokes"))
print(lemmatizer.lemmatize("lisa"))
print(lemmatizer.lemmatize("purple"))
