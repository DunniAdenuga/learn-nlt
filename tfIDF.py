import nltk
import urllib.request

from bs4 import BeautifulSoup
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

'''
pre-process text
'''
#retrieve raw text
response = urllib.request.urlopen("http://www.africa.upenn.edu/Articles_Gen/Letter_Birmingham.html")
html = response.read()
soup = BeautifulSoup(html, "html.parser")
text = soup.get_text(strip=True)

#tokenize text
tokens = word_tokenize(text)

#remove stop words
stopWords = stopwords.words("english")
cleanTokens = [t for t in tokens if not t in stopWords]

#stem
stemmer = SnowballStemmer("english")
processed_text = [stemmer.stem(c) for c in cleanTokens]
