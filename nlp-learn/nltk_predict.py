#predict male or female
import nltk
from nltk.corpus import names

#Load data and training
names = ([(name, 'male') for name in names.words('male.txt')] +
        [(name, 'female') for name in names.words('female.txt')])
def gender_features(word):
    return {'last_letter': word[-1]}

featuresets = [(gender_features(n), g) for (n, g) in names]


train_set = featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set)

#Predict
name = input("Name:" )
print(classifier.classify(gender_features('Frankie')))


'''
Sentiment Analysis
'''
