
import io
import sys
reload(sys)
sys.setdefaultencoding("utf8")

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

org_train_file = "training.1600000.processed.noemoticon.csv"
org_test_file = "testdata.manual.2009.06.14.csv"

def usefull_filed(org_file, output_file):
    output = io.open(output_file, "w")
    with io.open(org_file, buffering=10000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"', '')
                clf = line.split(',')[0]
                if clf == '0':
                    clf = [0, 0, 1]
                elif clf == '2':
                    clf = [0, 1, 0]
                elif clf == '4':
                    clf = [1, 0, 0]

                tweet = line.split(',')[-1]
                outputline = str(clf) + ':%:%:%:' + tweet
                output.write(outputline)
        except Exception as e:
            print(e)

    output.close()

usefull_filed(org_train_file, "training.csv")
usefull_filed(org_test_file, "testing.csv")

def create_lexicon(train_file):
    lex = []
    lemmatizer = WordNetLemmatizer()
    with io.open(train_file, buffering=10000, encoding='latin-1') as f:
        try:
            count_word = {}
            for line in f:
                tweet = line.split(':%:%:%:')[1]
                words = word_tokenize(line.lower())
                for word in words:
                    word = lemmatizer.lemmatize(word)
                    if word not in count_word:
                        count_word[word] = 1
                    else:
                        count_word[word] += 1

            count_word = OrderedDict(sorted(count_word.items(), key=lambda t: t[1]))
            for word in count_word:
                if count_word[word] < 100000 and count_word[word] > 100:
                    lex.append(word)
        except Exception as e:
            print(e)
    return lex

lex = create_lexicon('training.csv')

with io.open('lexcion.pickle', 'wb') as f:
    pickle.dump(lex, f)

def string_to_vector(input_file, output_file, lex):
    output_f = io.open(output_file, 'w')
    lemmatizer = WordNetLemmatizer()
    with io.open(input_file, buffering=10000, encoding='latin-1') as f:
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]

            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            features = list(features)
            output_line = str(label) + ":" + str(features) +'\n'
            output_f.write(unicode(output_line))
    output_f.close()

#string_to_vector('training.csv', 'training.vec', lex)
#string_to_vector('testing.csv', 'testing.vec', lex)
