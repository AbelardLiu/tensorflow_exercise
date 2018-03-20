
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

pos_file = 'pos.txt'
neg_file = 'neg.txt'

def create_lexicon(pos_file, neg_file):
    lex = []

    def process_file(f):
        with open(pos_file, 'r') as f:
            lex = []
            lines = f.readlines()

            for line in lines:
                words = word_tokenize(line.lower())
                lex += words
            return lex

    lex += process_file(pos_file)
    lex += process_file(neg_file)

    lemmatizer = WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]

    word_count = Counter(lex)

    lex = []
    for word in word_count:
        if word_count[word] < 2000 and word_count[word] > 20:
            lex.append(word)

    return lex


lex  = create_lexicon(pos_file, neg_file)

def normalize_dataset(lex):
    dataset = []

    def string_to_vector(lex, review, clf):
        words = word_tokenize(line.lower())
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex))
        for word in words:
            if word in lex:
                features[lex.index(word)] = 1
        return [features,clf]

    with open(pos_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [1,0])
            dataset.append(one_sample)

    with open(neg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            one_sample = string_to_vector(lex, line, [0,1])
            dataset.append(one_sample)

    return dataset

dataset = normalize_dataset(lex)
random.shuffle(dataset)

test_size = int(len(dataset) * 0.1)

dataset = np.array(dataset)

train_dataset = dataset[:-test_size]
test_dataset = dataset[-test_size:]

n_input_layer = len(lex)

n_layer_1 = 10000
n_layer_2 = 10000

n_output_layer = 2

def neural_network(data):
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), "b_": tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output

batch_size = 50

X = tf.placeholder('float', [None, len(train_dataset[0][0])])
Y = tf.placeholder('float')

def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=predict))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    epochs = 13
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        epoch_loss = 0

        i = 0
        random.shuffle(train_dataset)
        train_x = dataset[:, 0]
        train_y = dataset[:,1]
        for epoch in range(epochs):
            while i < len(train_x):
                start = i
                end = i + batch_size 

                batch_x = train_x[start:end]
                batch_y = train_y[start:end]

                _, c = session.run([optimizer, cost_func], feed_dict={X:list(batch_x), Y:list(batch_y)})
                epoch_loss += c
                i += batch_size

            print(epoch, " : ", epoch_loss)

        test_x = test_dataset[:,0]
        test_y = test_dataset[:,1]
        correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('accuracy: ',accuracy.eval({X:list(test_x), Y:list(test_y)}))


train_neural_network(X,Y)
