
import sys
reload(sys)
sys.setdefaultencoding('latin-1')

import io
import os
import random
import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

f = io.open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()

def get_random_line(file, point):
    file.seek(point)
    file.readline()
    return file.readline()

def get_n_random_line(file_name, n=150):
    lines =[]
    file = io.open(file_name, encoding='latin-1')
    total_bytes = os.stat(file_name).st_size
    for i in range(n):
        random_point = random.randint(0, total_bytes)
        lines.append(get_random_line(file, random_point))
    file.close()
    return lines

def get_test_dataset(test_file):
    with io.open(test_file, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y

test_x, test_y = get_test_dataset('testing.csv')

n_input_layer = len(lex)

n_layer_1 = 4000
n_layer_2 = 4000

n_output_layer = 3

def cnn_neural_network(X, dropout_keep_prob, num_classes):
    with tf.device('cpu:0'), tf.name_scope('embedding'):
        embedding_size = 64
        W = tf.Variable(tf.random_uniform([n_input_layer, embedding_size], -1.0,1.0))
        embedded_chars = tf.nn.embedding_lookup(W, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    num_filters = 64
    filter_sizes = [3,4,5]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
            conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1,1,1,1], padding="VALID")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            pooled = tf.nn.max_pool(h, ksize=[1, n_input_layer - filter_size + 1, 1, 1], strides=[1,1,1,1], padding="VALID")
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
        output = tf.nn.xw_plus_b(h_drop, W, b)

    return output

def neural_network(data):
    layer_1_w_b = {'w_':tf.Variable(tf.random_normal([n_input_layer, n_layer_1])), 'b_':tf.Variable(tf.random_normal([n_layer_1]))}
    layer_2_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_1, n_layer_2])), 'b_':tf.Variable(tf.random_normal([n_layer_2]))}
    layer_output_w_b = {'w_':tf.Variable(tf.random_normal([n_layer_2, n_output_layer])), 'b_':tf.Variable(tf.random_normal([n_output_layer]))}

    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output

num_classes = 3
#X = tf.placeholder(tf.int32, [None, n_input_layer])
X = tf.placeholder(tf.float32, [None, n_input_layer])
Y = tf.placeholder(tf.float32, [None, num_classes])
batch_size = 90

dropout_keep_prob = tf.placeholder(tf.float32)

def train_cnn_neural_network(X, Y, dropout_keep_prob, num_classes):
    output = cnn_neural_network(X, dropout_keep_prob, num_classes)

    optimizer = tf.train.AdamOptimizer(1e-3)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output))
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        lemmatizer = WordNetLemmatizer()
        i = 0
        while True:
            batch_x = []
            batch_y = []

            try:
                print("debug")
                lines = get_n_random_line('training.csv', batch_size)
                for line in lines:
                    label = line.split(':%:%:%:')[0]
                    tweet = line.split(':%:%:%:')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]

                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1

                    batch_x.append(list(features))
                    batch_y.append(eval(label))

                print("Debug")
                _, loss_ = sess.run([train_op, loss], feed_dict={X:batch_x, Y:batch_y, dropout_keep_prob:0.5})
                print(loss_)
            except Exception as e:
                print(e)

            if i-1 % 10 == 0:
                predictions = tf.argmax(output, 1)
                correct_predictions = tf.equal(predictions, tf.argmax(Y,1))
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
                accur = sess.run(accuracy, feed_dict={X:test_x, Y:test_y, dropout_keep_prob:1.0})
                print("accuracy: ", accur)
            i += 1


def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=predict))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    with tf.Session() as session:
        tf.global_variables_initializer().run()

        lemmatizer = WordNetLemmatizer()
        saver = tf.train.Saver()
        i = 0
        pre_accuracy = 0
        while True:
            batch_x = []
            batch_y = []

            try:
                lines = get_n_random_line('training.csv', batch_size)
                for line in lines:
                    label = line.split(':%:%:%:')[0]
                    tweet = line.split(':%:%:%:')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]

                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1


                    batch_x.append(list(features))
                    batch_y.append(eval(label))

                session.run([optimizer, cost_func], feed_dict={X:batch_x, Y:batch_y})
            except Exception as e:
                print(e)

            if i > 100:
                correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                accuracy = accuracy.eval({X:test_x, Y:test_y})
                if accuracy > pre_accuracy:
                    print("accuracy: ", accuracy)
                    pre_accuracy = accuracy
                    saver.save(session, 'model.ckpt')
                i = 0
                if accuracy >0.8:
                    break
            i += 1

train_neural_network(X,Y)
#train_cnn_neural_network(X, Y, dropout_keep_prob, num_classes)
