from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import random
import time
import re
import itertools
import _pickle as cPickle

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

cvs = 10
layers = 1
word_embed_size = 300
num_filters = 100
dropout_rate = 0.5
batch_size = 50
epochs = 100
num_classes = 2
type = 'mr'
langs = ['mn', 'ru', 'ar', 'no', 'ko', 'uk', 'it', 'fi', 'pl', 'fr']

def cnn_sentence(sequence_length, num_classes, vocab_size, embedding_size, 
        filter_sizes, num_filters, l2_reg_lambda, dropout_keep_prob, language):
    input_x = tf.placeholder(tf.int32, [None, sequence_length], name=language + "_input_x")
    
    embeddings = []
    all_embeddings = []
    
    with tf.name_scope('embed'):
        embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1, 0.1), name=language + "_embedding")
        embeddings.append(embedding)
        embedded_chars = tf.nn.embedding_lookup(embedding, input_x)
        all_embeddings.append(embedded_chars)
    
    all_embeddings = tf.cast(all_embeddings, tf.float32)
    all_embeddings = tf.transpose(all_embeddings, perm=[1,2,3,0])
    
    pooled_outputs = []
    regularizers = 0
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv-maxpool-%s' % i):
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name=language + "_W")
            b = tf.Variable(tf.constant(0.0, shape=[num_filters]), name=language + "_b")
            conv = tf.nn.conv2d(
                all_embeddings,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name=language + "_relu")
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length-filter_size+1, 1, 1],
                strides=[1, 1, 1, 1],
                padding="VALID",
                name=language + "_pool")
            pooled_outputs.append(pooled)
        
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    
    with tf.name_scope('dropout'):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob, name=language + "_h_drop")
    
    return input_x, embeddings, h_drop, regularizers

def batch_iter(data, batch_size, num_epochs, shuffle_indices, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 10
    lst = []
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            lst.append(shuffled_data[start_index:end_index])
    return lst

vs = {}
vs[type] = {}

languages = ['en'] + langs
x = cPickle.load(open('pickles/' + type + ".p", "rb"), encoding='latin1')
vectors = vs[type]
data, vocabs, maxes = x[0], x[1], x[2]
for language in languages:
    print('vector', language)
    if language not in vectors:
        vectors[language] = cPickle.load(open('vectors/' + type + '_' + language + '.p', 'rb'), encoding='latin1')

print('done')

for language in languages:
    print('cnn for language', language)
    
    vocab = vocabs[language]
    mx = maxes[language]
    len_vocab = len(vocab)
    
    for instance in data:
        ins = instance[language]
        if ins[0] != len_vocab:
            for i in range(4):
                ins.insert(0, len_vocab)
        while len(ins) < mx+8:
            ins.append(len_vocab)
        ins = ins[0:mx+8]
    
    maxes[language] += 8

accuracies = []
for cv in range(cvs):
    print('cv', cv)
    
    sess = tf.Session()
    with sess.as_default():
        random.seed(1234)
        np.random.seed(1234)
        tf.set_random_seed(1234)
        train = []
        test = []
        
        for instance in data:
            if instance['split'] == cv:
                test.append(instance)
            else:
                train.append(instance)
        
        train_y = []
        test_y = []
        
        for instance in train:
            train_y.append(instance['class'])
        for instance in test:
            test_y.append(instance['class'])
        
        train_y = np.array(train_y)
        test_y = np.array(test_y)

        train_xs = []
        test_xs = []
        
        inputs = []
        input_y = tf.placeholder(tf.float32, [None, num_classes], name=language + "_input_y")
        embeddings = {}
        h_drops = {}
        dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        is_training = tf.placeholder(tf.bool, [], name='is_training')
        regularizers = 0
        for language in languages:
            print('cnn for language', language)
            
            vocab = vocabs[language]
            mx = maxes[language]
            len_vocab = len(vocab)
            
            train_x = []
            test_x = []
            
            for instance in train:
                train_x.append(instance[language])
            for instance in test:
                test_x.append(instance[language])
            
            len_vocab += 1
            
            train_x = np.array(train_x)
            test_x = np.array(test_x)
            
            train_xs.append(train_x)
            test_xs.append(test_x)
            
            input_x, embedding, h_drop, regs = cnn_sentence(sequence_length=mx, num_classes=num_classes, 
                    vocab_size=len_vocab, embedding_size=word_embed_size, filter_sizes=[3,4,5],
                    num_filters=num_filters, l2_reg_lambda=0.0, dropout_keep_prob=dropout_keep_prob, language=language)
            
            inputs.append(input_x)
            embeddings[language] = embedding
            h_drops[language] = h_drop
            regularizers += regs
        
        X = {}
        Xb = {}
        x = {}
        V = {}
        Vb = {}
        W = tf.get_variable("W", shape=[300*len(languages), num_classes], initializer=xavier_initializer())
        U = {}
        T = {}
        for layer in range(layers):
            X[layer] = {}
            Xb[layer] = {}
            x[layer] = {}
            V[layer] = {}
            Vb[layer] = {}
            U[layer] = {}
            for language in languages:
                X[layer][language] = tf.get_variable("X" + language + str(layer), shape=[300, 300], initializer=xavier_initializer())
                Xb[layer][language] = tf.Variable(tf.constant(0.0, shape=[300]))
                x[layer][language] = tf.get_variable("x" + language + str(layer), shape=[300, 1], initializer=xavier_initializer())
                V[layer][language] = tf.get_variable("V" + language + str(layer), shape=[300*2, 300], initializer=xavier_initializer())
                Vb[layer][language] = tf.Variable(tf.constant(0.0, shape=[300]))
                U[layer][language] = tf.get_variable("U" + language + str(layer), shape=[300, 300], initializer=xavier_initializer())
    
        for language in languages:
            T[language] = tf.get_variable("T" + language, shape=[300, 1], initializer=xavier_initializer())
    
        for layer in range(layers):
            context_attended = {}
            contexts = []
            certainties = {}
        
            # SELF USABILITY
            for language in languages:
                certainties[language] = tf.nn.sigmoid(tf.matmul(h_drops[language], T[language]))
                certainties[language] = tf.nn.dropout(certainties[language], dropout_keep_prob)
            
            hX = {}
            hU = {}
            for language in languages:
                hX[language] = tf.matmul(h_drops[language], X[layer][language])
                hU[language] = tf.matmul(h_drops[language], U[layer][language])
            
            for language in languages:
                # RELATIVE USABILITY
                context = []
                for other_language in languages:
                    certain = certainties[other_language]
                    
                    a = tf.nn.tanh(hX[language] + hX[other_language]*certain + Xb[layer][language])
                    a = tf.nn.dropout(a, dropout_keep_prob)
                    
                    con = tf.matmul(a, x[layer][language])
                    context.append(con)
                context = tf.convert_to_tensor(context)
                context = tf.transpose(context, perm=[1,2,0])
                context = tf.nn.softmax(context)
                context2 = tf.transpose(context, perm=[2,0,1])
                contexts.append(context)
                
                weighted = []
                idx = 0
                for other_language in languages:
                    weight = tf.multiply(hU[other_language], context2[idx])
                    weighted.append(weight)
                    idx += 1

                convec = tf.convert_to_tensor(weighted)
                convec = tf.transpose(convec, perm=[1,0,2])
                convec = tf.reduce_sum(convec, axis=1)
                
                # VECTOR FIXING
                lang_with_context = tf.concat([h_drops[language], convec], axis=1)
                attention = tf.nn.sigmoid(tf.matmul(lang_with_context, V[layer][language]) + Vb[layer][language])
                attention = tf.nn.dropout(attention, dropout_keep_prob)
                
                context_attended[language] = tf.multiply(h_drops[language], attention)
            
            for language in languages:
                h_drops[language] = context_attended[language]
        
        contexts = tf.transpose(tf.convert_to_tensor(contexts), perm=[1,0,2,3]) 
        
        self_attended = []
        
        for i in range(len(languages)):
            self_attended.append(context_attended[languages[i]])
        
        self_attended = tf.concat(self_attended, axis=1)
        
        b1 = tf.Variable(tf.constant(0.0, shape=[num_classes]))
        scores = tf.nn.xw_plus_b(self_attended, W, b1)
        soft_scores = tf.nn.softmax(scores)
        predictions = tf.argmax(scores, 1, name="prediction")
        
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
            regularizers = 0
            for language in languages:
                regularizers += tf.nn.l2_loss(T[language])
            
            for layer in range(layers):
                for language in languages:
                    regularizers += tf.nn.l2_loss(V[layer][language])
                    regularizers += tf.nn.l2_loss(X[layer][language])
                    regularizers += tf.nn.l2_loss(U[layer][language])
        
            regularizers = 0
            regularizers += tf.nn.l2_loss(W)
            loss = tf.reduce_mean(losses) + 0.001 * regularizers
        
        with tf.name_scope('accuracy'):
            correct_predictions = tf.cast(tf.equal(predictions, tf.argmax(input_y, 1)), 'float')
            accuracy = tf.reduce_sum(correct_predictions, name='accuracy')
        
        optimizer = tf.train.AdadeltaOptimizer(1.0, 0.95, 1e-6)
        grads_and_vars = optimizer.compute_gradients(loss)
        capped_grads_and_vars = [(tf.clip_by_norm(gv[0], clip_norm=3, axes=[0]), gv[1]) for gv in grads_and_vars]
        optimizer = optimizer.apply_gradients(capped_grads_and_vars)
    
        sess.run(tf.global_variables_initializer())
        
        for language in languages:
            sess.run(embeddings[language][0].assign(vectors[language]))
        
        acc = 0
        sco = 0
        for epoch in range(epochs):
            a = time.time()
            x_batches = []
            shuffle_indices = np.random.permutation(np.arange(len(train_y)))
            y_batches = batch_iter(train_y, batch_size, 1, shuffle_indices=shuffle_indices, shuffle=False)
            for i in range(len(languages)):
                x_batches.append(batch_iter(train_xs[i], batch_size, 1, shuffle_indices=shuffle_indices, shuffle=False))
            
            ls = []
            for j in range(len(y_batches)):
                y_batch = np.array(y_batches[j])
                if len(y_batch) == 0:
                    continue
                
                feed_dict = {
                    input_y: y_batch,
                    dropout_keep_prob: dropout_rate,
                    is_training: True
                }
                for i in range(len(languages)):
                    feed_dict[inputs[i]] = np.array(x_batches[i][j])
                
                l, _ = sess.run([loss, optimizer], feed_dict)
                ls.append(l)
            
            correct = 0
            prediction = []
            temp_hidden = {}
            temp_context_attended = {}
            temp_context = []
            temp_self_attended = {}
            temp_self = []
            temp_scores = []
            temp_certainties = {}
            for language in languages:
                temp_hidden[language] = []
                temp_context_attended[language] = []
                temp_self_attended[language] = []
                temp_certainties[language] = []
            
            for j in range(0, len(test_y), batch_size):
                feed_dict = {
                    input_y: test_y[j:j+batch_size],
                    dropout_keep_prob: 1,
                    is_training: False
                }
                for i in range(len(languages)):
                    feed_dict[inputs[i]] = test_xs[i][j:j+batch_size]

                hi, ca, co, sa, c, p, ss, ce = sess.run([h_drops, context_attended, contexts, self_attended, accuracy, correct_predictions, soft_scores, certainties], feed_dict)
                for language in languages:
                    temp_hidden[language].extend(hi[language])
                    temp_context_attended[language].extend(ca[language])
                    temp_certainties[language].extend(ce[language])
                temp_context.extend(co)
                temp_scores.extend(ss)
                correct += c
                prediction.extend(p)

            if acc <= correct/test_y.shape[0]:
                acc = correct/test_y.shape[0]
                sco = temp_scores
                origs = []
            
            print("epoch:", epoch, "loss:", np.mean(ls), "accuracy:", correct/test_y.shape[0], acc, 'time:', time.time()-a)

        print("accuracy:", acc)
        
        accuracies.append(acc)
        tf.reset_default_graph()

f = open('attention.txt', 'a')
accuracies = np.array(accuracies)
f.write(type + '\t' + ','.join(languages) + '\t' + str(accuracies.mean()) + '\n')
f.close()
