from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import threading
import time
import collections
import random
import math

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
import gensim

batch_size = 128
num_skips = 4
skip_window = 4

filename = './GoogleNews-vectors-negative300.bin'
# Model is limited to 500,000 words at present due to memory constraints
# This should be properly run with no limit
model = gensim.models.KeyedVectors.load_word2vec_format(
    filename, binary=True, limit=50000)

print("syn0 shape = ", model.syn0.shape)
print("syn0 dtype = ", model.syn0.dtype)
print("syn0norm = ", model.syn0norm)
print("vocab length = ", len(model.vocab))
print("index2word length = ", len(model.index2word))
print("vector_size = ", model.vector_size)

print(model.doesnt_match("man woman boy tree girl".split()))
print(model.doesnt_match("up down left right north".split()))
print(model.doesnt_match("cow pig lamb zebra".split()))
print(model.doesnt_match("poultry pork cow beef".split()))
print(model.most_similar_cosmul(positive="shaved trimmed buzzed".split(),
                                negative="skinned".split(),  # filled not in top 100000
                                topn=5))
print(model.most_similar_cosmul(positive="dragon worm".split(),
                                negative="butterfly".split(),
                                topn=5))
print(model.most_similar_cosmul(positive="rock stone brick".split(),
                                negative="clay sand".split(),
                                topn=5))
print(model.most_similar_cosmul(positive="prince king girl woman".split(),
                                negative="man boy".split(),
                                topn=5))
print(model.index2word[0])
print(model.index2word[1])
print(model.index2word[2])
print(model.index2word[3])
print(model.index2word[114])
print(model.index2word[42164])

# Using a simple class definition for an ENUM of possible subtexts
# Enum is not available in Python 2.7
class Subtext:
    No_Subtext, Sexual, Violent, Depressive = range(4)

data = []
data_index = 0

# Create a function to generate a training batch of data using skim-gram model
def generate_batch(batch_size, num_skips, skip_window):
    """

    :param batch_size: number of individual pieces of training data per batch
    :param num_skips: How many times to use an input to generate a label
    :param skip_window: How many words to the left/right to consider
    :return: [batch_input, batch_labels], a batch of training data to
             be used to retrain our embeddings
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) & len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    #  Backtrack a bit to avoid skipping the latter words (when next batch is created)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels






# Remove extraneous paramaters if this function stays inside this .py file!!
def retrain_embedding(subtext, num_sampled, vocab_size, embed_size, num_steps):
    """
    Called to retrain the original Google word embeddings based on new data of type 'subtext'
    Uses skip-gram data and NCE-loss to create new embeddings

    :param subtext: subtext from class Subtext (e.g. Subtext.Depressive)
    :param num_sampled: number of negative samples per example
    :param vocab_size: number of words in vocab
    :param embed_size: size of each word's embedding
    :return:
    """
    rt_graph = tf.Graph()

    with rt_graph.as_default():

        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Define the word embeddings that are to be retrained
        new_embed = tf.Variable(initial_value=model)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocab_size, embed_size],
                                stddev=1.0 / math.sqrt(embed_size)))
        nce_biases = tf.Variable(tf.zeros([vocab_size]))

        partial_embed = tf.nn.embedding_lookup(new_embed, train_inputs)

        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=train_labels,
                inputs=partial_embed,
                num_sampled=num_sampled,
                num_classes=vocab_size
            )
        )

        # Construct the SGD optimizer using a learning rate of 1.0.
        # Should this optimzer be threaded ala word2vec's more advanced self._train ?
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    with tf.Session(graph=rt_graph) as session:
        init.run()
        print("Vars Initialized!")
        avg_loss = 0

        for step in xrange(num_steps):
            batch_input, batch_labels = generate_batch(
                batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_input, train_labels: batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            avg_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    avg_loss /= 2000
                # Average Loss is the average of the loss at each of the last 2000 steps
                print("Average Loss at step " , step, " is: " , avg_loss)
        # BE SURE TO INCLUDE DELETE STATEMENTS FOR *all* vars WHOSE JOB IS DONE
        del nce_weights
        del nce_biases
        # global data_index = 0

        return new_embed.eval()




