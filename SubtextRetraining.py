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


# Using a simple class definition for an ENUM of possible subtexts
# Enum is not available in Python 2.7
class Subtext:
    No_Subtext, Sexual, Violent, Depressive = range(4)

    def to_string(self):
        if self == 0:
            return "no_subtext"
        elif self == 1:
            return "sexual"
        elif self == 2:
            return "violent"
        elif self == 3:
            return "depressive"

data_index = 0


# Create a function to generate a training batch of data using skim-gram model
def generate_batch(batch_size, num_skips, skip_window, data):
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


def retrain_embeddings(vocab_size, num_steps, num_skips, skip_window, batch_size, data, embed_size=300, num_sampled=32):
    rt_graph = tf.Graph()
    with rt_graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Aquire the word embeddings that are to be retrained
        # Model is limited to 500,000 words at present due to (my comp's) memory constraints
        # This should be properly run with no limit
        pretrainedfile = './GoogleNews-vectors-negative300.bin'
        print("Loading pre-trained embeddings...")
        model = gensim.models.KeyedVectors.load_word2vec_format(
            pretrainedfile, binary=True, limit=500000)
        new_embed = tf.Variable(initial_value=model.syn0)
        del model  # (TODO: Make sure this doesn't erase the values in new_embed)

        print("Building the graph...")
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
        #  return rt_graph

# def retrain_embeddings(rt_graph, num_skips, skip_window, batch_size):
#     # TODO: create this from SubtextAnalyzer's graph training
    with tf.Session(graph=rt_graph) as session:
        init.run()
        print("Vars Initialized!")
        avg_loss = 0

        for step in xrange(num_steps):
            batch_input, batch_labels = generate_batch(
                batch_size, num_skips, skip_window, data)
            feed_dict = {train_inputs: batch_input, train_labels: batch_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            avg_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    avg_loss /= 2000
                # Average Loss is the average of the loss at each of the last 2000 steps
                print("Average Loss at step ", step, " is: ", avg_loss)
        # BE SURE TO INCLUDE DELETE STATEMENTS FOR *all* vars WHOSE JOB IS DONE
        # del nce_weights
        # del nce_biases
        # global data_index = 0

        return new_embed.eval()


def main(subtext, vocab_size, num_steps, batch_size, num_skips, skip_window):
    datafile = './ReadingSamples_Converted/' + Subtext.to_string(subtext) + str(vocab_size) + '.txt'
    data = open(datafile, mode='r')
    rt_graph = retrain_embeddings(vocab_size, num_steps, num_skips, skip_window, batch_size, data)
