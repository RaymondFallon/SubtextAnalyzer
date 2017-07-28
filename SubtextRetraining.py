from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import math
import pickle
from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
import gensim

data_index = 0
subtexts = ['no_subtext', 'violent', 'depressive', 'sexual']

# Function to generate a training batch of data using skim-gram model
def generate_batch(batch_size, num_skips, skip_window, data):
    """

    :param batch_size: number of individual pieces of training data per batch
    :param num_skips: How many times to use an input to generate a label
    :param skip_window: How many words to the left/right to consider
    :param data: writing sample that has been converted to indexed ints
    :return: [batch_input, batch_labels], a batch of training data to
             be used to retrain our embeddings
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=batch_size, dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buff = collections.deque(maxlen=span)
    for _ in range(span):
        buff.append(data[data_index])
        data_index = (data_index + 1) & len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buff[skip_window]
            labels[i * num_skips + j, 0] = buff[target]
        buff.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    #  Backtrack a bit to avoid skipping the latter words (when next batch is created)
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

def retrain_embeddings(vocab_size, num_steps, num_skips, skip_window, batch_size, data, embed_size=300, num_sampled=32):
    global data_index
    rt_graph = tf.Graph()
    with rt_graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        # Aquire the word embeddings that are to be retrained
        pretrainedfile = './GoogleNews-vectors-negative300.bin'
        print("Loading pre-trained embeddings...")
        model = gensim.models.KeyedVectors.load_word2vec_format(
            pretrainedfile, binary=True, limit=vocab_size)
        new_embed = tf.Variable(initial_value=model.syn0)
        vocab, index2word = model.vocab, model.index2word

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

        # Construct the SGD optimizer using a learning rate of 0.1.
        optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        # Add variable initializer.
        init = tf.global_variables_initializer()

    with tf.Session(graph=rt_graph) as session:
        init.run()
        print("Variables Initialized!")
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

        data_index = 0

        return new_embed.eval(), vocab, index2word


def main(vocab_size=50000, num_steps=100001,
         batch_size=128, num_skips=2, skip_window=2):
    for subtext in subtexts:
        datafile_name = './ReadingSamples_Converted/' + subtext + str(vocab_size) + '.txt'
        datafile = open(datafile_name, mode='r')
        data = pickle.load(datafile)
        new_model = gensim.models.KeyedVectors()
        new_model.syn0, new_model.vocab, new_model.index2word = retrain_embeddings(
            vocab_size, num_steps, num_skips, skip_window, batch_size, data)
        print("Retraining complete! Saving new embeddings...")
        savefile = './New_Embeddings/' + subtext + str(vocab_size)
        gensim.models.KeyedVectors.save_word2vec_format(new_model, savefile)

if __name__ == '__main__':
    main()