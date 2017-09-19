import sys
import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

NAMESPACE = 'gcn'


def GCN_layer_fw(embedding_size, hidden_layer1_size, hidden, Atilde_fw):
    W0_fw = tf.Variable(tf.random_uniform([embedding_size, hidden_layer1_size], 0, 0.1), name='W0_fw')
    b0_fw = tf.Variable(tf.random_uniform([hidden_layer1_size], -0.1, 0.1), name='b0_fw')
    left_X1_projection_fw = lambda x: tf.matmul(x, W0_fw) + b0_fw
    left_X1_fw = tf.map_fn(left_X1_projection_fw, hidden)
    left_X1_fw = tf.transpose(left_X1_fw, perm=[1, 0, 2], name='left_X1_fw')
    X1_fw = tf.nn.relu(tf.matmul(Atilde_fw, left_X1_fw))
    X1_fw = tf.transpose(X1_fw, perm=[1, 0, 2])
    return X1_fw


def GCN_layer_bw(embedding_size, hidden_layer1_size, hidden, Atilde_bw):
    W0_bw = tf.Variable(tf.random_uniform([embedding_size, hidden_layer1_size], 0, 0.1), name='W0_bw')
    b0_bw = tf.Variable(tf.random_uniform([hidden_layer1_size], -0.1, 0.1), name='b0_bw')
    left_X1_projection_bw = lambda x: tf.matmul(x, W0_bw) + b0_bw
    left_X1_bw = tf.map_fn(left_X1_projection_bw, hidden)
    left_X1_bw = tf.transpose(left_X1_bw, perm=[1, 0, 2], name='left_X1_bw')
    X1_bw = tf.nn.relu(tf.matmul(Atilde_bw, left_X1_bw))
    X1_bw = tf.transpose(X1_bw, perm=[1, 0, 2])
    return X1_bw


class GCNNerModel(object):
    _stack_dimension = 2
    _embedding_size = 160
    _internal_proj_size = 40
    _memory_dim = 160
    _vocab_size = 300
    _hidden_layer1_size = 160
    _hidden_layer2_size = 160
    _output_size = 19

    def __init__(self):

        with tf.variable_scope(NAMESPACE):
            config = tf.ConfigProto(allow_soft_placement=True)
            self.sess = tf.Session(config=config)

            # Input variables
            self.enc_inp = tf.placeholder(tf.float32, shape=(None, None, self._vocab_size))
            self.enc_inp_bw = tf.placeholder(tf.float32, shape=(None, None, self._vocab_size))

            # Dense layer before LSTM
            self.Wi = tf.Variable(tf.random_uniform([self._vocab_size, self._internal_proj_size], 0, 0.1))
            self.bi = tf.Variable(tf.random_uniform([self._internal_proj_size], -0.1, 0.1))
            self.internal_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Wi) + self.bi)
            self.enc_int = tf.map_fn(self.internal_projection, self.enc_inp)
            self.enc_int_bw = tf.map_fn(self.internal_projection, self.enc_inp_bw)

            # Bi-LSTM part
            self.enc_cell_fw = rnn.MultiRNNCell([rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                                                state_is_tuple=True)
            self.enc_cell_bw = rnn.MultiRNNCell([rnn.GRUCell(self._memory_dim) for _ in range(self._stack_dimension)],
                                                state_is_tuple=True)
            with tf.variable_scope('fw'):
                self.encoder_fw, _ = tf.nn.dynamic_rnn(self.enc_cell_fw, self.enc_int, time_major=True,
                                                       dtype=tf.float32)
            with tf.variable_scope('bw'):
                self.encoder_bw, _ = tf.nn.dynamic_rnn(self.enc_cell_bw, self.enc_int_bw, time_major=True,
                                                       dtype=tf.float32)
            self.encoder_outputs = tf.concat(values=[self.encoder_fw, self.encoder_bw], axis=2)

            # Dense layer before GCN
            self.Ws = tf.Variable(tf.random_uniform([self._memory_dim * 2, self._memory_dim], 0, 0.1))
            self.bs = tf.Variable(tf.random_uniform([self._memory_dim], -0.1, 0.1))
            self.first_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Ws) + self.bs)
            self.hidden = tf.map_fn(self.first_projection, self.encoder_outputs)

            # GCN part
            self.Atilde_fw = tf.placeholder(tf.float32, shape=(None, None, None), name="Atilde_fw")
            self.Atilde_bw = tf.placeholder(tf.float32, shape=(None, None, None), name="Atilde_bw")
            self.X1_fw = GCN_layer_fw(self._embedding_size,
                                      self._hidden_layer1_size,
                                      self.hidden,
                                      self.Atilde_fw)
            self.X1_bw = GCN_layer_bw(self._embedding_size,
                                      self._hidden_layer1_size,
                                      self.hidden,
                                      self.Atilde_bw)
            self.X3 = tf.concat(values=[self.X1_fw, self.X1_bw], axis=2)

            # Final feedforward layers
            self.Ws = tf.Variable(tf.random_uniform([self._hidden_layer2_size * 2, self._hidden_layer2_size], 0, 0.1),
                                  name='Ws')
            self.bs = tf.Variable(tf.random_uniform([self._hidden_layer2_size], -0.1, 0.1), name='bs')
            self.first_projection = lambda x: tf.nn.relu(tf.matmul(x, self.Ws) + self.bs)
            self.last_hidden = tf.map_fn(self.first_projection, self.X3)

            self.Wf = tf.Variable(tf.random_uniform([self._hidden_layer2_size, self._output_size], 0, 0.1), name='Wf')
            self.bf = tf.Variable(tf.random_uniform([self._output_size], -0.1, 0.1), name='bf')
            self.final_projection = lambda x: tf.matmul(x, self.Wf) + self.bf
            self.outputs = tf.map_fn(self.final_projection, self.last_hidden)

            # Loss function and training
            self.y_ = tf.placeholder(tf.float32, shape=(None, None, self._output_size), name='y_')
            self.gold_tags = tf.placeholder(tf.int32, shape=(None, None), name='gold_tags')
            self.sequence_lengths = tf.placeholder(tf.float32, shape=(None), name='sequence_lenghts')
            self.transition_params = tf.placeholder(tf.float32, shape=(self._output_size, self._output_size),
                                                    name='transition_params')
            self.trans_outputs = tf.transpose(self.outputs, perm=[1, 0, 2], name='trans_outputs')
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(self.trans_outputs, self.gold_tags,
                                                                  self.sequence_lengths, self.transition_params)
            self.cross_entropy = tf.reduce_mean(-log_likelihood)

            # Clipping the gradient
            optimizer = tf.train.AdamOptimizer(1e-4)
            gvs = optimizer.compute_gradients(self.cross_entropy)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs if var.name.find(NAMESPACE) != -1]
            self.train_step = optimizer.apply_gradients(capped_gvs)
            self.sess.run(tf.global_variables_initializer())

            # Adding the summaries
            tf.summary.scalar('cross_entropy', self.cross_entropy)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter('./train', self.sess.graph)

    def _add_identity(self, A):
        num_nodes = A.shape[0]
        identity = np.identity(num_nodes)
        return identity + A

    def __train(self, A_fw, A_bw, X, y, trans_prob):
        Atilde_fw = np.array([self._add_identity(item) for item in A_fw])
        Atilde_bw = np.array([self._add_identity(item) for item in A_bw])

        X = np.array(X)
        X2 = np.copy(X)

        seq_lengths = np.array([item.shape[0] for item in X])
        gold_tags = np.argmax(y, axis=2)

        X = np.transpose(X, (1, 0, 2))
        X2 = np.transpose(X2, (1, 0, 2))
        X2 = X2[::-1, :, :]

        y = np.array(y)
        y = np.transpose(y, (1, 0, 2))

        feed_dict = {self.enc_inp: X}
        feed_dict.update({self.enc_inp_bw: X2})

        feed_dict.update({self.Atilde_fw: Atilde_fw})
        feed_dict.update({self.Atilde_bw: Atilde_bw})
        feed_dict.update({self.y_: y})

        feed_dict.update({self.transition_params: trans_prob})
        feed_dict.update({self.sequence_lengths: seq_lengths})
        feed_dict.update({self.gold_tags: gold_tags})

        loss, _, summary = self.sess.run([self.cross_entropy, self.train_step, self.merged], feed_dict)
        return loss, summary

    def train(self, data, trans_prob, epochs=20):
        for epoch in range(epochs):
            loss, _ = self.__train([data[i][0] for i in range(len(data))],
                                   [data[i][1] for i in range(len(data))],
                                   [data[i][2] for i in range(len(data))],
                                   [data[i][3] for i in range(len(data))],
                                   trans_prob)
            sys.stdout.flush()

    def __predict(self, A_fw, A_bw, X, trans_prob):
        Atilde_fw = np.array([self._add_identity(item) for item in A_fw])
        Atilde_bw = np.array([self._add_identity(item) for item in A_bw])

        X = np.array(X)
        X2 = np.copy(X)

        seq_lengths = np.array([item.shape[0] for item in X])

        X = np.transpose(X, (1, 0, 2))
        X2 = np.transpose(X2, (1, 0, 2))
        X2 = X2[::-1, :, :]

        feed_dict = {self.enc_inp: X}
        feed_dict.update({self.enc_inp_bw: X2})
        feed_dict.update({self.Atilde_fw: Atilde_fw})
        feed_dict.update({self.Atilde_bw: Atilde_bw})

        feed_dict.update({self.transition_params: trans_prob})
        feed_dict.update({self.sequence_lengths: seq_lengths})

        y_batch = self.sess.run(self.outputs, feed_dict)
        return y_batch

    def predict_with_viterbi(self, A_fw, A_bw, X, trans_params):
        outputs = np.array(self.__predict([A_fw], [A_bw], [X], trans_params))
        outputs = np.transpose(outputs, [1, 0, 2])
        outputs = outputs[0]
        viterbi_sequence, __score = tf.contrib.crf.viterbi_decode(outputs, trans_params)
        prediction = []
        for item in viterbi_sequence:
            vector = [0.] * self._output_size
            vector[item] = 1.
            prediction.append(vector)
        return prediction

    # Loading and saving functions

    def save(self, filename):
        saver = tf.train.Saver()
        saver.save(self.sess, filename)

    def load_tensorflow(self, filename):
        saver = tf.train.Saver([v for v in tf.global_variables() if NAMESPACE in v.name])
        saver.restore(self.sess, filename)

    @classmethod
    def load(self, filename):
        model = GCNNerModel()
        model.load_tensorflow(filename)
        return model
