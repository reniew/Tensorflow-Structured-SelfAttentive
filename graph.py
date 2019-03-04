import tensorflow as tf
import tensorflow.keras.layers as L


class Graph:

    def __init__(self,
                vocab_size,
                dropout_rate = 0.5,
                embedding_dimension = 100,
                model_dimension = 300,
                attention_dimension = 350,
                num_of_multihops = 30,
                fully_connected_dimension = 3000,
                reuse = tf.AUTO_REUSE,
                dtype = tf.float32):
        """
        Arguments:
            vocab_size: Int. Size of vocabulary contatining special tokens.
            dropout_rate: Float between 0 and 1. The fraction of applying dropout to inputs.
            embedding_dimension: Int. d in paper, dimension of word vectorself.
            model_dimension: Int. u in paper, dimension of LSTM cell.
            attention_dimension: Int. a in paper, dimension of attention vector.
            num_of_multihops: Int. r in paper, number of multi hop attention.

        """

        self.embedding_layer = L.Embedding(input_dim = vocab_size, output_dim = embedding_dimension)
        self.dropout = L.Dropout(dropout_rate)
        self.biLSTM = L.Bidirectional(L.LSTM(units = model_dimension, return_sequences = True))
        self.w_s1 = tf.get_variable('w_s1', shape = (attention_dimension, 2*model_dimension))
        self.w_s2 = tf.get_variable('w_s2', shape = (num_of_multihops, attention_dimension))
        self.fc1 = L.Dense(fully_connected_dimension, activation = tf.nn.relu)
        self.fc2 = L.Dense(fully_connected_dimension, activation = tf.nn.relu)
        self.fc3 = L.Dense(1, activation = tf.nn.sigmoid)
        self.flatten = L.Flatten()
        self.num_of_multihops = num_of_multihops
        self.reuse = reuse
        self.dtype = dtype

    def build_graph(self, inputs):
        with tf.variable_scope('self-attention', reuse = self.reuse, dtype = self.dtype):
            s = self._build_embedding(inputs) # (bs, n, d)
            h = self.biLSTM(s) # (bs, n, 2u)
            h = self.dropout(h)
            a = self._get_self_attention(h) # (bs, r, n)
            m = tf.matmul(a = a, b = h) # (bs, r, 2u)
            p = self._get_penalization_term(a) # ()

        # classification networks
        with tf.variable_scope('classification_network', reuse = self.reuse, dtype = self.dtype):
            classify_inputs = self.flatten(m)
            logits = self._classify_networks(classify_inputs) # (bs, 1)

            return logits, p, a




    def _build_embedding(self, inputs):
        return self.dropout(self.embedding_layer(inputs))

    def _get_self_attention(self, inputs):
        score = tf.tanh(tf.transpose(tf.tensordot(self.w_s1,
                                    tf.transpose(inputs, perm = [0,2,1]),
                                    axes=[[1],[1]]),[1,0,2]))
        attention_output = tf.nn.softmax(tf.transpose(tf.tensordot(self.w_s2,
                                                    score,
                                                    axes=[[1],[1]]),[1,0,2]))

        return self.dropout(attention_output)

    def _classify_networks(self, inputs):
        outputs = self.dropout(self.fc1(inputs))
        outputs = self.dropout(self.fc2(outputs))
        logits = self.fc3(outputs)

        return logits

    def _get_penalization_term(self, a):
         penaly = tf.matmul(a = a, b = a, transpose_b = True) - tf.eye(self.num_of_multihops)
         frobenius_norm = tf.norm(penaly, ord = 'fro', axis = [-2, -1])

         return tf.reduce_mean(frobenius_norm)
