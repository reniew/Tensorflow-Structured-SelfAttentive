import tensorflow as tf
import tensorflow.keras.layers as L
from graph import Graph

class Model:

    def __init__(self):
        pass

    def model_function(self, mode, features, labels, params):

        TRAIN = mode == tf.estimator.ModeKeys.TRAIN
        EVAL = mode == tf.estimator.ModeKeys.EVAL
        PREDICT = mode == tf.estimator.ModeKeys.PREDICT

        inputs = features['inputs']
        self.targets = tf.to_float(labels)
        self.loss, self.train_op, self.metrics, self.prediction, self.global_step = None, None, None, None, None
        self._build_grpah(inputs, params['vocab_size'], PREDICT)

        accuracy_for_train = self._evaluate()

        logging_summary_dictionary = {'accuracy': accuracy_for_train,
                                    'progress': self.global_step // (25000//32),
                                    'prediction_loss': self.prediction_loss,
                                    'penalty': self.p}

        logging_hook = tf.train.LoggingTensorHook(logging_summary_dictionary, every_n_iter=100)
        self._write_summary(logging_summary_dictionary)


        return tf.estimator.EstimatorSpec(mode = mode,
                                        loss = self.loss,
                                        eval_metric_ops = self.metrics,
                                        train_op = self.train_op,
                                        training_hooks = [logging_hook],
                                        predictions = self.prediction)

    def _build_grpah(self, inputs, vocab_size, is_predict):
        graph = Graph(vocab_size = vocab_size)
        logits, self.p, attention_outputs = graph.build_graph(inputs)
        self.prediction = {'prediction': tf.round(logits), 'attention_output': attention_outputs}

        if not is_predict:
            self._build_loss(logits)
            self._build_train_op()
            self._build_metrics()

    def _write_summary(self, dictionary):
        for key, value in dictionary.items():
             tf.summary.scalar(key, value)

    def _evaluate(self):
        correct_pred = tf.equal(self.prediction['prediction'], self.targets)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        return accuracy

    def _build_loss(self, logits):
        self.prediction_loss = tf.losses.sigmoid_cross_entropy(self.targets, logits)
        self.loss = self.prediction_loss + self.p

    def _build_train_op(self):
        self.global_step = tf.train.get_global_step()
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step = self.global_step)

    def _build_metrics(self):
        self.metrics = {'accuracy': tf.metrics.accuracy(labels = self.targets, predictions = self.prediction['prediction'] ) }
