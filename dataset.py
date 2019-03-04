import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.data import Dataset

class SentDataset:

    def __init__(self, dataset, batch_size = 32, epoch = 100):
        """
        Parameters:
            dataset: 'sst1' or 'sst2' or 'imdb'

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = epoch
        self._build_dataset()
        self._build_vocab()
        print('Sucessfully Loading Dataset')

    def __len__(self):
        return self.length

    def _build_dataset(self):
        self._get_data_path()
        self._load_dataset()

    def _load_dataset(self):
        print('Loading Dataset')
        with open(self.data_path, 'rb') as f:
            (self.train_x, self.train_y), (self.test_x, self.test_y) = pickle.load(f)
        self.length = len(self.train_x)


    def _get_data_path(self):
        if self.dataset is 'sst1':
            self.data_path = './data/sst1.data'
            self.vocab_path = './data/sst1.voc'

        elif self.dataset is 'sst2':
            self.data_path = './data/sst2.data'
            self.vocab_path = './data/sst2.voc'

        elif self.dataset == 'imdb':
            self.data_path = './data/imdb/imdb.data'
            self.vocab_path = './data/imdb/imdb.voc'

    def _build_vocab(self):
        self.i2t = ['PAD', 'OOV']

        print('Building vodab')
        with open(self.vocab_path, 'rb') as f:
            vocab = pickle.load(f)

        for each in sorted(vocab.items(), key = lambda x: x[1]) :
            self.i2t.append(each[0])

        self.t2i = {voc: idx for idx, voc in enumerate(self.i2t)}
        self.vocab_size = len(self.i2t)



    def data_generator(self, x, y):

        current_index = 0
        length = len(x)

        while True:
            if current_index > length:
                return
            else:
                batch_x = x[current_index:current_index+self.batch_size]
                batch_y = np.expand_dims(y[current_index:current_index+self.batch_size], -1)
                max_size = len(batch_x[-1])

                padded_x = pad_sequences(batch_x, maxlen = max_size, padding = 'post')
                current_index += self.batch_size
                yield padded_x, batch_y


    def mapping_function(self, x, y):
        features = {'inputs': x}
        labels = y

        return features, labels

    def train_input_function(self):
        dataset = Dataset.from_generator(generator = lambda: self.data_generator(self.train_x, self.train_y),
                                        output_types = (tf.int32, tf.int32),
                                        output_shapes = ((None, None), (None, 1)))
        dataset = dataset.map(self.mapping_function).shuffle(self.length).repeat(self.epoch)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()

    def test_input_function(self):
        dataset = Dataset.from_generator(generator = lambda: self.data_generator(self.test_x, self.test_y),
                                        output_types = (tf.int32, tf.int32),
                                        output_shapes = ((None, None), (None, 1)))
        dataset = dataset.map(self.mapping_function)
        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next()
