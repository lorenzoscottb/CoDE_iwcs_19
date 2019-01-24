

from __future__ import division
from collections import Counter, defaultdict
import os
from random import shuffle
import tensorflow as tf
import numpy as np
from apt_toolkit.utils import vector_utils as vu


class NotTrainedError(Exception):
    pass

class NotFitToCorpusError(Exception):
    pass

class Model():
    def __init__(self, context_size=5, max_vocab_size=10000000, min_occurrences=1,
                cooccurrence_cap=100, batch_size=512, model_name=None):
        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")
        self.model_name = model_name
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        self.cooccurrence_cap = cooccurrence_cap
        self.batch_size = batch_size
        self.dict_size = 0
        self.__vocab_size = None
        self.__words = None
        self.__context_words = None
        self.__context_words_to_id = None
        self.__word_to_id = None
        self.__cooccurrence_matrix = None

    def reset_model(self):
        self.dict_size = None
        self.__words = None
        self.__context_words = None
        self.__context_words_to_id = None
        self.__word_to_id = None
        self.__cooccurrence_matrix = None

    def path_and_fit(self, vectors, prepare_batches=True, save_training_set=False):
        if self.dict_size == 0:
            self.dict_size = len(vectors.keys())
        self.__context_words = vectors.keys()
        self.vocab_path = apt_path_matrix(vectors, self.model_name, path_depth=3)
        # if self.vocab_size is None:
        self.__vocab_size = len(self.vocab_path)
        self.__fit_to_apt(self.vocab_path, vectors, self.max_vocab_size, self.min_occurrences,
                             self.left_context, self.right_context)
        self.__context_words_to_id = {word:number for number, word in
                                      enumerate(self.__context_words)}
        self.__dict_size = len(self.__context_words_to_id)
        if prepare_batches:
            print('Preparing shared-batches...')
            self.__i_indices, self.__j_indices, self.__counts = self.__shared_context_batches()

        if save_training_set:
            print('saving trainig set...')
            i_j_count_matrix = [self.__i_indices, self.__j_indices, self.__counts]
            save_object(i_j_count_matrix, self.model_name+'_training_set')

    def fit_to_apt(self, corpus, vectors, prepare_batches=True, save_training_set=False):
        self.__fit_to_apt(corpus, vectors, self.max_vocab_size, self.min_occurrences,
                             self.left_context, self.right_context)
        self.__context_words_to_id = {word:number for number, word in
                                      enumerate(self.__context_words)}
        self.__dict_size = len(self.__context_words_to_id)
        if prepare_batches:
            print('Preparing shared-batches...')
            self.__i_indices, self.__j_indices, self.__counts = self.__shared_context_batches()

        if save_training_set:
            print('saving trainig set...')
            i_j_count_matrix = [self.__i_indices, self.__j_indices, self.__counts]
            save_object(i_j_count_matrix, self.model_name+'_training_set')

    def __fit_to_apt(self, corpus, vectors, vocab_size, min_occurrences, left_size, right_size):
        cooccurrence_matrix = {}
        word_counts = Counter()
        cooccurrence_counts = defaultdict(float)
        print('Preparing corpus...')
        for region in corpus:
            word_counts.update(region)
            for l_context, word, r_context in _context_windows(region, left_size, right_size):
                for i, context_word in enumerate(l_context[::-1]):
                    # add (1 / distance from focal word) for this pair
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
                for i, context_word in enumerate(r_context):
                    cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
        if len(cooccurrence_counts) == 0:
            raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
        self.__words = [word for word, count in word_counts.most_common(vocab_size)
                        if count >= min_occurrences]
        print('Preparing word ID...')
        self.__word_to_id = {word: i for i, word in enumerate(self.__words)}
        # for words, count in cooccurrence_counts.items():
        #     print(words)
        #     break
        print('Building matrix..')
        # extract wich is path wich word for ppmi value extraction
        for count, co_set in enumerate(cooccurrence_counts.items()):
            # print(co_set)
            if co_set[0][1] in self.__word_to_id and co_set[0][0] in self.__word_to_id:
                if count % 2 == 0:
                    wrd = co_set[0][1]
                    pt = co_set[0][0]
                    cooccurrence_matrix[(self.__word_to_id[pt], self.__word_to_id[wrd])] = vectors[wrd][pt]
                else:
                    wrd = co_set[0][0]
                    pt = co_set[0][1]
                # print(count, wrd, pt)
                    cooccurrence_matrix[(self.__word_to_id[wrd], self.__word_to_id[pt])] = vectors[wrd][pt]
                # print(count, wrd, pt)
            self.__cooccurrence_matrix = cooccurrence_matrix

    def training_batch(self):

        batches = list(_batchify(self.batch_size, self.__i_indices, self.__j_indices, self.__counts))
        shuffle(batches)
        for batch_index, batch in enumerate(batches):
            i_s, j_s, counts = batch

        return i_s, j_s, counts

    def training_set(self):

        return self.__i_indices, self.__j_indices, self.__counts

    def __prepare_batches(self):
        if self.__cooccurrence_matrix is None:
            raise NotFitToCorpusError(
                "Need to fit model to corpus before preparing training batches.")
        cooccurrences = [(word_ids[0], word_ids[1], count)
                         for word_ids, count in self.__cooccurrence_matrix.items()]
        i_indices, j_indices, counts = zip(*cooccurrences)
        return list(_batchify(self.batch_size, i_indices, j_indices, counts))

    def __shared_context_batches(self):
        print('collecting batches...')
        # TOCHECK: j_indices must refer to diffrent matrix!!!
        print('start the collection...')
        shared_cooccurrences = [(word_ids[0],
                                 self.__context_words_to_id[_key_by_value(self.__context_words_to_id, word_ids[1])],
                                  count)
                               for word_ids, count in self.__cooccurrence_matrix.items()
                               if word_ids[1] < self.__dict_size
                               and _key_by_value(self.__context_words_to_id, word_ids[1]) in self.__context_words]
        i_indices, j_indices, counts = zip(*shared_cooccurrences)
        return i_indices, j_indices, counts

    @property
    def vocab_size(self):
        return len(self.__words)

    @property
    def words(self):
        if self.__words is None:
            raise NotFitToCorpusError("Need to fit model to corpus before accessing words.")
        return self.__words

    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotTrainedError("Need to train model before accessing embeddings")
        return self.__embeddings

    def id_for_word(self, word):
        if self.__word_to_id is None:
            raise NotFitToCorpusError("Need to fit model to corpus before looking up word ids.")
        return self.__word_to_id[word]

    def asimmetric_glove(self, dimension):

        from keras.layers import Input, Embedding, Dot, Reshape, Add
        from keras.models import Model
        from keras.optimizers import Adam
        import os
        import numpy as np

        self.OUTPUT_FOLDER = 'output/'
        self.DATA_FOLDER = 'data/'

        self.CENTRAL_EMBEDDINGS = 'central_embeddings'
        self.CONTEXT_EMBEDDINGS = 'context_embeddings'
        self.CENTRAL_BIASES = 'central_biases'
        self.CONTEXT_BIASES = 'context_biases'

        self.AGGREGATED_EMBEDDINGS = 'agg_embeddings'
        self.CORRELATION_MATRIX = 'corr_matrix'

        self.INDEX2WORD = 'index-word.pkl'
        self.WORD2INDEX = 'word-index.pkl'


        self.model = None

        self.vector_dim = dimension
        self.focal_size = self.vocab_size
        self.context_sixe = self.dict_size

        """
        A Keras implementation of the GloVe architecture
        :param focal_size: The number of distinct words
        :param vector_dim: The vector dimension of each word
        :return:
        """
        input_focal = Input((1,), name='central_word_id')
        input_context = Input((1,), name='context_word_id')

        central_embedding = Embedding(self.focal_size,
                                      self.vector_dim,
                                      input_length=1,
                                      name=self.CENTRAL_EMBEDDINGS)
        central_bias = Embedding(self.focal_size,
                                 1,
                                 input_length=1,
                                 name=self.CENTRAL_BIASES)

        context_embedding = Embedding(self.dict_size,
                                      self.vector_dim,
                                      input_length=1,
                                      name=self.CONTEXT_EMBEDDINGS)
        context_bias = Embedding(self.dict_size,
                                 1,
                                 input_length=1,
                                 name=self.CONTEXT_BIASES)

        vector_focal = central_embedding(input_focal)
        vector_context = context_embedding(input_context)

        bias_focal = central_bias(input_focal)
        bias_context = context_bias(input_context)

        dot_product = Dot(axes=-1)([vector_focal, vector_context])
        dot_product = Reshape((1,))(dot_product)
        bias_focal = Reshape((1,))(bias_focal)
        bias_context = Reshape((1,))(bias_context)

        prediction = Add()([dot_product, bias_focal, bias_context])

        model = Model(inputs=[input_focal, input_context], outputs=prediction)
        model.compile(loss=custom_loss, optimizer=Adam())

        self.model = model

    def train(self, train_sample, output_sample, epochs, batch_size):

        self.model.fit(train_sample, output_sample, epochs=epochs, batch_size=batch_size)

    def get_weights(self, layer_name='context_embeddings'):

        weights = None
        for layer in self.model.layers:
            if layer.name == layer_name:
                weights = layer.get_weights()

        return weights

    def set_weight(self, weight):

        for layer in self.model.layers:
            if layer.name == 'context_embeddings':
                layer.set_weights([weight])

    def np_tensor(self, x, y):

        import numpy as np

        return np.random.uniform(low=-1, high=1, size=(x, y))

    def save_object(self, object, name, dir=os.getcwd()):

        import pickle

        with open(dir + name + '.pkl', 'wb') as f:
            pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

    def glove_reverse(self, b_c, b_f, v_c, v_f):
        return np.power(2, np.inner(v_c, v_f) + b_c + b_f)

    def model_evaluation(self, context_embeddings=None):

        from sklearn.metrics.pairwise import euclidean_distances

        b_cntx = None
        b_cntr = None
        e_cntx = None
        e_cntr = None

        for layer in self.model.layers:
            if layer.name == self.CENTRAL_BIASES:
                b_cntr = layer.get_weights()[0]

            if layer.name == self.CENTRAL_EMBEDDINGS:
                e_cntr = layer.get_weights()[0]

            if layer.name == self.CONTEXT_BIASES:
                b_cntx = layer.get_weights()[0]

            if context_embeddings in None:
                if layer.name == self.CONTEXT_EMBEDDINGS:
                    e_cntx = layer.get_weights()[0]
            else:
                e_cntx = context_embeddings

        original_matrix = np.zeros((e_cntr.shape[0], e_cntr.shape[0]))
        new_matrix = np.zeros((e_cntr.shape[0], e_cntr.shape[0]))

        i_s, j_s, ppmis = fit_model.training_set()

        for index, ppmi in enumerate(ppmis):
            y = i_s[index]
            x = j_s[index]

            # print(x,y,ppmi)

            original_matrix[y][x] = ppmi
            new_matrix[y][x] = self.glove_reverse(b_cntx[x],
                                             b_cntr[y],
                                             e_cntx[x],
                                             e_cntr[y])

        return euclidean_distances(original_matrix, new_matrix)


def _context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = _window(region, start_index, i - 1)
        right_context = _window(region, i + 1, end_index)
        yield (left_context, word, right_context)


def _window(region, start_index, end_index):
    """
    Returns the list of words starting from `start_index`, going to `end_index`
    taken from region. If `start_index` is a negative number, or if `end_index`
    is greater than the index of the last word in region, this function will pad
    its return value with `NULL_WORD`.
    """
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens


def apt_path_matrix(vectors, path_end='amod', path_depth=3):

    print('preparing co-occ matrix for %s path' %path_end)
    vocab_path = []
    for cnt, word in enumerate(vectors.keys()):
        paths = list(vectors[word].keys())
        for cnt2, path in enumerate(paths):
            if type(path) is str:    # for some reasons weird stuff can appear instead of paths..
                clean_path, end_path = (path.replace('»', ' ')).split(':', 1)
            if len(clean_path.split()) <= path_depth and clean_path.endswith(path_end):
                vocab_path.append([path, word])
            # ppmi_value = vectors[word][path]

    vocab_size = len(vocab_path)

    print('done,the', path_end, 'vocabulary len is', vocab_size)

    return vocab_path


def _batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)


def _plot_with_labels(low_dim_embs, labels, path, size):
    import matplotlib.pyplot as plt
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    figure = plt.figure(figsize=size)  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right',
                     va='bottom')
    if path is not None:
        figure.savefig(path)
        plt.close(figure)


def _key_by_value(mydict, value):
    return list(mydict.keys())[list(mydict.values()).index(value)]


def save_object(object, name, dir=os.getcwd()):

    import pickle

    with open(dir + name + '.pkl', 'wb') as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def custom_loss(y_true, y_pred, X_MAX = 100, a = 3.0 / 4.0):

    import keras.backend as K

    """
    This is GloVe's loss function
    :param y_true: The actual values, in our case the 'observed' X_ij co-occurrence values
    :param y_pred: The predicted (log-)co-occurrences from the model
    :return: The loss associated with this batch
    """
    return K.sum(K.pow(K.clip(y_true / X_MAX, 0.0, 1.0), a) *
                 K.square(y_pred - K.log(y_true)), axis=-1)


