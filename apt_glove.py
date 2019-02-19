

from __future__ import division
import os
import numpy as np
from numba import jit
from apt_toolkit.utils import vector_utils as vu
from keras.layers import Input, Embedding, Dot, Reshape, Add
from keras.models import Model
from keras.optimizers import Adam


class Glove_Model():

    def __init__(self, model_name, paths, merging_operator='->'):

        self.model_name = model_name
        self.context_vocabulary_id = None
        self.context_vocabulary = None
        self.focal_vocabulary_id = None
        self.focal_vocabulary = None
        self.paths = paths
        self.__co_occurrences = None
        self.mrg = merging_operator

    def fit_to_vectors(self, vectors, use_sample=False):
        print('\nfitting the %s GloVe model...' %self.model_name)
        vectors = self.load_apt(vectors)
        if use_sample:
            print('using a semple of the overall vectors...')
            vectors = vector_sample(vectors)

        self.possible_paths = self.collect_paths(vectors, self.model_name)

        if not self.context_vocabulary:
            self.context_vocabulary = self.ctx_v(vectors, self.paths)  # words within single lexems
            self.context_vocabulary_id = {word: i for i, word in enumerate(self.context_vocabulary)}

        self.focal_vocabulary = self.fcl_v(vectors, self.possible_paths)  # vocab->path (for each path)
        self.focal_vocabulary_id = {word: i for i, word in enumerate(self.focal_vocabulary)}

        self.__co_occurrences = self.__prepare_training_set(vectors)

        self.__i_indices, self.__j_indices, self.__counts = zip(*self.__co_occurrences)
        print('fitting completed...')

    @jit(parallel=True)
    def load_apt(self, vec_dir):
        return vu.load_vector_cache(vector_in_file=vec_dir)

    @jit(parallel=True)
    def collect_paths(self, vectors, path_end, path_depth=3):
        print('collectiong possible word_paths...')
        pp = set([paths.split(':')[0] for word in vectors.keys()
                  for paths in vectors[word]
                  if paths.split(':')[0].endswith(path_end) and
                  len(paths.split(':')[0].split('»')) <= path_depth])

        return list(pp)

    @jit(parallel=True)
    def local_context(self, vectors, path_end, path_depth=3):

        ctx_v = set([paths.split(':')[1] for word in vectors.keys()
                     for paths in vectors[word]
                     if paths.split(':')[0].endswith(path_end) and
                     len(paths.split(':')[0].split('»')) <= path_depth])

        return list(ctx_v)

    @jit(parallel=True)
    def ctx_v(self, vectors, paths):
        print('collecting global context vocabulary...')
        word = []

        for path in paths:
            word += self.local_context(vectors, path)

        print('compleated. Context-vocabulary has len: %s' % len(word))

        return set(word)

    @jit(parallel=True)
    def fcl_v(self, vectors, possible_paths):

        print('collecting focal vocabulary...')
        words = [word + self.mrg + path for path in possible_paths
                for word in vectors.keys()]

        print('compleated. Focal-vocabulary has len: %s' % len(words))

        return words

    @jit(parallel=True)
    def __prepare_training_set(self, vectors):
        # 15/02/2019: we will train just over existing ctxw_psths,
        # rest will have random embeddings
        print('preparing training set...')

        co_occ = [
                  (self.focal_vocabulary_id[word + self.mrg + p_c.split(':')[0]],
                   self.context_vocabulary_id[p_c.split(':')[1]],
                   vectors[word][p_c])
                   for word in vectors.keys() for p_c in vectors[word].keys()
                   if p_c.split(':')[0].endswith(self.model_name) and
                   len(p_c.split(':')[0].split('»')) < 4
                  ]

        # co_occ_clm = [
        #               (self.focal_vocabulary_id[focal],
        #               self.context_vocabulary_id[context],
        #               vectors[focal.split(self.mrg, 1)[0]][focal.split(self.mrg, 1)[1] + ':' + context])
        #               for focal in self.focal_vocabulary for context in self.context_vocabulary
        #               # if focal.split(self.mrg, 1)[1] + ':' + context in vectors[focal.split(self.mrg, 1)[0]]
        #               ]
        # co_occ_row = [(context_vocabulary_id[context],
        #               focal_vocabulary_id[focal],
        #               vectors[focal.split('.', 1)[0]][focal.split('.', 1)[1] + ':' + context])
        #              for focal in self.focal_vocabulary for context in self.context_vocabulary]
        #
        # co_occ = co_oc_clm + co_oc_row

        return co_occ

    @jit(parallel=True)
    def training_set(self):

        return self.__i_indices, self.__j_indices, self.__counts

    def set_context(self, glove_object):

        print('importing existing context vocabulary...')
        self.context_vocabulary = glove_object.context_vocabulary
        self.context_vocabulary_id = glove_object.context_vocabulary_id

    @property
    def context_words(self):

        return self.context_vocabulary

    @property
    def focal_words(self):

        return self.focal_vocabulary

    @property
    def ppmis(self):
        return self.__counts

    def asimmetric_glove(self, dimension):

        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session

        # set limit to Keras' expansion on GPU
        from keras.backend.tensorflow_backend import set_session
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.Session(config=config)
        set_session(sess)  # set this TensorFlow session as the default session for Keras

        self.CENTRAL_EMBEDDINGS = 'central_embeddings'
        self.CONTEXT_EMBEDDINGS = 'context_embeddings'
        self.CENTRAL_BIASES = 'central_biases'
        self.CONTEXT_BIASES = 'context_biases'

        self.model = None

        self.vector_dim = dimension
        self.focal_size = len(self.focal_vocabulary)
        self.context_size = len(self.context_vocabulary)

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

        context_embedding = Embedding(self.context_size,
                                      self.vector_dim,
                                      input_length=1,
                                      name=self.CONTEXT_EMBEDDINGS)
        context_bias = Embedding(self.context_size,
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

    def model_evaluation(self, context_embeddings=None, metric='spermanr'):

        print('running %s model evaluation:' %self.model_name)
        # from sklearn.metrics.pairwise import euclidean_distances

        b_cntx = None
        b_cntr = None
        e_cntx = None
        e_cntr = None

        print('extracting embeddigns and wheights...')
        for layer in self.model.layers:
            if layer.name == self.CENTRAL_BIASES:
                b_cntr = layer.get_weights()[0]

            if layer.name == self.CENTRAL_EMBEDDINGS:
                e_cntr = layer.get_weights()[0]

            if layer.name == self.CONTEXT_BIASES:
                b_cntx = layer.get_weights()[0]

            if context_embeddings is None:
                if layer.name == self.CONTEXT_EMBEDDINGS:
                    e_cntx = layer.get_weights()[0]
            else:
                e_cntx = context_embeddings

        if metric == 'frobenius':
            original_matrix = np.zeros((e_cntr.shape[0], e_cntr.shape[0]))
            new_matrix = np.zeros((e_cntr.shape[0], e_cntr.shape[0]))

            i_s, j_s, ppmis = self.training_set()

            differences = np.zeros((len(ppmis), 1))
            print('collecting new matrix...')
            for index, ppmi in enumerate(ppmis):
                y = i_s[index]
                x = j_s[index]

                # print(x,y,ppmi)
                original_matrix[y][x] = ppmi
                new_matrix[y][x] = glove_reverse(b_cntx[x],
                                                      b_cntr[y],
                                                      e_cntx[x],
                                                      e_cntr[y])
            print('computing Frobenius distance...')
            return frobenius_distance(original_matrix, new_matrix)

        if metric == 'Jensen-Shannon':

            i_s, j_s, ppmis = self.training_set()

            new_ppmis = np.zeros(len(ppmis))
            print('collecting new distribution...')
            for index, ppmi in enumerate(ppmis):
                new_ppmis[index] = glove_reverse(b_cntx[j_s[index]],
                                                 b_cntr[i_s[index]],
                                                 e_cntx[j_s[index]],
                                                 e_cntr[i_s[index]])

            print('computing Jensen-Shannon divergence...')
            return jsd(ppmis, new_ppmis)

        if metric == 'spermanr':

            from scipy.stats.stats import spearmanr

            i_s, j_s, ppmis = self.training_set()

            new_ppmis = np.zeros(len(ppmis))
            print('collecting new distribution...')
            for index, ppmi in enumerate(ppmis):
                new_ppmis[index] = glove_reverse(b_cntx[j_s[index]],
                                                 b_cntr[i_s[index]],
                                                 e_cntx[j_s[index]],
                                                 e_cntr[i_s[index]])

            print('computing SPerman correlation...')
            return spearmanr(ppmis, new_ppmis), new_ppmis

    def save_model(self):

        new_model = Compressed_model(self.model_name,
                                     self.context_vocabulary_id,
                                     self.focal_vocabulary_id,
                                     self.__co_occurrences,
                                     self.model.get_weights())

        save_object(new_model, self.model_name)
        print('saved a reduced version of %s model...' % self.model_name)

    def load_model(self, model_object_dir):

        print('loading pre-trained model...')
        import pickle
        #
        with open(model_object_dir, 'rb') as f:
            load_model = pickle.load(f)

        self.model_name = load_model.model_name
        self.context_vocabulary_id = load_model.context_vocabulary_id
        self.context_vocabulary = self.context_vocabulary_id.keys()
        self.focal_vocabulary_id = load_model.focal_vocabulary_id
        self.focal_vocabulary = load_model.focal_vocabulary_id.keys()
        self.__co_occurrences = load_model.co_occ
        self.__i_indices, self.__j_indices, self.__counts = zip(*self.__co_occurrences)
        self.asimmetric_glove(load_model.model.get_weights()[0].shape[1])
        self.model.set_weights(load_model.weights)

        # self.model.set_weights(load_model.weights)


def vector_sample(vectors, sample=3):
    vv = {}
    for i, v in enumerate(vectors.keys()):
        vv[v] = vectors[v]
        if i == sample:
            break
    return vv


def _key_by_value(mydict, value):
    return list(mydict.keys())[list(mydict.values()).index(value)]


def save_object(object, name, dir=os.getcwd()):
    import pickle

    with open(dir + '/' + name + '.pkl', 'wb') as f:
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


@jit()
def weights_f(x, X_MAX = 100, a = 3.0 / 4.0):

    if x < X_MAX:
        return np.power((x/X_MAX), a)

    else:
        return 1.0


@jit()
def glove_reverse(b_c, b_f, v_c, v_f, weight=False):

    x = np.power(2, np.inner(v_c, v_f) + b_c + b_f)
    if weight:
        x = weights_f(x)

    return x


@jit()
def frobenius_distance(matrix_a, matrix_b):

    #  F(a,b) = sqr(trace((a-b)*(a-b)trnsp))

    a = np.matrix(matrix_a)
    b = np.matrix(matrix_b)
    t_b = b.getH()
    trace = ((a-b)*(a-b).getH()).trace()

    return np.sqrt(trace.item())


@jit()
def jsd(p, q, base=np.e):
    import scipy as sp

    '''
        from 
        Implementation of pairwise `jsd` based on  
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    '''
    ## convert to np.array
    p, q = np.asarray(p), np.asarray(q)
    ## normalize p, q to probabilities
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return sp.stats.entropy(p,m, base=base)/2. + sp.stats.entropy(q, m, base=base)/2.


class Compressed_model():

    def __init__(self, name, ct_v, fcl_v, co_occ, weights):

        self.model_name = name
        self.context_vocabulary_id = ct_v
        self.focal_vocabulary_id = fcl_v
        self.co_occ = co_occ
        self.weights = weights


