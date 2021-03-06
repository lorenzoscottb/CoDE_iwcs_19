

from __future__ import division
import os
import numpy as np
from numba import jit
from apt_toolkit.utils import vector_utils as vu
from keras.layers import Input, Embedding, Dot, Reshape, Add
from keras.models import Model
from keras.optimizers import Adam


class Glove_Model():

    def __init__(self, model_name, paths=['amod', 'dobj', 'nsubj'], merging_operator='->', max_depth=2):

        self.model_name = model_name
        self.context_vocabulary_id = None
        self.context_vocabulary = None
        self.focal_vocabulary_id = None
        self.focal_vocabulary = None
        self.paths = paths
        self.max_depth = max_depth
        self.__co_occurrences = None
        self.mrg = merging_operator

    def fit_to_vectors(self, vectors, use_sample=False, use_possible_paths=False, expand_vocabulary=False):
        '''''
        right now the context vocabulary is limited to the original one
        to remove this limit remove the last and condition from the
        listchomp.
        '''
        print('\nfitting the %s GloVe model...' %self.model_name)
        vectors = self.load_apt(vectors)
        self.original_vocabulary = vectors.keys()

        if not self.context_vocabulary:
            self.context_vocabulary = self.ctx_v(vectors, self.paths,
                                                 self.max_depth,
                                                 expand_vocabulary=expand_vocabulary)  # words within single lexems
            self.context_vocabulary_id = {word: i for i, word in enumerate(self.context_vocabulary)}

        if use_sample:
            print('using a semple of the overall vectors...')
            vectors = vector_sample(vectors)

        if use_possible_paths:
            self.possible_paths = self.collect_paths(vectors, self.model_name, self.max_depth)
            self.focal_vocabulary = self.fcl_v_pp(vectors, self.possible_paths)  # vocab->path (for all possible path)
            self.focal_vocabulary_id = {word: i for i, word in enumerate(self.focal_vocabulary)}
            self.__co_occurrences = self.__prepare_training_set_pp(vectors)

        else:
            self.focal_vocabulary = self.fcl_v(vectors)  # vocab->path (for existing path)
            self.focal_vocabulary_id = {word: i for i, word in enumerate(self.focal_vocabulary)}
            self.__co_occurrences = self.__prepare_training_set(vectors)


        # input_focal, input_context, (real)ppmi values
        self.__i_indices, self.__j_indices, self.__counts = zip(*self.__co_occurrences)
        print('fitting completed...')

    @jit(parallel=True)
    def load_apt(self, vec_dir):
        return vu.load_vector_cache(vector_in_file=vec_dir)

    @jit(parallel=True)
    def collect_paths(self, vectors, path_end, path_depth):
        print('collectiong possible word_paths...')
        pp = set([paths.split(':')[0] for word in vectors.keys()
                  for paths in vectors[word]
                  if self.model_name in paths.split(':')[0].split('»')[0] and
                  len(paths.split(':')[0].split('»')) <= path_depth])

        return list(pp)

    @jit(parallel=True)
    def local_context(self, vectors, path_end, path_depth, expand_vocabulary=False):

        ctx_v = set([paths.split(':')[1] for word in vectors.keys()
                         for paths in vectors[word]
                         if path_end in paths.split(':')[0].split('»')[0] and
                         len(paths.split(':')[0].split('»')) <= path_depth and
                         paths.split(':')[1] in self.original_vocabulary])

        return list(ctx_v)

    @jit(parallel=True)
    def ctx_v(self, vectors, paths, path_depth, expand_vocabulary=False):
        print('collecting global context vocabulary...')
        word = []

        for path in paths:
            word += self.local_context(vectors, path, path_depth, expand_vocabulary=expand_vocabulary)

        voc = set(word)
        print('compleated. Context-vocabulary has len: %s' % len(voc))

        return voc

    @jit(parallel=True)
    def fcl_v(self, vectors):

        print('collecting focal vocabulary...')
        words = [
                word + self.mrg + path.split(':')[0]
                for word in vectors.keys() for path in vectors[word].keys()
                if self.model_name in path.split(':')[0].split('»')[0] and
                len(path.split(':')[0].split('»')) < (self.max_depth + 1) and
                path.split(':')[1] in self.original_vocabulary
                ]

        voc = set(words)
        print('compleated. Focal-vocabulary has len: %s' % len(voc))
        return voc

    @jit(parallel=True)
    def fcl_v_pp(self, vectors, possible_paths):

        print('collecting possible-paths focal vocabulary...')
        words = [word + self.mrg + path for path in possible_paths
                for word in vectors.keys()]

        print('compleated. possible-paths Focal-vocabulary has len: %s' % len(words))
        return words

    @jit(parallel=True)
    def __prepare_training_set(self, vectors):
        # 15/02/2019: we will train just over existing ctxw_psths,
        # rest will have random embeddings
        print('preparing training set...')

        co_occ = [
                  (self.focal_vocabulary_id[word + self.mrg + feature.split(':')[0]],
                   self.context_vocabulary_id[feature.split(':')[1]],
                   vectors[word][feature])
                   for word in vectors.keys() for feature in vectors[word].keys()
                   if self.model_name in feature.split(':')[0].split('»')[0] and
                   len(feature.split(':')[0].split('»')) <= self.max_depth and
                    feature.split(':')[1] in vectors.keys()
                  ]

        return co_occ

    @jit(parallel=True)
    def __prepare_training_set_pp(self, vectors):
        # 15/02/2019: we will train just over existing ctxw_psths,
        # rest will have random embeddings
        print('preparing training set...')

        co_occ = [
                  (self.focal_vocabulary_id[word + self.mrg + p_c.split(':')[0]],
                   self.context_vocabulary_id[p_c.split(':')[1]],
                   vectors[word][p_c])
                   for word in vectors.keys() for p_c in vectors[word].keys()
                   if self.model_name in p_c.split(':')[0].split('»')[0] and
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

    def asimmetric_glove(self, dimension, allow_grouth=False):
        
        # set limit to Keras' expansion on GPU
        if allow_grouth:
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

    def model_evaluation(self, set_context_embeddings=None, metric='spermanr'):

        print('running %s model evaluation:' %self.model_name)
        # from sklearn.metrics.pairwise import euclidean_distances

        # new keras model will ad number to layer name
        if self.paths.index(self.model_name) != 0:
            index_end = '_'+str(self.paths.index(self.model_name)) # new keras model will ad number to layer name
        else:
            index_end = ''

        self.layer_index = {self.model.weights[index].name.split('/')[0]: index
                            for index, layer in enumerate(self.model.weights)}

        if set_context_embeddings is None:
            context_embeddings = [layer.get_weights()[0] for layer in self.model.layers
                                  if layer.name == self.CONTEXT_EMBEDDINGS]
        else:
            context_embeddings = [set_context_embeddings]

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
                new_matrix[y][x] = glove_reverse(self.model.get_weights()[self.layer_index['context_biases'+index_end]][j_s[index]],
                                                 self.model.get_weights()[layer_index['central_biases'+index_end]][i_s[index]],
                                                 context_embeddings[0][j_s[index]],
                                                 self.model.get_weights()[layer_index['central_embeddings'+index_end]][i_s[index]])
            print('computing Frobenius distance...')
            return frobenius_distance(original_matrix, new_matrix)

        if metric == 'Jensen-Shannon':

            i_s, j_s, ppmis = self.training_set()

            new_ppmis = np.zeros(len(ppmis))
            print('collecting new distribution...')
            for index, ppmi in enumerate(ppmis):
                new_ppmis[index] = glove_reverse(self.model.get_weights()[layer_index['context_biases'+index_end]][j_s[index]],
                                                 self.model.get_weights()[layer_index['central_biases'+index_end]][i_s[index]],
                                                 context_embeddings[0][j_s[index]],
                                                 self.model.get_weights()[layer_index['central_embeddings'+index_end]][i_s[index]])

            print('computing Jensen-Shannon divergence...')
            return jsd(ppmis, new_ppmis)

        if metric == 'spermanr':

            from scipy.stats.stats import spearmanr

            i_s, j_s, ppmis = self.training_set()

            new_ppmis = np.zeros(len(ppmis))
            print('collecting new distribution...')
            for index, ppmi in enumerate(ppmis):
                new_ppmis[index] = glove_reverse(self.model.get_weights()[self.layer_index['context_biases'+index_end]][j_s[index]],
                                                 self.model.get_weights()[self.layer_index['central_biases'+index_end]][i_s[index]],
                                                 context_embeddings[0][j_s[index]],
                                                 self.model.get_weights()[self.layer_index['central_embeddings'+index_end]][i_s[index]])

            print('computing SPerman correlation...')
            return spearmanr(ppmis, new_ppmis), new_ppmis

    def save_model(self, name=''):

        new_model = Compressed_model(self.model_name,
                                     self.context_vocabulary_id,
                                     self.focal_vocabulary_id,
                                     self.__co_occurrences,
                                     self.model.get_weights())

        save_object(new_model, name+self.model_name)
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
        self.vec_dim = load_model.weights[0].shape[1]
        self.weights = load_model.weights
        self.asimmetric_glove(self.vec_dim)
        self.model.set_weights(self.weights)

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


