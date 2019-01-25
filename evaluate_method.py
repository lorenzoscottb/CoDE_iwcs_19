
from keras.layers import Input, Embedding, Dot, Reshape, Add
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

OUTPUT_FOLDER = 'output/'
DATA_FOLDER = 'data/'

CENTRAL_EMBEDDINGS = 'central_embeddings'
CONTEXT_EMBEDDINGS = 'context_embeddings'
CENTRAL_BIASES = 'central_biases'
CONTEXT_BIASES = 'context_biases'

AGGREGATED_EMBEDDINGS = 'agg_embeddings'
CORRELATION_MATRIX = 'corr_matrix'

INDEX2WORD = 'index-word.pkl'
WORD2INDEX = 'word-index.pkl'

X_MAX = 100
a = 3.0 / 4.0


def np_tensor(x, y):

    import numpy as np

    return np.random.uniform(low=-1, high=1, size=(x, y))


def random_training_set(context_size=500, focal_size=30000):

    sample_range = int(np.random.randint(30000, high=2*focal_size, size=1))

    cm = {(
        int(np.random.randint(0, high=focal_size, size=1)),
        int(np.random.randint(0, high=context_size, size=1))):
        np.random.uniform(0.0, 2.5)
        for i in range(sample_range)}

    shared_cooccurrences = [(word_ids[0],
                             word_ids[1],
                             count)
                            for word_ids, count in cm.items()
                            ]

    return shared_cooccurrences, sample_range


def glove_model(focal_size=10, context_size=5, vector_dim=10):

    """
    A Keras implementation of the GloVe architecture
    :param focal_size: The number of distinct words
    :param vector_dim: The vector dimension of each word
    :return:
    """

    input_focal = Input((1,), name='central_word_id')
    input_context = Input((1,), name='context_word_id')

    central_embedding = Embedding(focal_size,
                                  vector_dim,
                                  input_length=1,
                                  name=CENTRAL_EMBEDDINGS)
    central_bias = Embedding(focal_size,
                             1,
                             input_length=1,
                             name=CENTRAL_BIASES)

    context_embedding = Embedding(context_size, vector_dim,
                                  input_length=1,
                                  name=CONTEXT_EMBEDDINGS)
    context_bias = Embedding(context_size,
                             1,
                             input_length=1,
                             name=CONTEXT_BIASES)

    vector_focal = central_embedding(input_focal)
    vector_context = context_embedding(input_context)

    bias_focal = central_bias(input_focal)
    bias_context = context_bias(input_context)

    dot_product = Dot(axes=-1)([vector_focal, vector_context])
    dot_product = Reshape((1, ))(dot_product)
    bias_focal = Reshape((1,))(bias_focal)
    bias_context = Reshape((1,))(bias_context)

    prediction = Add()([dot_product, bias_focal, bias_context])

    model = Model(inputs=[input_focal, input_context], outputs=prediction)
    model.compile(loss=custom_loss, optimizer=Adam())

    return model


def custom_loss(y_true, y_pred, X_MAX = 100, a = 3.0 / 4.0):

    """
    This is GloVe's loss function
    :param y_true: The actual values, in our case the 'observed' X_ij co-occurrence values
    :param y_pred: The predicted (log-)co-occurrences from the model
    :return: The loss associated with this batch
    """
    return K.sum(K.pow(K.clip(y_true / X_MAX, 0.0, 1.0), a) *
                 K.square(y_pred - K.log(y_true)), axis=-1)


def get_weights(model, layer_name='context_embeddings'):

    weights = None
    for layer in model.layers:
        if layer.name == layer_name:
            weights = layer.get_weights()

    return weights


def set_weight(weight, model):

    for layer in model.layers:
        if layer.name == 'context_embeddings':
            layer.set_weights([weight])


dict_size = 500
dim = 10
epochs = 33
sub_epochs = 3
a_t, a_samples = random_training_set()
a = glove_model(focal_size=len(a_t), context_size=dict_size, vector_dim=dim)

b_t, b_samples = random_training_set()
b = glove_model(focal_size=len(b_t), context_size=dict_size, vector_dim=dim)

c_t, c_samples = random_training_set()
c = glove_model(focal_size=len(b_t), context_size=dict_size, vector_dim=dim)

models = [a, b, c]
training_sets = [a_t, b_t, c_t]

print('\n')
shared_emb = np_tensor(dict_size, dim)

for epoch in range(epochs):

    print('Global epoch ', epoch+1)
    if epoch == epochs:
        last_train_status = True

    for sub_epoch in range(sub_epochs):

        # print('training %s model' % models[sub_epoch].model_name)
        print(sub_epoch+1)

        model = models[sub_epoch]
        set_weight(shared_emb, model)
        i_s, j_s, counts = zip(*training_sets[sub_epoch])
        model.fit([np.array(i_s), np.array(j_s)], np.array(counts),
                  epochs=sub_epochs,
                  batch_size=512)
        shared_emb = np.array(get_weights(model)).reshape(dict_size, dim)

