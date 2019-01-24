
from Keras_GloVe import Model
import numpy as np
import os
import utils
from apt_toolkit.utils import vector_utils as vu


# model settings
ideal_epochs = 100
dim = 10

vectors = 'path/to/vectors'
vectors = vu.load_vector_cache(vector_in_file=vectors)

print('setting up GloVe models...')
amod_model = Model(model_name='amod')
amod_model.path_and_fit(vectors, prepare_batches=True, save_training_set=False)
amod_model.asimmetric_glove(dim)

dobj_model = Model(model_name='dobj')
dobj_model.path_and_fit(vectors, prepare_batches=True, save_training_set=False)
dobj_model.asimmetric_glove(dim)

nsubj_model = Model(model_name='nsubj')
nsubj_model.path_and_fit(vectors, prepare_batches=True, save_training_set=False)
nsubj_model.asimmetric_glove(dim)

models = [amod_model, dobj_model, nsubj_model]

utils.shared_training(ideal_epochs, dim, 100)

utils.write_context_embeddings(models)
utils.write_focal_embeddings(models)

