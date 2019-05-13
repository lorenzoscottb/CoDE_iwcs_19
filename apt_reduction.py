
from Glove_Reduction import Glove_Model
import numpy as np
import os
import utils
from apt_toolkit.utils import vector_utils as vu


vectors = 'path/to/vectors'
vectors = vu.load_vector_cache(vector_in_file=vectors)

paths = ['amod', 'dobj', 'nsubj']
ideal_epochs = 100
epochs = (int(ideal_epochs/len(paths)))*len(paths)
sub_epochs = len(paths)
dim = 300

print('reduce sample? [yes/no]')
i = str(input())
if i == 'yes':
    sample_state = True
else:
    sample_state = False

print('setting up CoDE model...')
amod_model = Glove_Model(model_name='amod', paths=paths)
amod_model.fit_to_vectors(sample_vectors, use_sample=sample_state)
amod_model.asimmetric_glove(dim)

dobj_model = Glove_Model(model_name='dobj', paths=paths)
dobj_model.set_context(amod_model)
dobj_model.fit_to_vectors(sample_vectors, use_sample=sample_state)
dobj_model.asimmetric_glove(dim)

nsubj_model = Glove_Model(model_name='nsubj', paths=paths)
nsubj_model.set_context(amod_model)
nsubj_model.fit_to_vectors(sample_vectors, use_sample=sample_state)
nsubj_model.asimmetric_glove(dim)


models = [amod_model, dobj_model, nsubj_model]

print('\n')
context_size = len(amod_model.context_vocabulary)
shared_emb = utils.np_tensor(context_size, dim)

for epoch in range(epochs):

    print('Global epoch ', epoch+1)
    if epoch == epochs:
        last_train_status = True

    for sub_epoch in range(sub_epochs):

        print('training %s model' % paths[sub_epoch])
        # print(sub_epoch+1)

        model = models[sub_epoch]
        model.set_weight(shared_emb)
        i_s, j_s, counts = models[sub_epoch].training_set()
        model.train([np.array(i_s), np.array(j_s)],
                    np.array(counts),
                    sub_epochs,
                    512)
        shared_emb = np.array(model.get_weights()[0].reshape(context_size, dim))

for model in models:
    model.save_model()

utils.write_context_embeddings(models)
utils.write_focal_embeddings(models, paths)


