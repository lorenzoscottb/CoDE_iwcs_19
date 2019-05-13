
"""
author : Lorenzo Bertolini

code generates a CoDE (.txt) space, devided in one file for 
context representation and one focal for each selcted dependecy 
relation – here amod, dobj, nsubj and nmod

The code refers to the 2019 IWCS poster version
"""

from Glove_Reduction import Glove_Model
import numpy as np
import os
import utils
from apt_toolkit.utils import vector_utils as vu


from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


vectors = 'path_to_apt_vectors'
vec_name = 'vectors_name'


def np_tensor(x, y):

    return np.random.uniform(low=-1, high=1, size=(x, y))


def write_context_embeddings(models):

    print('Writing down context embeddings...')
    context_vocabulary = models[-1].context_words # last to train
    context_embeddings = np.array((models[-1]).get_weights()[0])
    file = open(os.getcwd()+'/'+vec_name+'_context_dim-'+str(dim)+'.txt', 'w')
    for index, word in enumerate(context_vocabulary):
        file.write(word+' '+str(list(context_embeddings[index])).replace(',', '').strip('[').strip(']')+'\n')
        # break
    file.close()


def write_focal_embeddings(models, dim):

    for index, model in enumerate(models):
        print('writing down focal embeddigns for', model.model_name, '...')
        vocab = models[index].focal_words
        embeddings = model.get_weights(layer_name='central_embeddings')[0]
        file = open(os.getcwd()+'/'+ vec_name +'_focal_'+model.model_name+'_dim-'+str(dim)+'.txt', 'w')
        for i, word in enumerate(vocab):
            file.write(word + ' ' + str(list(embeddings[i])).replace(',', '').strip('[').strip(']') + '\n')
            # break
        file.close()


paths = ['amod', 'dobj', 'nsubj', 'nmod']
ideal_epochs = 100
epochs = (int(ideal_epochs/len(paths)))*len(paths)
sub_epochs = 3
dim = 300

print('reduce sample? [yes/no]')
i = str(input())
if i == 'yes':
    sample_state = True
else:
    sample_state = False

print('setting up GloVe models...')
amod_model = Glove_Model(model_name='amod', paths=paths)
amod_model.fit_to_vectors(vectors, use_sample=sample_state)
amod_model.asimmetric_glove(dim)

dobj_model = Glove_Model(model_name='dobj', paths=paths)
dobj_model.set_context(amod_model)
dobj_model.fit_to_vectors(vectors, use_sample=sample_state)
dobj_model.asimmetric_glove(dim)

nsubj_model = Glove_Model(model_name='nsubj', paths=paths)
nsubj_model.set_context(amod_model)
nsubj_model.fit_to_vectors(vectors, use_sample=sample_state)
nsubj_model.asimmetric_glove(dim)

nmod_model = Glove_Model(model_name='nmod', paths=paths)
nmod_model.set_context(amod_model)
nmod_model.fit_to_vectors(vectors, use_sample=sample_state)
nmod_model.asimmetric_glove(dim)

models = [amod_model, dobj_model, nsubj_model, nmod_model]
r = [ i for i in range(0,99,15)]

print('\n')
context_size = len(amod_model.context_vocabulary)
shared_emb = np_tensor(context_size, dim)

for epoch in range(epochs):

    print('Global epoch ', epoch+1)
    if epoch == epochs:
        last_train_status = True

    for i_m in range(len(models)):

        print('training %s model' % paths[i_m])
        # print(sub_epoch+1)

        model = models[i_m]
        model.set_weight(shared_emb)
        i_s, j_s, counts = models[i_m].training_set()
        model.train([np.array(i_s), np.array(j_s)],
                    np.array(counts),
                    sub_epochs,
                    512)
        shared_emb = np.array(model.get_weights()[0].reshape(context_size, dim))
    if epoch in r:
        for model in models:
            model.save_model()
    
for model in models:
    model.save_model()

write_context_embeddings(models)
write_focal_embeddings(models, dim)

