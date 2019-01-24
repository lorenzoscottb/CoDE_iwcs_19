
from Keras_GloVe import Model
import numpy as np
import os
from apt_toolkit.utils import vector_utils as vu


def np_tensor(x, y):

    import numpy as np

    return np.random.uniform(low=-1, high=1, size=(x, y))


def write_context_embeddings(models):

    print('Writing down context embeddings...')
    context_vocabulary = models[-1]._Model__context_words # last to train
    context_embeddings = np.array((models[-1]).get_weights()[0])
    file = open(os.getcwd()+'/context_embeddigns.txt', 'w')
    for index, word in enumerate(context_vocabulary):
        file.write('en_'+word+' '+str(list(context_embeddings[index])).replace(',', '').strip('[').strip(']')+'\n')
        # break
    file.close()


def write_focal_embeddings(models):

    for index, model in enumerate(models):
        print('writing down focal embeddigns for', model.model_name, '...')
        vocab = models[index].words
        embeddings = model.get_weights(layer_name='central_embeddings')[0]
        file = open(os.getcwd()+'/'+paths[index]+'_focal_embeddigns.txt', 'w')
        for word in vocab:
            file.write('en_' + word + ' ' + str(list(embeddings[index])).replace(',', '').strip('[').strip(']') + '\n')
            # break
        file.close()


# model settings
test_range = 500
cwd = os.getcwd()
paths = ['amod', 'dobj', 'nsubj']
ideal_epochs = 100
epochs = (int(ideal_epochs/len(paths)))*len(paths)
sub_epochs = len(paths)
dim = 10
max_vocab = 10000000


# 96920 actual token
vectors = 'path/to/vectors'

# vectors = vu.load_vector_cache(vector_in_file=vectors)

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

print('\n')
shared_emb = np_tensor(amod_model.dict_size, dim)

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
        shared_emb = np.array(model.get_weights()[0].reshape(model.dict_size, dim))

write_context_embeddings(models)
write_focal_embeddings(models)

