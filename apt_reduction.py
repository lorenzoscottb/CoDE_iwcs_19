
from Glove_Reduction import Model
import numpy as np
import os
import utils
from apt_toolkit.utils import vector_utils as vu


# model settings
paths = ['amod', 'dobj', 'nsubj']
ideal_epochs = 33
epochs = (int(ideal_epochs/len(paths)))*len(paths)
sub_epochs = len(paths)
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
        
utils.write_context_embeddings(models)
utils.write_focal_embeddings(models)

new_ppmis = []
ppmis = []
model_tags = []
for index, model in enumerate(models):
    ppmi = model.ppmis
    model_tag = [model.model_name for element in ppmi]
    evaluation, new_ppmi = model.model_evaluation(context_embeddings=shared_emb)
    print('%s model correlation' % models[index].model_name, evaluation)
    new_ppmis += list(new_ppmi)
    ppmis += list(ppmi)
    model_tags += list(model_tag)

data_names = ['real_values', 'model_values', 'model']
data = [ppmis, new_ppmis, model_tags]
vr = pd.DataFrame()
for index, columns in enumerate(data_names):
    vr[columns] = data[index]

sns.set(style="whitegrid")
sns.scatterplot(x='real_values', y='model_values', hue='model', palette=['r', 'b', 'g'], data=vr)
plt.savefig(os.getcwd()+'/as_glove_original-predictec-values')

