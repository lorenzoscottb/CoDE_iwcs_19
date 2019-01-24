
import os

def np_tensor(x, y):

    import numpy as np

    return np.random.uniform(low=-1, high=1, size=(x, y))

def write_context_embeddings(models):

    print('Writing down context embeddings...')
    context_vocabulary = models[-1].context_words # last to train
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
        file = open(os.getcwd()+'/'+model.model_name+'_focal_embeddigns.txt', 'w')
        for word in vocab:
            file.write('en_' + word + ' ' + str(list(embeddings[index])).replace(',', '').strip('[').strip(']') + '\n')
            # break
        file.close()
     
def shared_training(models, emdeddings_dimensions, epochs)
    
    ideal_epochs = epochs
    epochs = (int(ideal_epochs/len(paths)))*len(paths)
    sub_epochs = len(models)

    print('\n')
    shared_emb = utils.np_tensor(amod_model.dict_size, emdeddings_dimensions)

    for epoch in range(epochs):

        print('Global epoch ', epoch+1)
#         if epoch == epochs:
#             last_train_status = True

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
        
