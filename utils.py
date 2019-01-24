
import os

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
        file = open(os.getcwd()+'/'+model.model_name+'_focal_embeddigns.txt', 'w')
        for word in vocab:
            file.write('en_' + word + ' ' + str(list(embeddings[index])).replace(',', '').strip('[').strip(']') + '\n')
            # break
        file.close()
