

import numpy as np
from txt2space import Space
from scipy.stats.stats import spearmanr
import pandas as pd

ctx = 'path_to_context'
amod = 'path_to_focal_amod'
nmod = 'path_to_focal_nmod'
dobj = 'path_to_focal_dobj'
nsubj = 'path_to_focal_nsubj'
ml_10 = 'path_to_mitchell_lapata_test'

embeddings = [amod, dobj, nsubj, ctx]
word_vec = {}
for f in embeddings:
    file = open(f, 'r')
    line = file.readline()
    for line in file:
        word, vec = line.split(' ')[0], np.fromstring(line.split(' ', 1)[1].strip('\n'), sep=' ')
        word_vec[word.replace('en_', '')] = vec
    print('done with', f)

 
ml_df = pd.read_csv(ml_10)
ml_values = []
cs_values = []
c = 0

for index, ml_e in enumerate(ml_df.values):
    
    if  ml_e[1] == 'adjectivenouns':
        try:
            c_1  = compose_vectors([word_vec[ml_e[3]+'->_amod'], composed_nn(ml_e[4])])
            c_2  = compose_vectors([word_vec[ml_e[5]+'->_amod'], composed_nn(ml_e[6])])

            cs_values.append(cosine_similarity(c_1, c_2))       
            ml_values.append(int(ml_e[-1]))
            #         print('%s:collected' % e[3:7])
            c+= 1
        except Exception as ex:
            print(ex)
    elif ml_e[1] == 'verbobjects':
        try:
            c_1  = compose_vectors([word_vec[ml_e[3]+'->_dobj'], composed_nn(ml_e[4])])
            c_2  = compose_vectors([word_vec[ml_e[5]+'->_dobj'], composed_nn(ml_e[6])])

            cs_values.append(cosine_similarity(c_1, c_2))       
            ml_values.append(int(ml_e[-1]))
            #         print('%s:collected' % e[3:7])
            c+= 1
        except Exception as ex:
            print(ex)
    elif ml_e[1] == 'compoundnouns':

        try:
            c_1  = compose_vectors([word_vec[ml_e[3]+'->_nmod'], composed_nn(ml_e[4])])
            c_2  = compose_vectors([word_vec[ml_e[5]+'->_nmod'], composed_nn(ml_e[6])])

            cs_values.append(cosine_similarity(c_1, c_2))       
            ml_values.append(int(ml_e[-1]))
            #         print('%s:collected' % e[3:7])
            c+= 1  
        except Exception as ex:
            print(ex)
      
print('Mithcell-Lapata 2010 task. coverage {}/{}: {}'.format(c, int(len(ml_df.values)), 
                                             spearmanr(ml_values, cs_values)))
