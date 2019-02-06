
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine  # cosine distance!!!
from scipy.stats.stats import spearmanr
from apt_toolkit.utils import vector_utils as vu
from apt_toolkit.composition import mozart


# 500 actual token
vector_file = 'path/to/apt_vectors.dill'

vectors = vu.load_vector_cache(vector_in_file=vector_file)

ctx_em = 'path/to/context_embeddigns.txt'
amod_em = 'path/to/amod_focal_embeddigns.txt'
dobj_em = 'path/to/dobj_focal_embeddigns.txt'
nsubj_em = 'path/to/nsubj_focal_embeddigns.txt'

ml_10 = '/path/to/ml_10_dataset.txt'


def cosine_similarity(v1, v2):

    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"

    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y

    return sumxy/math.sqrt(sumxx*sumyy)


def dir_split(dir):

    file_name = dir.split('/')[-1]
    file_dir = dir.replace(file_name, '')

    return file_name, file_dir


def string2vec(string_vecor):
    return np.fromstring(string_vecor,
                         dtype="float32",
                         sep=" ")


sv = lambda vec: str(vec).replace('[', '').replace(']', '').replace('\n', '')


def path_noun(noun, dir, path, dot=':'):

    file = open(dir, 'r')
    line = file.readline()
    emb = []
    for line in file:
        if line.split(' ', 1)[0] == 'en_'+ path + dot + noun:
            emb = line.split(' ', 1)[1]

    return string2vec(emb)


adj_emb = lambda a: np.concatenate((path_noun(a, amod_em, 'amod'),
                                    path_noun(a, amod_em, 'dobj»amod'),
                                    path_noun(a, amod_em, 'nsubj»amod')))

noun_emb = lambda n: np.concatenate((path_noun(n, ctx_em, '', dot=''),
                                     path_noun(n, dobj_em, 'dobj'),
                                     path_noun(n, nsubj_em, 'nsubj')))


def run_original_ml_10(vectors, test_phrase='adjectivenouns', composition='concatenation'):

    # test_phrase = 'adjectivenouns'
    # composition = 'add'
    df = pd.read_csv(ml_10)
    ml_values = []
    cs_values = []
    for index, element in enumerate(df.values):
        e = element[0].split()
        if e[1] != test_phrase:
            continue
        try:
            if composition == 'add':
                adj_v_1 = vu.create_offset_vector(vector=vectors[e[3]], offset_path='amod')
                nn_v_1 = vectors[e[4]]
                composed_v_1 = mozart.union_apts(vector_1=adj_v_1, vector_2=nn_v_1)
                adj_v_2 = vu.create_offset_vector(vector=vectors[e[5]], offset_path='amod')
                nn_v_2 = vectors[e[6]]
                composed_v_2 = mozart.union_apts(vector_1=adj_v_2, vector_2=nn_v_2)
    
            if composition == 'intersect':
                adj_v_1 = vu.create_offset_vector(vector=vectors[e[3]], offset_path='amod')
                nn_v_1 = vectors[e[4]]
                composed_v_1 = mozart.union_apts(vector_1=adj_v_1, vector_2=nn_v_1)
                adj_v_2 = vu.create_offset_vector(vector=vectors[e[5]], offset_path='amod')
                nn_v_2 = vectors[e[6]]
                composed_v_2 = mozart.intersect_apts(vector_1=adj_v_2, vector_2=nn_v_2)
    
            if composition == 'path_intersect':
                adj_v_1 = vu.create_offset_vector(vector=vectors[e[3]], offset_path='amod')
                nn_v_1 = vectors[e[4]]
                composed_v_1 = mozart.union_apts(vector_1=adj_v_1, vector_2=nn_v_1)
                adj_v_2 = vu.create_offset_vector(vector=vectors[e[5]], offset_path='amod')
                nn_v_2 = vectors[e[6]]
                composed_v_2 = mozart.path_intersect_apts(vector_1=adj_v_2, vector_2=nn_v_2)
    
            ml_values.append(int(e[-1]))
            cs_values.append(vu.calculate_similarity(composed_v_1,
                                                     composed_v_2,
                                                     cosine))
            print('%s:collected' % e[3:7])

        except:
            print('%s:something thing whent wrong' % e[3:7])

    print('oroginal vectors ', spearmanr(ml_values, cs_values))


def run_reduced_ml_10(test_phrase='adjectivenouns', composition='concatenation'):

#   test_phrase='adjectivenouns'
#   composition='concatenation'
    df = pd.read_csv(ml_10)
    ml_values = []
    cs_values = []
    for index, element in enumerate(df.values):
        e = element[0].split()
        if e[1] != test_phrase:
            continue
        try:
            # some useful debug
            # adj_emb(e[3])
            # adj_emb(e[5])
            # path_noun(e[4], ctx_em, '', dot='')
            # path_noun(e[5], ctx_em, '', dot='')
            if composition == 'concatenation':
                an_1 = np.concatenate((adj_emb(e[3]), noun_emb(e[4])))
                an_2 = np.concatenate((adj_emb(e[5]), noun_emb(e[6])))
            else:
                an_1 = adj_emb(e[3]) + noun_emb(e[4])
                an_2 = adj_emb(e[5]) + noun_emb(e[6])
            ml_values.append(int(e[-1]))
            cs_values.append(cosine_similarity(an_1, an_2))
            print('%s:collected' % e[3:7])
        # # if index == 8:
        # #     break
        except:
            print('%s:something thing whent wrong' % e[3:7])

    print('reduced vectors ', spearmanr(ml_values, cs_values))


run_original_ml_10(vectors)
run_reduced_ml_10()

