# CoDE: learning Composable Dependency Embeddings 
This repo contains code and data for the CoDE model, presented as a poster at the Vector Semantics for
Discourse and Dialogue [workshop](https://www.aclweb.org/anthology/W19-09.pdf) of the 2019 IWCS conference.

## Abstract 

Through the years, a restricted number of authors have tackled the non-trivial problem of encoding syntactic information in 
distributional representations by injecting dependency-relation knowledge directly into word embeddings. Although such 
representations should bring a clear advantage in complex representations, such as at the phrasal and sentence level, these 
models have been tested mainly through word-word similarity benchmarks or with rich neural architecture. Outside the 
embeddings domain, the APT model has offered an effective resource for modelling compositionality via syntactic 
contextualization. In this work, we present a novel model, built on top of GloVe, to reduce APT representations to a low-
dimensionality dense dependency-based vectors, that showcase APT-like composition ability. We then propose a detailed 
investigation of the nature of these representations, as well as their usefulness and contribution in semantic composition.  

## Vectors 

CoDE vectors used for experiment in the poster can be downloaded [here](https://drive.google.com/file/d/1IjJjSJYIU_u-qU-sa67TrjuErZopdiyJ/view)

## Acknowledgements
CoDE is a [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) based method, built over a [Keras](https://github.com/erwtokritos/keras-glove) implementation. Future versions of the model will be hosted elsewhere
