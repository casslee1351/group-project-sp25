import gensim
from gensim.models import Word2Vec
import gensim.downloader as api

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def apply_glove(sentences, model="glove-wiki-gigaword-100"):

    # # initialize glove model
    # glove_model = api.load("glove-wiki-gigaword-100")

    print("Models available for use:")
    print(list(gensim.downloader.info()['models'].keys()))

    glove_model = api.load(model)

    ### initialize model
    base_model = Word2Vec(vector_size=100, min_count=1)
    base_model.build_vocab(sentences)
    total_examples = base_model.corpus_count

    base_model.build_vocab(glove_model.index_to_key, update=True)
    base_model.train(sentences, total_examples=total_examples, epochs=base_model.epochs)

    return base_model

def create_glove_matrix(data, target_col, verbose:bool = True):
    """
    Description
    ----------
    TODO

    If glove.42B.300d.txt is not in the project directory, please navigate to
    https://nlp.stanford.edu/projects/glove/
    and follow the instructions there for downloading.

    Inputs
    ----------
    data = A pandas dataframe with a column of text we wish to embed
    target_col = The column within data we want to embed
    """
    assert target_col in data.columns

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data[target_col])

    max_length = max(len(data) for data in data[target_col])
    word_index = tokenizer.word_index
    vocab_size = len(word_index)    

    # padding text data
    sequences = tokenizer.texts_to_sequences(data[target_col])
    padded_seq = pad_sequences(sequences, maxlen=12630, padding='post', truncating='post')

    # create embedding index
    embedding_index = {}
    with open('glove.42B.300d.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    # create embedding matrix
    embedding_matrix = np.zeros((vocab_size+1, 300))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    if verbose:
        print(f'GloVe Embedded Lyrics: Shape = {embedding_matrix.shape}')

    return embedding_matrix