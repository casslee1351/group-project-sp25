import numpy as np

import string 
from gensim.parsing.preprocessing import remove_stopwords
import nltk 
from nltk.tokenize import word_tokenize
import re

import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm


def remove_between_brackets(text):
  """Removes all text between any matching pair of brackets, including the brackets themselves."""
  return re.sub(r'\[.*?\]', '', text)

def remove_punctuation(text):
    """Removes punctuation from a string."""
    return text.translate(str.maketrans('', '', string.punctuation))

def clean_and_tokenize_lyrics(data, col):

    df = data.copy()

    df[f'clean_{col}'] = df[col].apply(remove_between_brackets)
    df[f'clean_{col}'] = df[f'clean_{col}'].apply(remove_punctuation)
    df[f'clean_{col}'] = df[f'clean_{col}'].str.lower()
    df[f'clean_{col}'] = df[f'clean_{col}'].str.replace(",", '')
    df[f'clean_{col}'] = df[f'clean_{col}'].apply(remove_stopwords)
    df[f'tokenized_{col}'] = df[f'clean_{col}'].apply(word_tokenize)

    return df[f'tokenized_{col}']

def embed_all_lyrics(
    data, target_col:str,
    custom_max_seq_len = None,
    verbose:bool = True
):
    """
    Description
    ----------
    This function iterates through the provided `target_col` in `data` to apply
    the GloVe embedding.

    Inputs
    ----------
    data = A pandas dataframe containing the `target_col` we want to embed.
    target_col = A column within `data` we want to apply the GloVe embedding to
    custom_max_seq_len = If not none, this should specify how many of the first 
        n words we want to embed via GloVe.
    verbose = If true, prints useful intermediates

    Returns
    ----------
    embeddings = A torch tensor containing the embedded lyrics
        Shape is (n_songs, max_seq_len, 300)
    embedding_index = A dict containing the arrays of GloVe embeddings for words
    """
    assert target_col in data.columns

    # perform cleaning and tokenization
    lyrics = clean_and_tokenize_lyrics(data = data, col = target_col)

    # fit tokenizer to text
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lyrics)

    # define maxl length and use that to pad text
    ## word_index: ex = {'im': 1, 'dont': 2, 'like': 3, know': 4, ... }
    ## padded_seq: shape = (n_songs, max_seq_len)
    if custom_max_seq_len is not None:
        max_length = custom_max_seq_len
    else:
        max_length = max(len(song) for song in lyrics)
    word_index = tokenizer.word_index
    word_index_inv = {v: k for k, v in word_index.items()}
    vocab_size = len(word_index)
    sequences = tokenizer.texts_to_sequences(lyrics)
    padded_seq = pad_sequences(sequences, maxlen = max_length, padding = 'post', truncating = 'post')

    # create embedding index
    # embedding_index: for each word, value shape = (n_songs, )
    # example: {'example': array([-0.13427  ,  0.034675 , ... , -0.0084217,  0.064313])}
    embedding_index = {}
    if verbose:
        print('Extracting GloVe Embedding Index...')
    with open('glove.42B.300d.txt', encoding = 'utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embedding_index[word] = coefs

    # create embedding matrix
    embeddings = []

    ## for each song...
    if verbose:
        print('\nConverting Word Indices to GloVe Vectors...')
    for i in tqdm(range(len(padded_seq))):
        song = padded_seq[i, :] # ex: [2221 1063 1388 ...    0    0    0]
        all_word_vecs = []
        ### for each word in this song...
        for word_idx in song:
            #### extract word and use that to retrieve GloVe embedding
            try:
                word = word_index_inv[word_idx] # get the word
                word_vec = embedding_index[word] # get the glove embedding for the word
            except:
                # word_idx not in word_index (i.e., it's a padding token)
                word_vec = np.zeros(300, )

            #### store word vec to word_vecs
            all_word_vecs.append(word_vec)
        
        ### store the all_word_vecs to embeddings
        embeddings.append(all_word_vecs)

    # convert embeddings to torch tensor
    embeddings = np.array(embeddings) # first convert to np array to speed up tensor conversion.
    embeddings = torch.tensor(embeddings, dtype = torch.float32)        

    if verbose:
        print(f'\nGloVe Embedded {target_col.title()}: Shape = (n_songs, max_seq_len, embed_len) = {embeddings.shape}')
        print(f'\tPadded Sequences: Shape = (n_songs, max_seq_len) = {padded_seq.shape}')

    return embeddings, embedding_index

def read_glove_embedding_index(verbose:bool = True):
    """
    Description
    -----------
    If we are not training the models, we still need to retireve the Glove
    Embedding Index. This function performs exactly that function.

    Inputs
    ----------
    verbose = If true, prints useful intermediates

    Returns
    ----------
    embedding_index = A dict containing the GloVe embedding dict
    """
    # create embedding index
    # embedding_index: for each word, value shape = (n_songs, )
    # example: {'example': array([-0.13427  ,  0.034675 , ... , -0.0084217,  0.064313])}
    embedding_index = {}
    if verbose:
        print('Extracting GloVe Embedding Index...')
    with open('glove.42B.300d.txt', encoding = 'utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype = 'float32')
            embedding_index[word] = coefs

    return embedding_index

# =================
# === Graveyard ===
# =================

# def apply_glove(sentences, model="glove-wiki-gigaword-100"):

#     # # initialize glove model
#     # glove_model = api.load("glove-wiki-gigaword-100")

#     print("Models available for use:")
#     print(list(gensim.downloader.info()['models'].keys()))

#     glove_model = api.load(model)

#     ### initialize model
#     base_model = Word2Vec(vector_size=100, min_count=1)
#     base_model.build_vocab(sentences)
#     total_examples = base_model.corpus_count

#     base_model.build_vocab(glove_model.index_to_key, update=True)
#     base_model.train(sentences, total_examples=total_examples, epochs=base_model.epochs)

#     return base_model

# def create_glove_matrix(data, target_col, verbose:bool = True):
#     """
#     Description
#     ----------
#     TODO

#     If glove.42B.300d.txt is not in the project directory, please navigate to
#     https://nlp.stanford.edu/projects/glove/
#     and follow the instructions there for downloading.

#     Inputs
#     ----------
#     data = A pandas dataframe with a column of text we wish to embed
#     target_col = The column within data we want to embed
#     """
#     assert target_col in data.columns

#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts(data[target_col])

#     max_length = max(len(data) for data in data[target_col])
#     word_index = tokenizer.word_index
#     vocab_size = len(word_index)    

#     # padding text data
#     sequences = tokenizer.texts_to_sequences(data[target_col])
#     padded_seq = pad_sequences(sequences, maxlen=12630, padding='post', truncating='post')

#     # create embedding index
#     embedding_index = {}
#     with open('glove.42B.300d.txt', encoding='utf-8') as f:
#         for line in f:
#             values = line.split()
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             embedding_index[word] = coefs

#     # create embedding matrix
#     embedding_matrix = np.zeros((vocab_size+1, 300))
#     for word, i in word_index.items():
#         embedding_vector = embedding_index.get(word)
#         if embedding_vector is not None:
#             embedding_matrix[i] = embedding_vector

#     if verbose:
#         print(f'GloVe Embedded Lyrics: Shape = {embedding_matrix.shape}')

#     return embedding_matrix