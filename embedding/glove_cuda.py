import numpy as np
import string
import re
from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def remove_between_brackets(text):
    """Removes all text between any matching pair of brackets, including the brackets themselves."""
    return re.sub(r'\[.*?\]', '', text)


def remove_punctuation(text):
    """Removes punctuation from a string."""
    return text.translate(str.maketrans('', '', string.punctuation))


def clean_and_tokenize_lyrics(data, col):
    """
    Cleans and tokenizes lyrics from a dataframe column.
    """
    df = data.copy()
    df[f'clean_{col}'] = df[col].apply(remove_between_brackets)
    df[f'clean_{col}'] = df[f'clean_{col}'].apply(remove_punctuation)
    df[f'clean_{col}'] = df[f'clean_{col}'].str.lower()
    df[f'clean_{col}'] = df[f'clean_{col}'].str.replace(",", '')
    df[f'clean_{col}'] = df[f'clean_{col}'].apply(remove_stopwords)
    df[f'tokenized_{col}'] = df[f'clean_{col}'].apply(word_tokenize)
    return df[f'tokenized_{col}']


def load_glove_embeddings(file_path, verbose=True):
    """
    Loads GloVe embeddings from a file into a dictionary.
    """
    embeddings_index = {}
    if verbose:
        print('Extracting GloVe Embedding Index...')
    with open(file_path, encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


def embed_all_lyrics(data, target_col: str, glove_path='glove.42B.300d.txt',
                     custom_max_seq_len=None, verbose=True):
    """
    Embeds lyrics using pre-trained GloVe embeddings in a GPU-accelerated manner.
    
    Steps:
      1. Clean and tokenize the lyrics.
      2. Tokenize and pad the sequences.
      3. Build an embedding matrix from the GloVe file.
      4. Create a PyTorch nn.Embedding layer with the pre-loaded weights.
      5. Perform the embedding lookup on the GPU.
      
    Returns:
      - embedded: A torch tensor of shape (n_songs, max_seq_len, 300)
      - embedding_index: The dictionary of GloVe embeddings
    """
    # 1. Clean and tokenize lyrics
    lyrics = clean_and_tokenize_lyrics(data=data, col=target_col)

    # 2. Tokenizer and pad sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lyrics)
    sequences = tokenizer.texts_to_sequences(lyrics)
    if custom_max_seq_len is not None:
        max_length = custom_max_seq_len
    else:
        max_length = max(len(seq) for seq in sequences)
    padded_seq = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    
    # Build word-index mapping and determine vocab size.
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1  # +1 for the padding token

    # 3. Load GloVe embeddings and create embedding matrix
    embedding_dim = 300
    glove_embeddings = load_glove_embeddings(glove_path, verbose=verbose)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        # else, row stays as zeros

    # 4. Create PyTorch embedding layer with pre-loaded weights
    embedding_layer = nn.Embedding(num_embeddings=vocab_size,
                                   embedding_dim=embedding_dim,
                                   padding_idx=0)
    embedding_layer.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
    # Optionally, freeze the weights if you don't plan to fine-tune:
    embedding_layer.weight.requires_grad = False

    # Move the layer to GPU
    embedding_layer = embedding_layer.to(DEVICE)

    # 5. Convert padded sequences to a tensor and move to GPU
    padded_seq_tensor = torch.tensor(padded_seq, dtype=torch.long).to(DEVICE)

    # Perform the embedding lookup on the GPU in a vectorized manner.
    if verbose:
        print("Performing GPU-accelerated embedding lookup...")
    embedded = embedding_layer(padded_seq_tensor)  # Shape: (n_songs, max_seq_len, embedding_dim)

    if verbose:
        print(f'\nGloVe Embedded {target_col.title()}: Shape = {embedded.shape}')
        print(f'\tPadded Sequences: Shape = {padded_seq_tensor.shape}')

    return embedded, glove_embeddings


def read_glove_embedding_index(glove_path='glove.42B.300d.txt', verbose=True):
    """
    Retrieves the full GloVe Embedding Index without embedding the lyrics.
    """
    return load_glove_embeddings(glove_path, verbose=verbose)
