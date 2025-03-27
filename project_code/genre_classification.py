# predict a user provided songs genre
import sys
if '..' not in sys.path:
    sys.path.append('..')

import numpy as np
import pandas as pd

from transformers import DistilBertTokenizer, DistilBertModel
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .data_gathering import clean_lyrics
from embedding import glove

# Load DistilBERT model and tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
bert_model = DistilBertModel.from_pretrained(model_name).to(device)

def distilerbert_clf_prediction(
    lyrics:str,
    clf_model,
    label_mapping,
    distilbert_model = bert_model, 
    tokenizer = tokenizer,
    device:str = 'cpu',
    verbose:bool = True
):
    """ 
    Description
    ----------
    This function takes the user specified set of custom lyrics, embeds them
    using the DistilBERT model, and passes those embeddings to the trained
    classification model to predict it's genre.

    Inputs
    ----------
    lyrics = A string of lyrics to be provided to the model.
    distilbert_model = The distilbert model we will use for embedding
    tokenizer = The distilbert tokenizer
    clf_model = A (trained) pytorch neural network
    label_mapping = A dict containing the mapping of idx to genre
    device = The device we are working on
    verbose = If true, prints the output

    Returns
    ----------
    pred_genre = A string denoting the predicted genre
    """
    # set the neural net to eval mode 
    clf_model = clf_model.eval()

    # clean user entered lyrics in a similar way as the model
    lyrics_clean = clean_lyrics(lyric = lyrics)

    # tokenize the lyrics
    inputs = tokenizer(lyrics_clean, return_tensors = 'pt', truncation = True, padding = 'max_length', max_length = 512)

    # move inputs to device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # pass the inputs through DistilBERT
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        embedded_lyrics = outputs.last_hidden_state[:, 0, :] # extract CLS token representation

    # predict the genre
    with torch.no_grad():
        logits = clf_model(embedded_lyrics) # pass throguh classifier
        probabilities = F.softmax(logits, dim = -1) # convert logits to probabilities
        pred_idx = torch.argmax(probabilities, dim = -1).item() # get highest prob idx

    # use the pred idx to map back to a genre
    pred_genre = label_mapping[pred_idx]

    if verbose:
        print(f'Lyrics:\n{lyrics}')
        print(f'\nPredicted Genre: {pred_genre} (idx = {pred_idx})')

    return pred_genre

def glove_clf_prediction(
    lyrics:str,
    clf_model,
    glove_index:dict,
    label_mapping:dict,
    max_seq_len:int,
    device:str = 'cpu',
    verbose:bool = True
):
    """ 
    Description
    ----------
    This function takes the user specified set of custom lyrics, embeds
    them using the GloVe model, and passes those embeddings to the trained 
    classification model to predict it's genre.

    Inputs
    ----------
    lyrics = A string of lyrics to be provided to the model.
    clf_model = A (trained) pytoch neural network
    glove_index = A dict containing the word embeddings from glove
    label_mapping = A dict containing the mapping of genre idx to genre
    max_seq_len = The max sequence length which the model is expecting
    device = The device where we are working
    verbose = If true, prints useful intermediates

    Returns
    ----------
    pred_genre = A string denoting the predicted genre
    """
    # set the neural net to eval mode
    clf_model = clf_model.eval()

    # transform the lyrics to lists of words in the same way 
    # as we did for glove training
    df = pd.DataFrame({'lyrics': [lyrics]})
    clean_lyrics = glove.clean_and_tokenize_lyrics(data = df, col = 'lyrics')[0]

    # transform tokens into glove matrix

    ## instantiate the matrix
    embed = np.zeros((max_seq_len, 300))

    ## iteratively update the matrix
    for i in range(len(clean_lyrics)):
        if i < max_seq_len:
            # extract the word at the list position
            word = clean_lyrics[i]

            # find the glove embedding for the word
            word_vec = glove_index[word]

            # update embed with the word vec
            embed[i, :] = word_vec

    # reshape the embed to the expected shape
    embed = torch.tensor(embed, dtype = torch.float32)
    embed = embed.unsqueeze(0)

    # use the matrix as an input to the model
    with torch.no_grad():
        logits = clf_model(embed)
        probabilities = F.softmax(logits, dim = -1)
        pred_idx = torch.argmax(probabilities, dim = -1).item()

    # use the pred idx to map back to a genre
    pred_genre = label_mapping[pred_idx]

    if verbose:
        print(f'Lyrics:\n{lyrics}')
        print(f'\nPredicted Genre: {pred_genre} (idx = {pred_idx})')

    return pred_genre