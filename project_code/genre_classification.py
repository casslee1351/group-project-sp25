# predict a user provided songs genre
from transformers import DistilBertTokenizer, DistilBertModel
import torch 
import torch.nn as nn
import torch.nn.functional as F

from .data_gathering import clean_lyrics

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