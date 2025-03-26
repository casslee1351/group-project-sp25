from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import DataLoader, Dataset 
import numpy as np
from tqdm import tqdm

def distilbert_embed_doc(doc):
    """
    Description
    ----------
    Uses the DistilBert model to embed the specified document.
    Intended to be used within the apply_distilbert_embedding 
    wrapper function.

    Inputs
    ---------
    doc = The text document we want to embed

    Returns
    ----------

    """
    # retrieve objects from transformers
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # extract tokens
    tokens = tokenizer(
        doc,
        truncation = True,
        padding = True, 
        max_length = 512, #DistilBERT has a max token limit of 512
        return_tensors = "pt" #converting output as PyTorch since DistilBERT expects tensors not token IDs
    )
    
    # removes gradient calculation to save memory usage since inferences are 
    # not needed as we are predicting, not training
    with torch.no_grad(): 
        output = model(**tokens)

    cls_embedding = output.last_hidden_state[:, 0] #extract first token with CLS
    cls_embedding = cls_embedding.detach() #detach from PyTorch's gradient computation 
    cls_embedding = cls_embedding.cpu() #converting tensor to CPU to ensure compatability (ie. NumPy array conversion)
    cls_embedding = cls_embedding.squeeze() #remove any extra dimensions
    
    embedding = cls_embedding.numpy() #converting into NumPy array
    return embedding

def distilbert_embed_all_docs(data, target_col:str, verbose:bool = True):
    """
    Description
    ----------
    Applies the distilbert_embed_doc function to each document
    in the dataset.

    Inputs
    ------------
    data = A pandas dataframe containing the text data we want to embed
    target_col = The column within data we want to embed
    verbose = If true, prints useful intermediates

    Returns
    ----------

    """
    assert target_col in data.columns

    output = [distilbert_embed_doc(doc) for doc in tqdm(data[target_col])]
    output = np.array(output) # convert to numpy array

    if verbose:
        print(f'DistilBert Embedded Lyrics: Shape = {output.shape}')

    return output

class LyricsDataset(Dataset):
    def __init__(self, lyrics):
        self.lyrics = lyrics

    def __len__(self):
        return len(self.lyrics)

    def __getitem__(self, idx):
        return self.lyrics[idx]

def embed_all_lyrics_v2(
    data, target_col:str,
    batch_size:int = 32,
    device:str = 'cpu',
    verbose:bool = True
):
    """
    Description
    ----------
    This function applies the DistilBert model to embed all lyrics
    in our dataset. It does so in mini-batches to speed up the 
    embedding process without crashing our machine.

    NOTE: We are choosing to extract the CLS Token summary from the last 
    hidden state. We could instead extract the entire sequence of token
    embeddings. In which case the output shape would instead be 
    [n_songs, max_sequence_length, 768].

    Inputs
    ----------
    data = A pandas dataframe containing the lyrics we want to embed
    target_col = A column within `data` which has the lyrics.
    device = The device we are training on

    Returns
    ----------
    embeddings = A torch tensor containing the embedded lyrics with 
        shape [n_songs, 768]..
    """
    assert target_col in data.columns
    assert device in ['cpu', 'cuda']

    # Load DistilBERT model and tokenizer
    model_name = 'distilbert-base-uncased'
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    bert_model = DistilBertModel.from_pretrained(model_name)

    # move model to device
    bert_model.to(device)
    bert_model.eval() # disables dropout

    # helper function
    def collate_fn(batch):
        """Tokenize and pad batch"""
        encoded = tokenizer(batch, padding = True, truncation = True, max_length = 512, return_tensors = 'pt').to(device)
        output = {key: val.to(device) for key, val in encoded.items()}
        return output 
    
    # convert lyrics column to list
    lyrics_list = data[target_col].tolist()

    # create DataLoader for batching
    lyrics_dataset = LyricsDataset(lyrics_list)
    lyrics_loader = DataLoader(lyrics_dataset, batch_size = batch_size, collate_fn = collate_fn)

    # instantiate object to store embeddings
    embeddings = []

    # iterate through batches to generate embeddings
    with torch.no_grad():
        for batch in tqdm(lyrics_loader):
            outputs = bert_model(**batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :] # extract CLS token
            embeddings.append(cls_embeddings)

    # convert to tensor
    embeddings = torch.cat(embeddings, dim = 0)

    if verbose:
        print(f'DistilBERT Embedded Lyrics: {embeddings.shape}')

    return embeddings

# =================
# === GRAVEYARD ===
# =================

# def embed_all_docs_v2(
#     data, target_col:str, 
#     max_length:int = 512, use_cls:bool = True,
#     verbose:bool = True
# ):
#     """
#     Description
#     ----------
#     This function applies the DistilBert embedding to all records in the 
#     `target_col` in `data`.

#     Inputs
#     ----------
#     data = A pandas dataframe containing the text we want to embed
#         via DistilBert  
#     target_col = The column we wish to embed  
#     max_length = The max token length for DistilBert  
#     use_cls = If true, extracts the [CLS] token embedding, o'wise uses mean 
#         pooling  
#     verbose = If true, prints useful intermediates  
    
#     Returns
#     ----------
#     output = A numpy array of the embedded records
#     """
#     # retrieve objects from transformers
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#     model = DistilBertModel.from_pretrained('distilbert-base-uncased')
#     model.eval() # no grad computation

#     # tokenize records
#     tokens = tokenizer(
#         data[target_col].tolist(),
#         padding = True,
#         truncation = True,
#         max_length = max_length,
#         return_tensors = 'pt'
#     )

#     # extract last hidden state from output
#     with torch.no_grad():
#         outputs = model(**tokens)

#     last_hidden = outputs.last_hidden_state # (batch_size, seq_len, 768)

#     # create embeddings based on strategy selected
#     if use_cls:
#         # extract the [CLS] toen embedding (first token)
#         embeddings = last_hidden[:, 0, :] # (batch_size, 768)
#     else:
#         # mean pooling across all tokens (excluding padding tokens)
#         attention_mask = tokens['attention_mask'].unsqueeze(-1)
#         embeddings = (last_hidden @ attention_mask).sum(dim = 1) / attention_mask.sum(dim = 1)

#     if verbose:
#         print(f'DistilBERT Embedded Lyrics: {embeddings.numpy().shape}')

#     return embeddings.numpy()