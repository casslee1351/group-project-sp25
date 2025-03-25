from transformers import DistilBertTokenizer, DistilBertModel
import torch
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

def embed_all_docs_v2(
    data, target_col:str, 
    max_length:int = 512, use_cls:bool = True,
    verbose:bool = True
):
    """
    Description
    ----------
    This function applies the DistilBert embedding to all records in the 
    `target_col` in `data`.

    Inputs
    ----------
    data = A pandas dataframe containing the text we want to embed
        via DistilBert  
    target_col = The column we wish to embed  
    max_length = The max token length for DistilBert  
    use_cls = If true, extracts the [CLS] token embedding, o'wise uses mean 
        pooling  
    verbose = If true, prints useful intermediates  
    
    Returns
    ----------
    output = A numpy array of the embedded records
    """
    # retrieve objects from transformers
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.eval() # no grad computation

    # tokenize records
    tokens = tokenizer(
        data[target_col].tolist(),
        padding = True,
        truncation = True,
        max_length = max_length,
        return_tensors = 'pt'
    )

    # extract last hidden state from output
    with torch.no_grad():
        outputs = model(**tokens)

    last_hidden = outputs.last_hidden_state # (batch_size, seq_len, 768)

    # create embeddings based on strategy selected
    if use_cls:
        # extract the [CLS] toen embedding (first token)
        embeddings = last_hidden[:, 0, :] # (batch_size, 768)
    else:
        # mean pooling across all tokens (excluding padding tokens)
        attention_mask = tokens['attention_mask'].unsqueeze(-1)
        embeddings = (last_hidden @ attention_mask).sum(dim = 1) / attention_mask.sum(dim = 1)

    if verbose:
        print(f'DistilBERT Embedded Lyrics: {embeddings.numpy().shape}')

    return embeddings.numpy()