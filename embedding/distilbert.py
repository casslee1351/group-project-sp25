from transformers import DistilBertTokenizer, DistilBertModel
import torch
import numpy as np

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

    output = [distilbert_embed_doc(doc) for doc in data[target_col]]
    output = np.array(output) # convert to numpy array

    if verbose:
        print(f'DistilBert Embedded Lyrics: Shape = {output.shape}')

    return output