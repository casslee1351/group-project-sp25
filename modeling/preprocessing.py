import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import pandas as pd
import numpy as np

from transformers import GPT2Tokenizer
from datasets import Dataset

def create_datasets(
    data_embed, labels,
    label_mapping:dict,
    val_pct:float = 0.15,
    batch_size:int = 32,
    verbose:bool = True
):
    """
    Description
    ----------
    This function takes the embedded data and splits it into a training,
    validation, and testing sets. It then loads it into a DataLoader 
    before entering the training loop.

    Inputs
    ---------
    data_embed = A numpy array of the data which has already been embedded
    labels = A pandas series containing the data labels
    label_mapping = A dict containing the manner by which to translate str
        genres to idx
    val_pct = The percent of data we want separated into the val and test
        sets. NOTE that we are assuming a (1 - 2x) / (x) / (x) split.
    batch_size = The size of batches we want for the data loader
    verbose = If true, prints useful intermediates

    Returns
    ----------
    train_loader = A DataLoader object containing the training data
    val_loader = A DataLoader object containing the validation data
    test_loader = The dataset for the test items we want to evaluate
        the fully trained model against
    """
    assert type(labels) == pd.core.series.Series

    # convert data_embed and labels to torch tensor
    X_tensor = data_embed.clone().detach() # torch.tensor(data_embed)
    genre_map_flip = {v: k for k, v in label_mapping.items()}
    y_tensor = torch.tensor(labels.map(genre_map_flip), dtype = torch.long)
    # y_tensor = torch.tensor(labels.astype('category').cat.codes.values, dtype = torch.long)

    # create tensor dataset of X and y, then perform train-val-test split
    dataset = TensorDataset(X_tensor, y_tensor)
    splits = [(1 - (val_pct * 2)), val_pct, val_pct]
    train_dataset, val_dataset, test_dataset = random_split(dataset, splits)

    # load datasets into Loaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True)

    if verbose:
        # x_batch, y_batch = next(iter(train_loader))
        print(f'Train: {len(train_loader)} Batches of Size {batch_size} For Training')
        print(f'Val: {len(val_loader)} Batches of Size {batch_size} For Training')
        print(f'Test: {len(test_loader)} Batches of Size {batch_size} For Final Eval')

    return train_loader, val_loader, test_loader

def gpt2_create_datasets(
    data,
    label_mapping:dict,
    input_col:str, label_col:str,
    val_pct:float = 0.15,
    verbose:bool = True
):
    """ 
    Description
    ----------
    This function performs the preprocessing expected for the GPT2 
    Fine Tuning process.

    Inputs
    ---------
    data = The complete dataset containing our lyrics and genres
    label_mapping = A dict containing the label mapping we wish to
        use to translate `label_col` to numbers.
    input_col = The column from `data` we wish to encode for inputs
    label_col = The column from `data` we wish to encore as labels
    val_pct = The percent of data we want to withhold for validation
        *and* testing
    verbose = If true, prints useful intermediates

    Returns
    ----------
    train_dataset, val_dataset, test_dataset = The dataset objects used for
        fine tuning the GPT2 model from hugging face.
    tokenizer = The tokenizer used on the inputs. We export it here to 
        ensure it can be used again for training
    """
    # prepare df for downstream processing
    # X = np.array(data[input_col])
    inv_map = {v: k for k, v in label_mapping.items()}
    y = np.array(data[label_col].map(inv_map))
    df = data.copy().rename(columns = {'genre': 'label'})
    df['label'] = y

    # initialize input tokenizer
    # This is used to convert the text data(lyrics) to a format understandable by GPT-2 model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # tokenize the inputs and load into dataset
    def tokenize_fx(ex):
        output = tokenizer(ex[input_col], padding = 'max_length', truncation = True, max_length = 512)
        return output
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenize_fx, batched = True)

    # train val test split
    train_test_split1 = dataset.train_test_split(test_size = (2 * val_pct))
    train_dataset = train_test_split1['train']
    train_test_split2 = train_test_split1['test'].train_test_split(test_size = 0.5)
    val_dataset = train_test_split2['train']
    test_dataset = train_test_split2['test']

    if verbose:
        print(f'Train: Length = {len(train_dataset)}')
        print(f'Val:   Length = {len(val_dataset)}')
        print(f'Test:  Length = {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset, tokenizer