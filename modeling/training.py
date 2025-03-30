import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformers import GPT2ForSequenceClassification, Trainer, TrainingArguments
from datetime import datetime

def nn_training(
    model,
    train_loader, val_loader,
    embed_strategy:str,
    learning_rate:float = 0.001,
    num_epochs:int = 100, patience:int = 5,
    device:str = 'cpu',
    verbose:bool = True, print_every:int = 5
):
    """
    Description
    ----------
    This function takes a specified model and trains it on the train_loader,
    evaluates the epoch round on the val_loader, and allows for early stopping
    if there is no improvement against the val set after `patience` epochs.

    Inputs
    ----------
    model = An instantiated torch neural network model that we want to train
    train_loader = The DataLoader object containing our training set
    val_loader = The DataLoader object containing our validation set
    embed_strategy = The method by which we embedded the songs
    learning_rate = The rate at which we want to learn
    num_epochs = The (max) number of epochs we want to train for
    patience = The number of epochs beyond the current best epoch that we
        keep training before stopping early
    device = The device we are training on
    verbose = If true, prints useful intermediates
    print_every = If verbose, this defines how frequently we print out
        model performance results

    Returns
    ----------
    None, but the best model weights and biases will be stored in the models folder
    """
    # set up training objects
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # instantiate constants
    best_val_loss = float('inf')
    best_epoch = 0
    waiting = 0

    # train the model
    for epoch in range(num_epochs):       

        # === training loop ===
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader):

            ## store batch to device
            X, y = X.to(device), y.to(device)

            ## forward pass
            optimizer.zero_grad()
            y_pred = model(X)

            ## backward pass
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            ## store loss from batch
            train_loss += loss.item()

        # === validation loop ===
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:

                ## store batch to device
                X, y = X.to(device), y.to(device)

                ## forward pass
                y_pred = model(X)

                ## store loss from batch
                loss = criterion(y_pred, y)
                val_loss += loss.item()

        # epoch wrap up

        ## take average of loss
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        ## check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            waiting = 0
            torch.save(model.state_dict(), f'models/{embed_strategy}_{model.__class__.__name__}_Trained.pth')
            if verbose and (epoch + 1) % print_every == 0:
                print(
                    f'[{epoch + 1} / {num_epochs}] '
                    f'Train Loss = {train_loss:.4f}, '
                    f'Val Loss = {val_loss:.4f} '
                    f'**New Best Model**'
                )
        else:
            waiting += 1
            if verbose and (epoch + 1) % print_every == 0:
                    print(
                        f'[{epoch + 1} / {num_epochs}] '
                        f'Train Loss = {train_loss:.4f}, '
                        f'Val Loss = {val_loss:.4f}'
                    )

        if waiting >= patience:
            print(f'Early Stopping Triggered. Training Stopped.')
            print(f'\tBest Epoch = {best_epoch}, Best Val Loss = {best_val_loss}')
            break

def evaluate_nn_model_against_test_set(
    model, test_dataset,
    device:str = 'cpu',
    verbose:bool = True
):
    """
    Description
    ----------
    This function uses the provided neural network `model` to predict
    the song genre of the provided lyrics in `input_lyrics` and compares
    them against the `target_genres`

    Inputs
    ----------
    model = The neural network we want to use to generate predictions
    test_dataset = The withheld test set for evaluation
    device = The device we are working on

    Returns
    ----------

    """
    assert device in ['cpu', 'cuda']

    # iterate through the test set to generate predictions
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in test_dataset:
            
            ## store tensors to device
            X, y = X.to(device), y.to(device)

            ## use model to predict y
            outputs = model(X)
            y_pred = torch.argmax(outputs, dim = -1)

            ## cumulatively calc accuracy
            correct += (y_pred == y).sum().item()
            total += y.size(0)

    accuracy = correct / total 

    if verbose:
        print(f'Model Accuracy = {100 * accuracy:.2f}%')

    return accuracy

def gpt2_fine_tuning(
    train_dataset, val_dataset, test_dataset,
    input_tokenizer,
    num_labels:int,
    batch_size:int = 4,
    num_epochs:int = 5,
    learning_rate:float = 0.001,
    save_model:bool = True,
    verbose:bool = True
):
    """ 
    Description
    ----------
    
    Inputs
    ----------
    train_dataset = The dataset we want to fine tune the GPT model
        against.
    val_dataset = The dataset we want to use to evaluated the GPT 
        model fine tuning process after each epoch
    test_dataset = The dataset we want to use to test the performance
        of the fine tuned GPT2 model after training is complete.
    input_tokenizer = The tokenizer used to initially tokenize the 
        inputs.
    num_labels = The number of unique values in our classifier model
    batch_size = The size of batches we want for training
    num_epochs = The number of epochs we wish to train for
    learning_rate = The rate at which we want to learn
    save_model = If true, saves the fine tuned model
    verbose = If true, prints useful intermediates

    Returns
    ----------

    """
    # load GPT-2 model with classification head
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels = num_labels)
    model.config.pad_token_id = model.config.eos_token_id # ensure padding works correctly

    if verbose:
        print(model)

    # define training arguments
    training_args = TrainingArguments(
        output_dir = './results',
        eval_strategy = 'epoch',
        save_strategy = 'epoch',
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = num_epochs,
        learning_rate = learning_rate,
        weight_decay = 0.01,
        logging_dir = './logs',
        logging_steps = 10,
        load_best_model_at_end = True
    )

    # initialize Trainer
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        processing_class = input_tokenizer 
    )

    # train the model
    if verbose:
        train_start = datetime.now()
        print(f'GPT2 Fine Tuning: Start Time = {train_start.strftime("%Y-%m-%d %H:%M:%S")}')

    trainer.train()

    if verbose:
        train_end = datetime.now()
        train_time = (train_end - train_start).seconds
        print(f'GPT2 Fine Tuning: Trained. Check the models subfolder for the trained model.')
        print(f'GPT2 Fine Tuning: End Time = {train_start.strftime("%Y-%m-%d %H:%M:%S")}, Duration = {train_time / 60:.2f}min')


    # save the trained model
    model.save_pretrained('models/GPT2_FineTune_Trained')
    input_tokenizer.save_pretrained('tokenizers/GPT2_FineTune_Trained')

    # evaluate model performance
    test_eval = trainer.evaluate(test_dataset)

    if verbose:
        print(f'GPT2 Fine Tuning: Test Performance...')
        for k, v in test_eval.items():
            print(f'\t{k}: {v}')

    return model