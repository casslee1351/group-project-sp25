import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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