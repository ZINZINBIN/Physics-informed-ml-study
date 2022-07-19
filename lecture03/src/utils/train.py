from tqdm import tqdm
from typing import Optional
import torch
import torch.nn

def train_per_epoch(
    model : torch.nn.Module, 
    dataloader : torch.utils.data.DataLoader, 
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module, 
    device : str = "cpu"):

    train_loss = 0

    for idx, batch in enumerate(dataloader):

        optimizer.zero_grad()

        batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        batch_output = model(batch_x)
        batch_loss = loss_fn(batch_output, batch_y)

        batch_loss.backward()
        optimizer.step()
        
        train_loss += batch_loss.detach().cpu().numpy()

    train_loss /= idx + 1

    return train_loss
    
def valid_per_epoch(
    model : torch.nn.Module, 
    dataloader : torch.utils.data.DataLoader, 
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module, 
    device : str = "cpu"
    ):

    valid_loss = 0

    for idx, batch in enumerate(dataloader):
        with torch.no_grad():
            optimizer.zero_grad()
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_output = model(batch_x)
            batch_loss = loss_fn(batch_output, batch_y)
            valid_loss += batch_loss.detach().cpu().numpy()

    valid_loss /= idx + 1

    return valid_loss

def train(
    model : torch.nn.Module, 
    dataloader : torch.utils.data.DataLoader, 
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module, 
    num_epochs : int = 42,
    device : str = "cpu"
    ):

    train_loss_list = []

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_per_epoch(
            model,
            dataloader,
            optimizer,
            loss_fn,
            device
        )

        train_loss_list.append(train_loss)

    return train_loss_list