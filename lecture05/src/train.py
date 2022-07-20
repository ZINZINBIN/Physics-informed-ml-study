from tqdm.auto import tqdm
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
    model.train()
    model.to(device)

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
    model.to(device)
    model.eval()

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
    train_loader : torch.utils.data.DataLoader, 
    valid_loader : torch.utils.data.DataLoader,
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module, 
    num_epochs : int = 42,
    device : str = "cpu",
    verbose : int = 1,
    ):

    train_loss_list = []
    valid_loss_list = []

    for epoch in tqdm(range(num_epochs)):
        train_loss = train_per_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )

        valid_loss = valid_per_epoch(
            model,
            valid_loader,
            optimizer,
            loss_fn,
            device
        )

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if epoch % verbose == 0:
            print("Epoch : {}, train_loss : {:.3f}, valid_loss : {:.3f}".format(epoch+1, train_loss, valid_loss))

    return train_loss_list, valid_loss_list