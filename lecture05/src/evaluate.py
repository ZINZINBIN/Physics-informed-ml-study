from typing import Optional
import torch
import torch.nn

def evaluate(
    model : torch.nn.Module, 
    dataloader : torch.utils.data.DataLoader, 
    optimizer : torch.optim.Optimizer,
    loss_fn : torch.nn.Module, 
    device : str = "cpu"
    ):

    test_loss = 0
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
            test_loss += batch_loss.detach().cpu().numpy()

    test_loss /= idx + 1

    return test_loss