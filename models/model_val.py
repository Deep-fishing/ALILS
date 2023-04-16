import torch
import numpy as np


def val(model, val_iterator, criterion, device, metric):
    model.eval()
    loss_list, acc_list = [], []

    with torch.no_grad():
        for i, (batch, trg) in enumerate(val_iterator):
            batch = batch.to(device)
            trg = trg.to(device)

            output = model(batch)

            loss = criterion(output, trg)

            acc = metric(output, trg).mean()

            loss_list.append(loss.item())
            acc_list.append(acc.item())

        return np.mean(loss_list), np.mean(acc_list)
