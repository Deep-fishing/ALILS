
import numpy as np


def train(model, train_iterator, optimizer, criterion, device, metric):
    model.train()
    loss_list, acc_list = [], []

    for i, (batch, trg) in enumerate(train_iterator):

        batch = batch.to(device)
        trg = trg.to(device)

        output = model(batch)

        loss = criterion(output, trg)
        loss.backward()

        acc = metric(output, trg).mean()

        optimizer.step()
        optimizer.zero_grad()

        loss_list.append(loss.item())
        acc_list.append(acc.item())

    return np.mean(loss_list), np.mean(acc_list)
