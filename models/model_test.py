import torch
import os
from sklearn.metrics import f1_score, matthews_corrcoef, zero_one_loss


def test(model, test_iterator, device, model_path, dataset_name, sub):
    model.eval()
    print('\nTesting model on {} dataset......'.format(dataset_name+sub))

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path)['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(model_path))

    num_elements = len(test_iterator.dataset)
    batch_size = test_iterator.batch_size
    predictions = torch.zeros(num_elements, 1)
    true_labels = torch.zeros(num_elements, 1)

    with torch.no_grad():
        for i, (batch, trg) in enumerate(test_iterator):
            batch = batch.to(device)

            trg = trg.to(device)

            output = model(batch)

            start = i * batch_size
            end = start + batch_size

            pre = torch.softmax(output, 1)
            prob, pre = torch.max(pre.data, 1)

            predictions[start:end] = pre.unsqueeze(1)
            true_labels[start:end] = trg.unsqueeze(1).long()

    f_weighted = f1_score(true_labels, predictions, average='weighted')
    f_macro = f1_score(true_labels, predictions,  average='macro')
    mcc = matthews_corrcoef(true_labels, predictions)
    zl = zero_one_loss(true_labels, predictions)*100

    return f_weighted, f_macro, mcc, zl
