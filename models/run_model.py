from .model_test import test
from .model_val import val
from .model_train import train
import torch
from tqdm import tqdm


def run(train_iterator, val_iterator, test_iterator, dataset_name,
        model, criterion, optimizer,
        epochs, device, save_path, seed, sub,
        early_stop, patience):

    best_val, best_epoch, count = 0, 0, 0

    progress_bar = tqdm(range(epochs))
    for epoch in progress_bar:
        progress_bar.set_description('Dataset seed ' + str(seed))

        tran_loss, train_acc = train(model=model,
                                     train_iterator=train_iterator,
                                     optimizer=optimizer,
                                     criterion=criterion,
                                     device=device,
                                     metric=metric)

        val_loss, val_acc = val(model=model,
                                val_iterator=val_iterator,
                                criterion=criterion,
                                device=device,
                                metric=metric)

        if val_acc >= best_val:
            best_val = val_acc
            best_epoch = save_checkpoint(save_path=save_path, model=model, epoch=epoch, best_val=best_val)
            count = 0
        else:
            count += 1

        if early_stop and count >= patience and epoch > 0.5 * epochs:
            print(f'Early stopping in epoch of {epoch}')
            break
        progress_bar.set_postfix(val_acc='%.3f' % val_acc, val_loss='%.3f' % val_loss)

    print(f'\t\tSave the best model on {best_epoch} epoch with {best_val}')

    f_weighted, f_macro, mcc, zl = test(model=model,
                                        test_iterator=test_iterator,
                                        model_path=save_path,
                                        device=device,
                                        dataset_name=dataset_name,
                                        sub=sub)

    print('\t\tf_weighted: {}'.format(f_weighted))
    print('\t\tf_macro:    {}'.format(f_macro))
    print('\t\tmcc:        {}'.format(mcc))
    print('\t\tzl:         {}'.format(zl))
    return f_weighted, f_macro, mcc, zl


def metric(pre: torch.Tensor, trg: torch.Tensor):

    pre = torch.softmax(pre, 1)
    _, pre = torch.max(pre.data, 1)
    return (pre == trg.data).float()


def save_checkpoint(save_path=None, model=None, epoch=None, best_val=None):

    states = {
        'epoch': epoch + 1,
        'best_val': best_val,
        'state_dict': model.state_dict()
    }

    torch.save(states, save_path)

    return epoch
