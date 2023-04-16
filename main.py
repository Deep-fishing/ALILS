import torch
from dataset.dataloder import load_data, get_iterator
from opts import get_arguments
from sklearn.model_selection import StratifiedKFold
from models.model_builder import get_model
from models.run_model import run
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def main():

    trans_list = ['RG', 'GC', 'GT', 'GG']

    for trans in trans_list:

        train_all_trans(trans)


def train_all_trans(trans):
    args = get_arguments()
    if args.sub and (args.dataset_name is 'lilac'):
        raise RuntimeError('Not support lilac dataset with sub')
    set_seed(124578)

    print(f'\nTraining {trans} model with mode of {args.mode}......')

    sub = '_sub' if args.sub else ''
    de = '_del' if args.de else ''
    current, voltage, labels, num_class = load_data(root_path=args.root_path,
                                                    dataset_name=args.dataset_name,
                                                    input_len=args.input_len,
                                                    sub=sub,
                                                    mode=args.mode,
                                                    de=args.de)

    print(f'Training model on {args.dataset_name+sub} dataset with {de.strip("_")} ......')

    in_2d_channels = 3 if args.dataset_name == 'lilac' else 1

    if args.input is 'p':
        current = current*voltage

        mean, std = np.mean(current), np.std(current)
        current = (current - mean) / std

        print('Use the power as input')
    elif args.input == 'cv':
        current = np.concatenate((current, voltage), axis=1)
        in_2d_channels = 2

        print('Use the current and voltage  as input')
    elif args.input == 'f':
        _, current = fryze_power_decomposition(current, voltage)

        print('Use the fryze current as input')
    else:

        print('Use the current as input')

    model_save_path = f'./checkpoint/{args.dataset_name+trans+sub+args.mode+de+args.input}.pth'

    f_weighted_list, f_macro_list, mcc_list, zl_list = [], [], [], []

    skf = StratifiedKFold(n_splits=10, random_state=798, shuffle=True)  # 42

    for i, (train_index, test_index) in enumerate(skf.split(current, labels)):

        model, criterion, optimizer, device = get_model(layers=args.layers,
                                                        num_class=num_class,
                                                        in_2d_channels=in_2d_channels,
                                                        dropout=args.dropout,
                                                        lr=args.lr,
                                                        trans=trans,
                                                        mode=args.mode)

        model_input = torch.FloatTensor(current)
        x_train, x_test = model_input[train_index], model_input[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        train_iterator, val_iterator, test_iterator = get_iterator(x_train, x_test, y_train, y_test,
                                                                   batch_size=args.batch_size)

        f_weighted, f_macro, mcc, zl = run(train_iterator=train_iterator,
                                           val_iterator=val_iterator,
                                           test_iterator=test_iterator,
                                           model=model,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           epochs=args.n_epoch,
                                           device=device,
                                           save_path=model_save_path,
                                           seed=i,
                                           dataset_name=args.dataset_name,
                                           sub=sub,
                                           early_stop=args.early_stop,
                                           patience=args.patience)

        f_weighted_list.append(f_weighted*100)
        f_macro_list.append(f_macro*100)
        mcc_list.append(mcc)
        zl_list.append(zl)

    print(f'The mean of {args.num_val} times val on {args.dataset_name+sub+de} dataset for {trans}')
    print('*'*50)
    print('f_weighted mean: {} std: {}'.format(np.mean(np.array(f_weighted_list)), np.std(np.array(f_weighted_list))))
    print('f_macro mean:    {} std: {}'.format(np.mean(np.array(f_macro_list)), np.std(np.array(f_macro_list))))
    print('mcc mean:        {} std: {}'.format(np.mean(np.array(mcc_list)), np.std(np.array(mcc_list))))
    print('zl mean:         {} std: {}'.format(np.mean(np.array(zl_list)), np.std(np.array(zl_list))))
    print('-'*50)
    print('f_weighted_list: {}'.format(f_weighted_list))
    print('f_macro_list:    {}'.format(f_macro_list))
    print('mcc_list:        {}'.format(mcc_list))
    print('zl_list:         {}'.format(zl_list))
    print('*'*50)


def fryze_power_decomposition(i, v):

    p = i*v
    vrs = v**2
    p_mean = np.mean(p, axis=-1)[:, :, None]
    vrs_mean = np.mean(vrs, axis=-1)[:, :, None]
    i_active = p_mean*v/vrs_mean
    i_non_active = i - i_active
    return i_active, i_non_active


def set_seed(seed=75286):
    import os
    print("Set seed: {}".format(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
