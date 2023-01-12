import torch
from dataset.dataloder import load_data, get_iterator
from opts import get_arguments
from sklearn.model_selection import StratifiedKFold, train_test_split
from models.model_builder import get_model
from models.run_model import run
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def main():
    trans_list = ['RG']
    # trans_list = ['RG', 'GC', 'GT', 'FC']
    # trans_list = ['VI']
    # trans_list = ['AWRG']
    trans_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for trans in trans_list:
        print(f'The rate use {trans}')
        train_all_trans(trans)

    # dataset_seed_list = [random.randint(0, 100000) for x in range(20)]

    # for seed in dataset_seed_list:
    #     print(seed)
    #     train_all_trans(dataset_seed=seed)


def train_all_trans(rate):
    args = get_arguments()
    set_seed()

    trans = 'RG'
    print(f'Training {trans} model with mode of {args.mode}......')
    isc = '_isc' if args.isc else ''
    sub = '_sub' if args.sub else ''
    de = '_del' if args.de else ''
    current, voltage, labels, num_class = load_data(root_path=args.root_path,
                                                    dataset_name=args.dataset_name,
                                                    isc=isc,
                                                    input_len=args.input_len,
                                                    sub=sub,
                                                    trans=trans,
                                                    mode=args.mode,
                                                    de=args.de)

    print(f'\nTraining model on {args.dataset_name+isc+sub} dataset with {de.strip("_")} ......')

    in_2d_channels = 3 if args.dataset_name == 'lilac' else 1
    model_save_path = f'./checkpoint/{args.dataset_name+trans+isc+sub+args.mode+de}.pth'

    f_weighted_list, f_macro_list, mcc_list, zl_list = [], [], [], []

    skf = StratifiedKFold(n_splits=10, random_state=798, shuffle=True)  # 42

    for i, (train_index, test_index) in enumerate(skf.split(current, labels)):
        # print(test_index)
        model, criterion, optimizer, device = get_model(layers=args.layers,
                                                        num_class=num_class,
                                                        in_2d_channels=in_2d_channels,
                                                        dropout=args.dropout,
                                                        lr=args.lr,
                                                        trans=trans,
                                                        mode=args.mode)

        model_input = torch.FloatTensor(current)
        x_train_all, x_test = model_input[train_index], model_input[test_index]
        y_train_all, y_test = labels[train_index], labels[test_index]

        x_train, _, y_train, _ = train_test_split(x_train_all, y_train_all, test_size=rate,
                                                  random_state=42, stratify=y_train_all)

        train_iterator, val_iterator, test_iterator = get_iterator(x_train, x_test, y_train, y_test,
                                                                   batch_size=args.batch_size)
        # train_iterator, val_iterator, test_iterator = get_iterator(model_input, model_input, labels, labels,
        #                                                            batch_size=args.batch_size)

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
                                           isc=isc,
                                           sub=sub,
                                           early_stop=args.early_stop,
                                           patience=args.patience)

        f_weighted_list.append(f_weighted*100)
        f_macro_list.append(f_macro*100)
        mcc_list.append(mcc)
        zl_list.append(zl)

    print(f'The mean of {args.num_val} times val on {args.dataset_name+isc+sub+de} dataset for {trans}')
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
