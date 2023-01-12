from opts import get_arguments
from dataset.dataloder import load_data, get_iterator
from sklearn.model_selection import StratifiedKFold
import torch
from models.model_builder import CNN1dT2d
from models.model_test import test
import warnings
warnings.filterwarnings("ignore")


def main():

    args = get_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trans = 'FC'
    isc = '_isc' if args.isc else ''
    sub = '_sub' if args.sub else ''
    in_2d_channels = 3 if args.dataset_name == 'lilac' else 1
    model_save_path = f'./checkpoint/{args.dataset_name+isc+sub}.pth'

    current, _, labels, num_class = load_data(root_path=args.root_path,
                                              dataset_name=args.dataset_name,
                                              isc=isc,
                                              input_len=args.input_len,
                                              sub=sub)

    model = CNN1dT2d(in_2d_channels=in_2d_channels,
                     layer_list=args.layers,
                     dropout=args.dropout,
                     num_class=num_class,
                     trans=trans).to(device)

    skf = StratifiedKFold(n_splits=10, random_state=375557, shuffle=True)
    train_index, test_index = next(skf.split(current, labels))

    model_input = torch.FloatTensor(current)
    x_train, x_test = model_input[train_index], model_input[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    # train_iterator, val_iterator, test_iterator = get_iterator(x_train, x_test, y_train, y_test,
    #                                                            batch_size=args.batch_size)

    train_iterator, val_iterator, test_iterator = get_iterator(model_input, model_input, labels, labels,
                                                               batch_size=args.batch_size)

    f_weighted, f_macro, mcc, zl = test(model=model,
                                        test_iterator=test_iterator,
                                        device=device,
                                        model_path=model_save_path,
                                        dataset_name=args.dataset_name,
                                        isc=isc,
                                        sub=sub)

    print('f_weighted: {}'.format(f_weighted))
    print('f_macro:    {}'.format(f_macro))
    print('mcc:        {}'.format(mcc))
    print('zl:         {}'.format(zl))


if __name__ == "__main__":
    main()
