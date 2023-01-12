
import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Train a neural network\
                                     for energy disaggregation - \
                                     network input = mains window; \
                                     network target = the states of \
                                     the target appliance.')

    parser.add_argument('--root_path',
                        type=str,
                        default='./data',
                        help='this is the directory of the training samples')

    parser.add_argument('--dataset_name',
                        type=str,
                        default='plaid2018',
                        help='this is the name of dataset')

    parser.add_argument('--mode',
                        type=str,
                        default='CNN',
                        help='this is the mode')

    parser.add_argument('--input',
                        type=str,
                        default='c',
                        help='The input type, including c(current), '
                             'cv(current and voltage), p(power), f(no activate i)')

    parser.add_argument('--sub',
                        default=False,
                        action="store_true",
                        help='use the sub dataset in plaid')

    parser.add_argument('--isc',
                        default=False,
                        action="store_true",
                        help='use the isc dataset in lilac')

    parser.add_argument('--de',
                        default=False,
                        action="store_true",
                        help='use the del 2018sub dataset in plaid')

    parser.add_argument('--early_stop',
                        default=False,
                        action="store_true",
                        help='use the early stopping in training process')

    parser.add_argument('--patience',
                        type=int,
                        default=60,
                        help='the number of epoch with no improve in using early stopping')

    parser.add_argument('--input_len',
                        type=int,
                        default=200,
                        help='the input time len')

    parser.add_argument('--num_val',
                        type=int,
                        default=10,
                        help='the times of val')

    parser.add_argument('--layers', default=[1, 1, 2, 4],
                        type=int,
                        nargs="+",
                        help='layers list')

    parser.add_argument('--save_dir',
                        type=str,
                        default='./models',
                        help='this is the directory to save the trained models')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='The batch size of training examples')

    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--dropout', '--do', default=0.5, type=float,
                        metavar='DO', help='dropout ratio (default: 0.5)')

    parser.add_argument('--n_epoch',
                        type=int,
                        default=1000,
                        help='The number of epochs.')

    return parser.parse_args()
