import torch.utils.data as data
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import torch
import torch.nn as nn


def load_data(root_path=None, dataset_name=None, input_len=None, sub=None, mode=None, de=None):

    if dataset_name in ['lilac', 'plaid2018']:
        if sub:
            with open(f'{root_path}/{dataset_name}/sub/{dataset_name}{sub}.pickle', 'rb') as handle:
                current, voltage, labels, _ = pickle.load(handle)
            del_list = [1874, 1875]
            labels = np.delete(labels, del_list)
            type_count_dict = {}
            for item in list(labels):
                type_count_dict.update({item: list(labels).count(item)})

            current = np.delete(current, del_list, axis=0)
            voltage = np.delete(voltage, del_list, axis=0)
            if de:
                lit_type_list = [k for k, v in type_count_dict.items() if v < 21]
                del_list = []
                for c in lit_type_list:
                    for i, label in enumerate(labels):
                        if label == c:
                            del_list.append(i)
                labels = np.delete(labels, del_list)
                current = np.delete(current, del_list, axis=0)
                voltage = np.delete(voltage, del_list, axis=0)
                type_count_dict = {}
                for item in list(labels):
                    type_count_dict.update({item: list(labels).count(item)})
                print(type_count_dict)
        else:
            current = np.load(f'{root_path}/{dataset_name}/aggregated/current.npy')
            voltage = np.load(f'{root_path}/{dataset_name}/aggregated/voltage.npy')
            labels = np.load(f'{root_path}/{dataset_name}/aggregated/labels.npy')

        if dataset_name == 'lilac':
            correct_1_phase_motor = [920, 923, 956, 959, 961, 962, 1188]
            correct_hair = [922, 921, 957, 958, 960, 963, 1181, 1314]
            correct_bulb = [1316]
            labels[correct_1_phase_motor] = '1-phase-async-motor'
            labels[correct_hair] = 'Hair-dryer'
            labels[correct_bulb] = 'Bulb'

    elif dataset_name is 'whited':
        with open(f'{root_path}/{dataset_name}/{dataset_name}.pickle', 'rb') as handle:
            current, voltage, labels, _ = pickle.load(handle)

    else:
        raise RuntimeError('Not support this {} dataset'.format(dataset_name))

    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    labels = label_encoder.transform(labels)
    num_class = np.max(labels) - np.min(labels) + 1

    if len(current.shape) == 2:
        current = current[:, None, :]
        voltage = voltage[:, None, :]

    if mode in ['LSTM', 'GRU', 'Transformer']:
        current = multi_dimension_paa(current, emb_size=25)
        if len(current.shape) == 2:
            current = current[:, None, :]
        return current, voltage, labels, num_class

    current = torch.Tensor(current)
    current = nn.functional.interpolate(
        current,
        mode='nearest',
        size=input_len)
    current = np.array(current)

    voltage = torch.Tensor(voltage)
    voltage = nn.functional.interpolate(
        voltage,
        mode='nearest',
        size=input_len)
    voltage = np.array(voltage)

    return current, voltage, labels, num_class


def get_iterator(x_train=None, x_test=None, y_train=None, y_test=None, batch_size=16):

    train_dataset = Dataset(x_train, y_train)

    val_dataset = Dataset(x_test, y_test)
    test_dataset = Dataset(x_test, y_test)

    train_iterator = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_iterator = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_iterator = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_iterator, val_iterator, test_iterator


class Dataset(data.Dataset):

    def __init__(self, input_data, labels):
        self.input_data = input_data
        self.labels = labels

    def __getitem__(self, index):
        input_feature = self.input_data[index]
        label = self.labels[index]

        return input_feature, label

    def __len__(self):

        return len(self.labels)


def multi_dimension_paa(series: np.array, emb_size: int):

    paa_out = np.zeros((series.shape[0], emb_size))

    for k in range(series.shape[0]):
        paa_out[k] = paa(series[k].flatten(), emb_size)
    return paa_out


def paa(series: np.array, emb_size: int):

    series_len = len(series)

    if series_len == emb_size:
        return np.copy(series)
    else:
        res = np.zeros(emb_size)

        if series_len % emb_size == 0:
            inc = series_len // emb_size
            for i in range(0, series_len):
                idx = i // inc
                np.add.at(res, idx, series[i])

            return res / inc

        else:
            for i in range(0, emb_size * series_len):
                idx = i // series_len
                pos = i // emb_size
                np.add.at(res, idx, series[pos])

            return res / series_len
