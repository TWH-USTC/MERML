import numpy as np
import globalvaribale
import os
from preprocess import data_load, delete_nan_mask, train_test_split


def global_variable_init():
    globalvaribale.init()
    globalvaribale.set_value("SEED", 0)
    globalvaribale.set_value("PATH", os.getcwd() + "/Fenton-data.csv")
    POLLUTANT_TYPE = ['phenol', 'chlorophenol', 'nitrophenol', 'p-hydroxybenzoic acid', 'quinol',
                      'p-Hydroxybenzaldehyde',
                      'p-Hydroxyacetophenone', 'p-hydroxyanisole', 'Methyl 4-hydroxybenzoate',
                      'p-Hydroxybenzyl Alcohol',
                      '4-acetamidophenol', 'p-methylphenol']
    SMILES = ['C1=CC=C(C=C1)O', "C1=CC(=CC=C1O)Cl", "C1=CC(=CC=C1[N+](=O)[O-])O", "C1=CC(=CC=C1C(=O)O)O",
              'C1=CC(=CC=C1O)O',
              'C1=CC(=CC=C1C=O)O', 'CC(=O)C1=CC=C(C=C1)O', 'COC1=CC=C(C=C1)O', 'COC(=O)C1=CC=C(C=C1)O',
              'C1=CC(=CC=C1CO)O',
              'CC(=O)NC1=CC=C(C=C1)O', 'CC1=CC=C(C=C1)O']
    MAP_POLLUTANT_TO_SMILES = dict(zip(POLLUTANT_TYPE, SMILES))
    globalvaribale.set_value("MAP_POLLUTANT_TO_SMILES", MAP_POLLUTANT_TO_SMILES)

    # structural_descriptors_path = os.getcwd() + "/structural-descriptors.npy"
    # globalvaribale.set_value('structural_descriptors_path', structural_descriptors_path)


def data_init(type, ID=0, random=False):
    data = data_load()
    # data = data[12:, :]
    data[:, -7:] = (1 - data[:, -7:])
    # data[:, -8] = data[:, -8]/data[:, -10]

    F_H = data[:, -9]/data[:, -8]
    F_C = data[:, -9]/data[:, -10]
    H_C = data[:, -8]/data[:, -10]
    data = np.c_[np.c_[np.c_[np.c_[data[:, :-9], F_C], H_C], F_H], data[:, -7:]]

    # data[:, -1] = data[:, -1] * data[:, -10]
    # data[:, -2] = data[:, -2] * data[:, -10]
    # data[:, -3] = data[:, -3] * data[:, -10]
    # data[:, -4] = data[:, -4] * data[:, -10]
    # data[:, -5] = data[:, -5] * data[:, -10]
    # data[:, -6] = data[:, -6] * data[:, -10]
    if random:
        from preprocess import random_label
        data = random_label(data)
    else:
        pass

    if type == 'evaluate':
        mask_train, mask_test = train_test_split(data, 0.2)

    elif type == 'pollutant':
        mask_train = []
        mask_test = []
        for index in range(data.shape[0]):
            if index % 12 == ID:
                mask_test.append(index)
            else:
                mask_train.append(index)

    elif type == 'concentration':
        np.random.seed(ID)
        id_list = np.random.randint(0, 8, size=3)
        mask_test_1 = id_list[0]*12 + np.arange(0, 12, 1)
        mask_test_1 = mask_test_1.tolist()
        mask_test_2 = id_list[1] * 12 + np.arange(0, 12, 1) + 96
        mask_test_2 = mask_test_2.tolist()
        mask_test_3 = id_list[2] * 12 + np.arange(0, 12, 1) + 96*2
        mask_test_3 = mask_test_3.tolist()
        mask_test = mask_test_1 + mask_test_2 + mask_test_3
        mask_all = [index for index in range(data.shape[0])]
        from preprocess import list_reduce
        mask_train = list_reduce(mask_all, mask_test)

    elif type == 'learning curve':
        mask_train_ori, mask_test = train_test_split(data, 0.2)
        train_ori = data[mask_train_ori, :]
        mask_train = train_test_split(train_ori, 1-ID)[0]

    train = data[mask_train, :]
    test = data[mask_test, :]
    mask_nan_train = delete_nan_mask(train)
    train = train[mask_nan_train, :]
    mask_nan_test = delete_nan_mask(test)
    test = test[mask_nan_test, :]
    return train, test, data










