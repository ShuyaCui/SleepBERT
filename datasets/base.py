from .utils import *
from config import RAW_DATASET_ROOT_FOLDER

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
tqdm.pandas()

from abc import *
from pathlib import Path
import os
import tempfile
import shutil
import pickle
from sklearn.model_selection import train_test_split
import csv

class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        self.args = args
        self.test = args.test
        self.val = args.val
        self.bin_num = args.bin_num
        # assert self.min_uc >= 2

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    def raw_code(cls):
        return cls.code()

    @abstractmethod
    def load_act_df(self, file_path):
        pass

    def load_dataset(self):
        dataset_path = self._get_preprocessed_dataset_path()
        if os.path.isfile(dataset_path):
            bin_edges = np.load(self._get_bin_edge_dataset_path())
            dataset = pickle.load(dataset_path.open('rb'))
        else:
            bin_edges = self.preprocess(self.test, self.val)
            dataset = pickle.load(dataset_path.open('rb'))
        return dataset, bin_edges

    def preprocess(self, test=True, val = True):
        dataset_path = self._get_preprocessed_dataset_path()
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)

        folder_path = self._get_rawdata_folder_path()
        input_files = os.listdir(folder_path)
        # all_files = {}
        min_val = 0
        max_val = 0
        for idx in trange(len(input_files)):
            file = input_files[idx]
            file_path = folder_path.joinpath(file)
            input_files[idx] = file_path
            act = self.load_act_df(file_path)
            if np.min(act) < min_val:
                min_val = np.min(act)
            if np.max(act) > max_val:
                max_val = np.max(act)
            # all_files[idx] = act
        
        # calculate the bin edges using numpy's linspace function
        bin_edges = np.linspace(min_val, max_val, num=self.bin_num+1)
        print(bin_edges)
        tmp_path = self._get_bin_edge_dataset_path()
        np.save(tmp_path, bin_edges)
        # # use numpy's digitize function to categorize the values
        # for idx in range(len(input_files)):
        #     act_bin = np.digitize(all_files[idx], bin_edges)
        #     all_files[idx] = act_bin
        
        if test & val:
            data_train, data_test = train_test_split(input_files, test_size=self.args.test_set_size, random_state=self.args.dataset_split_seed)
            data_train, data_val = train_test_split(input_files, test_size=self.args.val_set_size, random_state=self.args.dataset_split_seed)
            dataset = {'train': data_train,
                       'val': data_val,
                       'test': data_test}
        elif test:
            data_train, data_test = train_test_split(input_files, test_size=self.args.test_set_size, random_state=self.args.dataset_split_seed)
            dataset = {'train': data_train,
                       'test': data_test}
        else:
            data_train, data_val = train_test_split(input_files, test_size=self.args.val_set_size, random_state=self.args.dataset_split_seed)
            dataset = {'train': data_train,
                       'val': data_val}

        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

        return bin_edges

    def _get_rawdata_root_path(self):
        return Path(RAW_DATASET_ROOT_FOLDER)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.raw_code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_root_path()
        return folder.joinpath('dataset.pkl')
    
    def _get_bin_edge_dataset_path(self):
        folder = self._get_preprocessed_root_path()
        return folder.joinpath('bin_edge.npy')

