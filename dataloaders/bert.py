from .base import AbstractDataloader
from .negative_samplers import negative_sampler_factory
from .data_augmentation import *
from .similarity import *

import torch
import torch.utils.data as data_utils
import torch.distributed as dist
import copy
import random
import numpy as np
import pandas as pd
import csv

class BertDataloader(AbstractDataloader):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)

        # -----------   online based on shared item embedding for item similarity --------- #

        #args.online_similarity_model = OnlineItemSimilarity(item_size=args.bin_num + 2)

        self.max_len = args.bert_max_len
        self.mask_prob = args.bert_mask_prob
        self.bin_num = args.bin_num
        self.CLOZE_MASK_TOKEN = 1e-5
        self.num_positive = args.num_positive
        self.rng = random.Random(args.model_init_seed)
        self.device = args.device

    @classmethod
    def code(cls):
        return 'bert'

    def get_pytorch_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        test_loader = self._get_test_loader()
        return train_loader, val_loader, test_loader

    def get_pretrain_dataloaders(self):
        train_loader = self._get_train_loader()
        val_loader = self._get_val_loader()
        return train_loader, val_loader

    def _get_train_loader(self):
        dataset = self._get_train_dataset()

        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True, num_workers=self.args.num_workers)
        return dataloader

    def _get_train_dataset(self):
        dataset = BertTrainDataset(self.data_train, self.num_positive, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.bin_num, self.bin_edges, self.rng, self.device)
        return dataset

    def _get_val_loader(self):
        return self._get_eval_loader(mode='val')

    def _get_test_loader(self):
        return self._get_eval_loader(mode='test')

    def _get_eval_loader(self, mode):

        batch_size = self.args.val_batch_size if mode == 'val' else self.args.test_batch_size

        dataset = self._get_eval_dataset(mode)

        dataloader = data_utils.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True, pin_memory=True,num_workers=self.args.num_workers)
        return dataloader

    def _get_eval_dataset(self, mode):
        if mode=='val':
            dataset = BertEvalDataset(self.data_val, self.max_len, self.mask_prob, self.CLOZE_MASK_TOKEN, self.bin_num, self.bin_edges, self.rng, self.device)
        else:
            answers = self.data_test
            dataset = BertTestDataset(self.data_train, self.data_val, answers, self.max_len, self.bin_num, self.CLOZE_MASK_TOKEN)
        return dataset

class RecWithContrastiveLearning():
    def __init__(self, args, MASK_TOKEN):
        self.args = args
        self.max_len = args.bert_max_len
        # currently apply one transform, will extend to multiples
        # it takes one sequence of items as input, and apply augmentation operation to get another sequence
        self.similarity_model = [args.offline_similarity_model, args.online_similarity_model]
        self.augmentations = {'truncation': Truncation(truncation_rate=args.truncation_rate),
                              'mask': Mask(gamma=args.mask_rate,mask_token=MASK_TOKEN),
                              'reorder': Reorder(beta=args.reorder_rate),
                              'substitute': Substitute(self.similarity_model,
                                                substitute_rate=args.substitute_rate),
                              'insert': Insert(self.similarity_model,
                                               insert_rate=args.insert_rate,
                                               max_insert_num_per_pos=args.max_insert_num_per_pos),
                              'random': Random(truncation_rate=args.truncation_rate, gamma=args.mask_rate,
                                                beta=args.reorder_rate, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate,
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate,
                                                augment_threshold=self.args.augment_threshold,
                                                augment_type_for_short=self.args.augment_type_for_short),
                              'combinatorial_enumerate': CombinatorialEnumerate(truncation_rate=args.truncation_rate, gamma=args.mask_rate,
                                                beta=args.reorder_rate, item_similarity_model=self.similarity_model,
                                                insert_rate=args.insert_rate,
                                                max_insert_num_per_pos=args.max_insert_num_per_pos,
                                                substitute_rate=args.substitute_rate, n_views=args.n_views)
                            }
        if self.args.base_augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.base_augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.base_augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.base_augment_type]

    def augment(self, input_ids):
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len:]

            augmented_seqs.append(augmented_input_ids)
        return augmented_seqs

class BertTrainDataset(data_utils.Dataset):
    def __init__(self, samples, num_positive, max_len, mask_prob, mask_token, bin_num, bin_edges, rng, device):
        self.samples = samples
        self.num_positive = num_positive
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.bin_num = bin_num
        self.bin_edges = bin_edges
        self.rng = rng
        self.device = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        seq = self._getseq(sample)
        seq = torch.from_numpy(seq).long()
        seq = torch.cat((seq, torch.tensor([0]))).to(self.device)
        #negs = self.negative_samples[sample]
        return_list=[]
        for i in range(self.num_positive):
            tokens, labels = self.get_masked_seq(seq)
            return_list.append(torch.LongTensor(tokens))
            return_list.append(torch.LongTensor(labels))
        return tuple(return_list)

    def _getseq(self, sample):        
        act = self._load_act_df(sample)
        act_1 = self._get_seq_slidewindow(act)
        act_bin = np.digitize(act_1, self.bin_edges)
        return act_bin

    def _load_act_df(self, file_path):
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the first row
            next(reader)
            act = []
            for row in reader:
                if row[1] == "NA":
                    act.append(0)
                else:
                    act.append(row[1])
        act = np.array(act)
        act = act.astype(np.float)
        return act

    def _get_seq_slidewindow(self, seq):
        seq_len = len(seq)
        beg_idx = self.rng.randint(0, seq_len-self.max_len)
        temp = seq[beg_idx:beg_idx + self.max_len]
        return temp
    
    def get_masked_seq(self, seq):
        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.bin_num))
                else:
                    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]
            mask_len = self.max_len - len(tokens)
            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels
        return tokens,labels

class BertFinetuneDataset(data_utils.Dataset):

    def __init__(self, seq, label, max_len, mask_token):

        self.seq = seq

        self.samples = sorted(self.seq.keys())
        self.max_len = max_len

        self.mask_token = mask_token
        self.label = label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        seq = self._getseq(sample)
        negs = self.negative_samples[sample]

        tokens = seq+[self.mask_token]
        labels = self.label[sample]
        tokens = tokens[-self.max_len:]
        negs = negs[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * (self.max_len-1) + labels
        negs = [0] * (self.max_len-len(negs)) + negs

        return torch.LongTensor(tokens),  torch.LongTensor(labels), torch.LongTensor(negs)

    def _getseq(self, sample):
        return self.seq[sample]

class BertCLDataset(data_utils.Dataset):

    def __init__(self, seq,cl_data):
        self.seq = seq
        self.samples = sorted(self.seq.keys())
        self.cl_data=cl_data


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]
        seq = self._getseq(sample)
        aug = self.cl_data.augment(seq)

        return torch.LongTensor(aug[0]), torch.LongTensor(aug[1])

    def _getseq(self, sample):
        return self.seq[sample]

class BertEvalDataset(data_utils.Dataset):
    # self.data_train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples
    def __init__(self, samples, max_len, mask_prob, mask_token, bin_num, bin_edges, rng, device):
        self.samples = samples
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.bin_num = bin_num
        self.bin_edges = bin_edges
        self.rng = rng
        self.device = device
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        seq = self._getseq(sample)
        seq = torch.from_numpy(seq).long()
        seq = torch.cat((seq, torch.tensor([0])))
        tokens, labels = self.get_masked_seq(seq)
        return torch.LongTensor(tokens), torch.LongTensor(labels)
    
    def get_masked_seq(self, seq):
        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.bin_num))
                else:
                    tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

            tokens = tokens[-self.max_len:]
            labels = labels[-self.max_len:]
            mask_len = self.max_len - len(tokens)
            tokens = [0] * mask_len + tokens
            labels = [0] * mask_len + labels
        return tokens,labels
    
    def _getseq(self, sample):        
        act = self._load_act_df(sample)
        act_1 = self._get_seq_slidewindow(act)
        act_bin = np.digitize(act_1, self.bin_edges)
        return act_bin
    
    def _load_act_df(self, file_path):
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            # Skip the first row
            next(reader)
            act = []
            for row in reader:
                if row[1] == "NA":
                    act.append(0)
                else:
                    act.append(row[1])
        act = np.array(act)
        act = act.astype(np.float)
        return act

    def _get_seq_slidewindow(self, seq):
        seq_len = len(seq)
        beg_idx = self.rng.randint(0, seq_len-self.max_len)
        temp = seq[beg_idx:beg_idx + self.max_len]
        return temp

class BertTestDataset(data_utils.Dataset):
    # self.data_train, answers, self.max_len, self.CLOZE_MASK_TOKEN, self.test_negative_samples
    def __init__(self, seq, max_len, bin_num, mask_token):
        self.seq = seq
        self.val_seq = val_seq
        self.samples = sorted(self.seq.keys())
        self.max_len = max_len
        self.bin_num = bin_num
        self.mask_token = mask_token

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        seq = self._getseq(sample)
        seq = torch.from_numpy(seq).long()
        seq = torch.cat((seq, torch.tensor([0]))).to(self.device)
        for i in range(self.num_positive):
            tokens, labels = self.get_masked_seq(seq)
            return torch.LongTensor(tokens), torch.LongTensor(labels)
