# -*- coding: utf-8 -*-

from options import args
from models import BERTModel
from datasets import dataset_factory
from dataloaders.bert import BertDataloader
from trainers.bert import BERTTrainer
from utils import *
import multiprocessing
import torch

def pretrain():
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    export_root = setup_train(args)

    dataset = dataset_factory(args)

    train_loader, val_loader = BertDataloader(args, dataset).get_pretrain_dataloaders()

    model = BERTModel(args)

    trainer = BERTTrainer(args, model=model, train_loader=train_loader, val_loader=val_loader,test_loader=None, export_root=export_root)

    trainer.train()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', True)
    pretrain()

