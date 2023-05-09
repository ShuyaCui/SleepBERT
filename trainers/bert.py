from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks


import torch
import torch.nn as nn
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def to_np(x):
    return x.cpu().detach().numpy()

# define MixUp Loss
class MixUpLoss(nn.Module):

    def __init__(self, args, device):
        super(MixUpLoss, self).__init__()
        
        self.tau = args.tau
        self.device = device
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, z_aug, z_1, z_2, lam, batch_size):

        # z_1 = nn.functional.normalize(z_1)
        # z_2 = nn.functional.normalize(z_2)
        # z_aug = nn.functional.normalize(z_aug)

        labels_lam_0 = lam*torch.eye(batch_size, device=self.device)
        labels_lam_1 = (1-lam)*torch.eye(batch_size, device=self.device)

        labels = torch.cat((labels_lam_0, labels_lam_1), 1)
        logits = torch.cat((torch.mm(z_aug, z_1.T),
                         torch.mm(z_aug, z_2.T)), 1)
        loss = self.cross_entropy(logits / self.tau, labels)

        return loss

    def cross_entropy(self, logits, soft_targets):
        return torch.mean(torch.sum(- soft_targets * self.logsoftmax(logits), 1))


class BERTTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader,  export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.alpha = args.alpha
        self.mixup_alpha = args.mixup_alpha
        self.MixUpLoss = MixUpLoss(self.args, self.args.device)

        self.lambda_=args.lambda_
        self.theta = 0

    @classmethod
    def code(cls):
        return 'bert'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass

    def calculate_loss(self, batch):
        # b = self.args.batch_size
        main_loss = 0
        mixup_loss = 0
        seqs = batch[0]
        labels = batch[1] # [b,max_length], [b, max_length]
        logits, c = self.model(seqs) # [b, max_length, bin_size+1], [b, max_length, hidden_size]
        logits_flat = logits.view(logits.shape[0] * logits.shape[1], logits.shape[2])
        labels_flat = labels.view(labels.shape[0] * labels.shape[1])
        loss_k = self.ce(logits_flat, labels_flat)
        main_loss += loss_k

        # calculate mixup contrastive loss
        z_1, _ = self.model(batch[2])
        z_2, _ = self.model(batch[3])
        z_aug, _ = self.model(batch[4]) # [batch_size, seq_len, bin_size+1]
        mixup_loss = self.MixUpLoss(z_aug.reshape(logits.shape[0], logits.shape[1] * logits.shape[2]), z_1.reshape(logits.shape[0], logits.shape[1] * logits.shape[2]), z_2.reshape(logits.shape[0], logits.shape[1] * logits.shape[2]), self.mixup_alpha, torch.tensor(logits.shape[0]))

        num_main_loss = main_loss.data.item()
        num_mixup_loss = mixup_loss.data.item()
        theta_hat = num_main_loss/(num_main_loss+self.lambda_*num_mixup_loss)
        self.theta = self.alpha*theta_hat+(1-self.alpha)*self.theta
        print(self.theta)
        total_loss = main_loss + self.theta*mixup_loss

        return total_loss, main_loss, mixup_loss

    def calculate_metrics(self, batch):
        # b = self.args.batch_size
        seqs, labels = batch
        scores, _ = self.model(seqs) # [b, max_length, bin_size+1], [b, max_length, hidden_size]
        softmax = nn.Softmax(dim=-1)
        final = softmax(scores)[..., 1:-1]
        predictions = final.argmax(dim=-1) + 1

        correct_num = ((labels != self.args.bert_mask_token) * (predictions == labels)).sum(dim=-1)[0].item()
        val_num = (labels != self.args.bert_mask_token).sum(dim=-1)[0].item()
        acc = correct_num / val_num
        return acc