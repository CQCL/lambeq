#!/usr/bin/env python
# coding: utf-8

# # Training: Classical case

# ## Preparation
# 
# We start with importing PyTorch and specifying some training hyperparameters.

# In[1]:
from pytorch_trainer_cosinesim import PytorchTrainerCosineSim
from lambeq import PytorchModel
import torch
import logging
from discopy import grammar
logging.basicConfig(level=logging.INFO)
from mithun.utils.utils import *
from mithun.utils.dataset_mithun import Dataset
from lambeq import BobcatParser, SpacyTokeniser

class NLITrainer:
    def __init__(self):
        self.config = read_config()
        self.tokeniser = SpacyTokeniser()
        self.parser = BobcatParser(verbose='text')

    def read_data(self,path):
        labels, data = read_data_float_label(
            get_full_path(self.config['BASE_PATH_DATA'], self.config['SNLI_TRAIN_LAMBEQ_FORMAT_CLAIM_SMALL']))
        return labels, data


    def get_path(self,data_type):
        return get_full_path(self.config['BASE_PATH_DATA'], self.config[data_type])

    def create_train_dev_test(self):
        self.train_labels, self.train_data_claim = self.read_data(self.get_path('SNLI_TRAIN_LAMBEQ_FORMAT_CLAIM_SMALL'))


        path = get_full_path(self.config['BASE_PATH_DATA'], self.config['SNLI_TRAIN_LAMBEQ_FORMAT_CLAIM_SMALL'])
        self.train_labels, self.train_data_claim = self.read_data(path)
        print(self.train_labels)
        print(self.train_data_claim)

        path = get_full_path(self.config['BASE_PATH_DATA'], self.config['SNLI_TRAIN_LAMBEQ_FORMAT_CLAIM_SMALL'])
        self.train_labels, self.train_data_claim = self.read_data(path)
        print(self.train_labels)
        print(self.train_data_claim)

        path = get_full_path(self.config['BASE_PATH_DATA'], self.config['SNLI_TRAIN_LAMBEQ_FORMAT_CLAIM_SMALL'])
        self.train_labels, self.train_data_claim = self.read_data(path)
        print(self.train_labels)
        print(self.train_data_claim)

        #
        # train_labels, train_data_evidence = read_data_float_label(
        #     get_full_path(config['BASE_PATH_DATA'], config['SNLI_TRAIN_LAMBEQ_FORMAT_EVIDENCE_SMALL']))
        #
        # val_labels, val_data_claim = read_data_float_label(
        #     get_full_path(config['BASE_PATH_DATA'], config['SNLI_DEV_LAMBEQ_FORMAT_CLAIM_SMALL']))
        # val_labels, val_data_evidence = read_data_float_label(
        #     get_full_path(config['BASE_PATH_DATA'], config['SNLI_DEV_LAMBEQ_FORMAT_EVIDENCE_SMALL']))
        #

def main():
    nlitrainer=NLITrainer()
    labels, data= nlitrainer.read_data("")
    nlitrainer.create_train_dev_test()


if __name__=="__main__":
    main()


