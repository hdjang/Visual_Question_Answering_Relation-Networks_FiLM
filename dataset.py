from __future__ import print_function
import argparse
import os
import pickle
import random
import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset


class SortOfCLEVR(Dataset):
    def __init__(self, name, dataset, args):
        self.name = name
        self.no_img_norm = args.no_img_norm
        self.dataset_rel, self.dataset_nonrel = self.preprocess(dataset)
        self.dataset = self.dataset_rel + self.dataset_nonrel
        
        random.shuffle(self.dataset)
        
        if self.name == "train":
            print("{}-dataset loaded. {} rel+nonrel samples".format(name, len(self)))
        elif self.name == "val":
            print("{}-dataset loaded. {}/{} rel/nonrel samples".format(name, len(self), len(self)))
        
    def preprocess(self, dataset):
        dataset_rel    = list()
        dataset_nonrel = list()
        for img, rel_qas, nonrel_qas in dataset:
            for qst,ans in zip(rel_qas[0], rel_qas[1]):
                dataset_rel.append([img,qst,ans])
            for qst,ans in zip(nonrel_qas[0], nonrel_qas[1]):
                dataset_nonrel.append([img,qst,ans])
        return dataset_rel, dataset_nonrel
    
    def __len__(self):
        if self.name == "train":
            return len(self.dataset)
        elif self.name == "val":
            return len(self.dataset_rel)
        
    def __getitem__(self, idx):
        if self.name == "train":
            if self.no_img_norm:
                img = torch.from_numpy(self.dataset[idx][0]).permute(2,0,1).float()
            else:
                img = torch.from_numpy(self.dataset[idx][0]/255.0).permute(2,0,1).float()
            qst = torch.from_numpy(self.dataset[idx][1]).float()
            ans = torch.tensor(self.dataset[idx][2]).long()
            return img, qst, ans
        elif self.name == "val":
            if self.no_img_norm:
                img_rel = torch.from_numpy(self.dataset_rel[idx][0]).permute(2,0,1).float()
            else:
                img_rel = torch.from_numpy(self.dataset_rel[idx][0]/255.0).permute(2,0,1).float()
            qst_rel = torch.from_numpy(self.dataset_rel[idx][1]).float()
            ans_rel = torch.tensor(self.dataset_rel[idx][2]).long()
            
            if self.no_img_norm:
                img_nonrel = torch.from_numpy(self.dataset_nonrel[idx][0]).permute(2,0,1).float()
            else:
                img_nonrel = torch.from_numpy(self.dataset_nonrel[idx][0]/255.0).permute(2,0,1).float()
            qst_nonrel = torch.from_numpy(self.dataset_nonrel[idx][1]).float()
            ans_nonrel = torch.tensor(self.dataset_nonrel[idx][2]).long()
            
            return [img_rel, img_nonrel], [qst_rel, qst_nonrel], [ans_rel, ans_nonrel]

        
def get_dataset(args):
    clevr = pickle.load(open(args.data_dir, 'rb'))
    train_set = SortOfCLEVR("train", clevr[0], args)
    val_set   = SortOfCLEVR("val", clevr[1], args)
    return train_set, val_set