import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import os


def weight_init(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if type(m.weight) is torch.nn.parameter.Parameter:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
                
                
                
                
def draw_tSNE(data, label, label_names, num_axis=2, figsize=(9,6)):
    # Args:
    #    data        = np-array (N, embed-dim)
    #    label       = np-array (N, 1)
    #    label_names = list
    
    # import lib
    from sklearn import datasets
    from sklearn.manifold import TSNE
    from matplotlib import pyplot as plt
    #%matplotlib inline
    
    # tsne cfg
    tsne = TSNE(n_components=num_axis, random_state=0)
    data_dim_reduced = tsne.fit_transform(data)
    
    # visualization cfg
    colors = plt.cm.hsv(np.linspace(0, 1, len(label_names))).tolist()
    plt.figure(figsize=figsize)
    
    for idx, (color, label_name) in enumerate(zip(colors, label_names)):
        dim_1 = data_dim_reduced[label == idx,0]
        dim_2 = data_dim_reduced[label == idx,1]
        plt.scatter(dim_1, dim_2, c=color, label=label_name)
    plt.legend()
    plt.show()