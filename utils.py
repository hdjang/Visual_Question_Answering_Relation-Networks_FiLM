import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import pickle
import time

import pdb


def weight_init(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.zero_()
            
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            if type(m.weight) is torch.nn.parameter.Parameter:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
def draw_tSNE(data, label, label_names, num_axis=2, figsize=(9,6)):
    # Args:
    #    data        = np-array (N, embed-dim)
    #    label       = np-array (N, 1)
    #    label_names = list
    
    # tsne cfg
    tsne = TSNE(n_components=num_axis, random_state=0)
    data_dim_reduced = tsne.fit_transform(data)
    
    # visualization cfg
    colors = plt.cm.hsv(np.linspace(0, 0.8, len(label_names)), 1).tolist()
    plt.figure(figsize=figsize)
    
    for idx, (color, label_name) in enumerate(zip(colors, label_names)):
        dim_1 = data_dim_reduced[label == idx,0]
        dim_2 = data_dim_reduced[label == idx,1]
        plt.scatter(dim_1, dim_2, c=color, label=label_name, alpha=1)
    plt.legend()
    plt.show()
    


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def clear(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.
        
        
    
class Manifold_handler:
    def __init__(self, args):
        self.args = args
        self.save_path = os.path.join(*args.checkpoint.split("/")[:-1]) + "/{}_manifold_src.pickle".format(args.model)
        if args.model == "FiLM":
            self.manifold_src = {
                "gamma_1":[],
                "beta_1" :[],
                "gamma_2":[],
                "beta_2" :[],
                "gamma_3":[],
                "beta_3" :[],
                "gamma_4":[],
                "beta_4" :[],
                "label":[],
            }
        elif args.model == "RN":
            self.manifold_src = {
                "rn-g" :[],
                "rn-f" :[],
                "label":[],
            }
            
    def accumulate(self, manifold_src):
        batch_size = manifold_src['qst_rel'].shape[0]
        for b in range(batch_size):
            
            # parse label
            qst_rel = manifold_src['qst_rel'][b].cpu().numpy()
            qst_rel = np.expand_dims(qst_rel, axis=0)
            qst_nonrel = manifold_src['qst_nonrel'][b].cpu().numpy()
            qst_nonrel = np.expand_dims(qst_nonrel, axis=0)
            label_rel = np.concatenate([np.ones((1,1)), qst_rel], axis=1)
            label_nonrel = np.concatenate([np.zeros((1,1)), qst_nonrel], axis=1)
            
            # accumulate label
            self.manifold_src['label'].append(label_rel)
            self.manifold_src['label'].append(label_nonrel)
            
            if self.args.model == "FiLM":
                # parse FiLM model param
                gamma_1_rel = manifold_src['model_param_rel'][0][0][b].cpu().detach().numpy()
                beta_1_rel = manifold_src['model_param_rel'][0][1][b].cpu().detach().numpy()
                gamma_2_rel = manifold_src['model_param_rel'][1][0][b].cpu().detach().numpy()
                beta_2_rel = manifold_src['model_param_rel'][1][1][b].cpu().detach().numpy()
                gamma_3_rel = manifold_src['model_param_rel'][2][0][b].cpu().detach().numpy()
                beta_3_rel = manifold_src['model_param_rel'][2][1][b].cpu().detach().numpy()
                gamma_4_rel = manifold_src['model_param_rel'][3][0][b].cpu().detach().numpy()
                beta_4_rel = manifold_src['model_param_rel'][3][1][b].cpu().detach().numpy()
                
                gamma_1_rel = np.expand_dims(gamma_1_rel, axis=0)
                beta_1_rel = np.expand_dims(beta_1_rel, axis=0)
                gamma_2_rel = np.expand_dims(gamma_2_rel, axis=0)
                beta_2_rel = np.expand_dims(beta_2_rel, axis=0)
                gamma_3_rel = np.expand_dims(gamma_3_rel, axis=0)
                beta_3_rel = np.expand_dims(beta_3_rel, axis=0)
                gamma_4_rel = np.expand_dims(gamma_4_rel, axis=0)
                beta_4_rel = np.expand_dims(beta_4_rel, axis=0)
                
                gamma_1_nonrel = manifold_src['model_param_nonrel'][0][0][b].cpu().detach().numpy()
                beta_1_nonrel = manifold_src['model_param_nonrel'][0][1][b].cpu().detach().numpy()
                gamma_2_nonrel = manifold_src['model_param_nonrel'][1][0][b].cpu().detach().numpy()
                beta_2_nonrel = manifold_src['model_param_nonrel'][1][1][b].cpu().detach().numpy()
                gamma_3_nonrel = manifold_src['model_param_nonrel'][2][0][b].cpu().detach().numpy()
                beta_3_nonrel = manifold_src['model_param_nonrel'][2][1][b].cpu().detach().numpy()
                gamma_4_nonrel = manifold_src['model_param_nonrel'][3][0][b].cpu().detach().numpy()
                beta_4_nonrel = manifold_src['model_param_nonrel'][3][1][b].cpu().detach().numpy()
                
                gamma_1_nonrel = np.expand_dims(gamma_1_nonrel, axis=0)
                beta_1_nonrel = np.expand_dims(beta_1_nonrel, axis=0)
                gamma_2_nonrel = np.expand_dims(gamma_2_nonrel, axis=0)
                beta_2_nonrel = np.expand_dims(beta_2_nonrel, axis=0)
                gamma_3_nonrel = np.expand_dims(gamma_3_nonrel, axis=0)
                beta_3_nonrel = np.expand_dims(beta_3_nonrel, axis=0)
                gamma_4_nonrel = np.expand_dims(gamma_4_nonrel, axis=0)
                beta_4_nonrel = np.expand_dims(beta_4_nonrel, axis=0)
                
                # accumulate FiLM model param
                self.manifold_src['gamma_1'].append(gamma_1_rel)
                self.manifold_src['beta_1'].append(beta_1_rel)
                self.manifold_src['gamma_2'].append(gamma_2_rel)
                self.manifold_src['beta_2'].append(beta_2_rel)
                self.manifold_src['gamma_3'].append(gamma_3_rel)
                self.manifold_src['beta_3'].append(beta_3_rel)
                self.manifold_src['gamma_4'].append(gamma_4_rel)
                self.manifold_src['beta_4'].append(beta_4_rel)
                
                self.manifold_src['gamma_1'].append(gamma_1_nonrel)
                self.manifold_src['beta_1'].append(beta_1_nonrel)
                self.manifold_src['gamma_2'].append(gamma_2_nonrel)
                self.manifold_src['beta_2'].append(beta_2_nonrel)
                self.manifold_src['gamma_3'].append(gamma_3_nonrel)
                self.manifold_src['beta_3'].append(beta_3_nonrel)
                self.manifold_src['gamma_4'].append(gamma_4_nonrel)
                self.manifold_src['beta_4'].append(beta_4_nonrel)
                
            elif self.args.model == "RN":
                # parse RN model param
                #
                # accumulate
                #
                print("ERROR: Not supported yet.")
                
        print("Accumulated manifold for batch-size of {}".format(batch_size))
                
            
    def terminate(self):
        # process manifold source
        for k in self.manifold_src:
            self.manifold_src[k] = np.concatenate(self.manifold_src[k], axis=0)
        
        # save manifold source
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.manifold_src, f, pickle.HIGHEST_PROTOCOL)
            
        print("Manifold for {} is saved!".format(self.args.model))
    
    
    
    
    
    
    
    
    
    
    