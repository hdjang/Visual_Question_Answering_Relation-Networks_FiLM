import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import os
from torch.nn.init import kaiming_normal, kaiming_uniform
from utils import weight_init


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        chs = args.film_cnn_chs.split(",")
        chs = [int(ch) for ch in chs]
        cnn = []
        ch_i = 3
        for ch_o in chs:
            cnn.append(nn.Conv2d(ch_i, ch_o, 3, stride=2, padding=1, bias=False))
            cnn.append(nn.BatchNorm2d(ch_o))
            cnn.append(nn.ReLU(inplace=True))
            ch_i = ch_o
        self.cnn = nn.Sequential(*cnn)
        
    def forward(self, img):
        return self.cnn(img)


class FiLM_Generator(nn.Module):
    def __init__(self, args):
        super(FiLM_Generator, self).__init__()
        self.args = args
        self.film_gen_dim = 2*args.film_module_dim # film #out-ch per module
        self.linear = nn.Linear(args.qst_dim, self.film_gen_dim * args.film_num_module)
        
    def forward(self, x):
        out = []
        x = self.linear(x) # (N,qst_dim) -> (N,2*film_module_dim*4)
        for i in range(self.args.film_num_module):
            g_idx_s = i * 2 * self.args.film_module_dim
            g_idx_e = g_idx_s + self.args.film_module_dim
            x[:,g_idx_s:g_idx_e] = x[:,g_idx_s:g_idx_e] + 1
            out.append([x[:,g_idx_s:g_idx_e], x[:,g_idx_e:g_idx_e+self.args.film_module_dim]]) # gamma/beta pair for each module
        return out
        

class FiLMed_ResBlock(nn.Module):
    def __init__(self, args):
        super(FiLMed_ResBlock, self).__init__()
        ch = args.film_module_dim
        self.conv1 = nn.Conv2d(ch + 2, ch, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch, affine=False)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x, gamma, beta):
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.film(x2, gamma, beta)
        x2 = self.relu2(x2)
        return x1 + x2
        
    def film(self, x, gamma, beta):
        gamma = gamma.unsqueeze(2).unsqueeze(3).expand_as(x)
        beta = beta.unsqueeze(2).unsqueeze(3).expand_as(x)
        x = (gamma * x) + beta
        return x
        
        
class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        # conv
        conv_chs = args.film_cls_conv_chs.split(",")
        conv_chs = [int(ch) for ch in conv_chs]
        cls_conv = []
        ch_i = args.film_module_dim + 2
        for ch_o in conv_chs:
            cls_conv.append(nn.Conv2d(ch_i, ch_o, args.film_cls_filter_size, bias=False))
            cls_conv.append(nn.BatchNorm2d(ch_o))
            cls_conv.append(nn.ReLU(inplace=True))
            ch_i = ch_o
        self.cls_conv = nn.Sequential(*cls_conv)
        
        # mlp
        mlp_chs = args.film_cls_mlp_chs.split(",") + [str(args.num_cat)]
        cls_mlp = []
        ch_i = conv_chs[-1]
        for ch_o in mlp_chs:
            if ch_o[-1].lower() == "d":
                ch_o = int(ch_o[:-1])
                cls_mlp.append(nn.Linear(ch_i, ch_o))
                if args.use_mlp_bn:
                    rn_g.append(nn.BatchNorm1d(ch_o))
                cls_mlp.append(nn.ReLU(inplace=True))
                cls_mlp.append(nn.Dropout())
                ch_i = ch_o
            else:
                ch_o = int(ch_o)
                cls_mlp.append(nn.Linear(ch_i, ch_o))
                if args.use_mlp_bn:
                    rn_g.append(nn.BatchNorm1d(ch_o))
                cls_mlp.append(nn.ReLU(inplace=True))
                ch_i = ch_o
        self.cls_mlp = nn.Sequential(*cls_mlp)
    
    def forward(self, x):
        x = self.cls_conv(x)
        x = F.max_pool2d(x, kernel_size=x.shape[2:]).squeeze(2).squeeze(2)
        x = self.cls_mlp(x)
        return F.log_softmax(x, dim=1)
        

class FiLMed_Network(nn.Module):
    def __init__(self, args):
        super(FiLMed_Network, self).__init__()
        # FiLMed resblock
        self.film_resblocks = nn.ModuleList([FiLMed_ResBlock(args) for _ in range(args.film_num_module)])
        # classifier
        self.classifier = Classifier(args)
        # positional encoding
        self.pos = self.get_positional_encoding(args) # (1,2,H,W)
          
    def forward(self, cnn_feat, film_params):
        x = cnn_feat
        # filmed resblock
        for module, film_param in zip(self.film_resblocks, film_params):
            x = torch.cat([x, self.pos.repeat(x.shape[0],1,1,1)], dim=1)
            x = module(x, film_param[0], film_param[1])
        # classifier
        x = torch.cat([x, self.pos.repeat(x.shape[0],1,1,1)], dim=1)
        x = self.classifier(x)
        return x
        
    def get_positional_encoding(self, args):
        h = args.cnn_feat_h
        w = args.cnn_feat_w
        x = np.linspace(-1,1,w)
        y = np.linspace(-1,1,h)
        coords_x, coords_y = np.meshgrid(x,y)
        coords_x = np.expand_dims(coords_x, 0)
        coords_y = np.expand_dims(coords_y, 0)
        coords = np.concatenate((coords_x, coords_y), axis=0)
        coords = np.expand_dims(coords, 0)
        coords = torch.from_numpy(coords).float().to(args.device)
        return coords
        
        
class FiLM(nn.Module):
    def __init__(self, args):
        super(FiLM, self).__init__()
        # CNN
        self.cnn = CNN(args)
        # FiLM Generator
        self.film_generator = FiLM_Generator(args)
        # FiLM-ed Network
        self.filmed_network = FiLMed_Network(args)
        # weight initialization
        #initialize(self.modules())
        weight_init(self.modules())
        # model device cfg
        self.to(args.device)
        
    def forward(self, img, qst):
        film_params = self.film_generator(qst)
        cnn_feat = self.cnn(img)
        cls_prob = self.filmed_network(cnn_feat, film_params)
        return cls_prob
    
    def loss(self, pred, target):
        return F.nll_loss(pred, target)
    
    
def initialize(modules, init_type='uniform'):
    if init_type.lower() == 'normal':
        init_params = kaiming_normal
    elif init_type.lower() == 'uniform':
        init_params = kaiming_uniform
    else:
        print("ERROR: not supproted type of initialization.")
        return
    
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            init_params(m.weight)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        