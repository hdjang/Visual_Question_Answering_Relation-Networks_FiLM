import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import pdb
import os
from utils import weight_init




class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        chs = args.cnn_chs.split(",")
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
    
    
class RN_G(nn.Module):
    def __init__(self, args):
        super(RN_G, self).__init__()
        chs = args.rn_g_chs.split(",")
        rn_g = []
        ch_i = (int(args.cnn_chs.split(",")[-1])+2)*2+args.qst_dim
        for ch_o in chs:
            if ch_o[-1].lower() == "d":
                ch_o = int(ch_o[:-1])
                rn_g.append(nn.Linear(ch_i, ch_o))
                if args.use_mlp_bn:
                    rn_g.append(nn.BatchNorm1d(ch_o))
                rn_g.append(nn.ReLU(inplace=True))
                rn_g.append(nn.Dropout())
                ch_i = ch_o
            else:
                ch_o = int(ch_o)
                rn_g.append(nn.Linear(ch_i, ch_o))
                if args.use_mlp_bn:
                    rn_g.append(nn.BatchNorm1d(ch_o))
                rn_g.append(nn.ReLU(inplace=True))
                ch_i = ch_o
        self.rn_g = nn.Sequential(*rn_g)
    
    def forward(self, x):
        return self.rn_g(x)

    
class RN_F(nn.Module):
    def __init__(self, args):
        super(RN_F, self).__init__()
        chs = args.rn_f_chs.split(",")
        rn_f = []
        ch_i = args.rn_g_chs.split(",")[-1]
        ch_i = int(ch_i[:-1]) if ch_i[-1]=="d" else int(ch_i)
        for ch_o in chs:
            if ch_o[-1].lower() == "d":
                ch_o = int(ch_o[:-1])
                rn_f.append(nn.Linear(ch_i, ch_o))
                if args.use_mlp_bn:
                    rn_f.append(nn.BatchNorm1d(ch_o))
                rn_f.append(nn.ReLU(inplace=True))
                rn_f.append(nn.Dropout())
                ch_i = ch_o
            else:
                ch_o = int(ch_o)
                rn_f.append(nn.Linear(ch_i, ch_o))
                if args.use_mlp_bn:
                    rn_f.append(nn.BatchNorm1d(ch_o))
                rn_f.append(nn.ReLU(inplace=True))
                ch_i = ch_o
        self.rn_f = nn.Sequential(*rn_f)
        
    def forward(self, x):
        return self.rn_f(x)
    
    
class RefFinder(nn.Module):
    def __init__(self, args):
        super(RefFinder, self).__init__()
        chs = args.cnn_chs.split(",")
        chs = [int(ch) for ch in chs]
        ch_i = chs[-1]
        self.finder = nn.Sequential(
            nn.Conv2d(ch_i+args.qst_dim, 1, 1, stride=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )
            
    def forward(self, cnn_feat, qst):
        
        # get soft-attention
        x = torch.cat([cnn_feat, qst.unsqueeze(2).unsqueeze(2).repeat(1,1,cnn_feat.shape[2],cnn_feat.shape[3])], dim=1)
        x = self.finder(x)
        N,C,H,W = x.shape
        x = x.view(N,-1)
        x = F.softmax(x, dim=1)
        x = x.view(N,C,H,W)
        
        # get max-index
        ref_idx = torch.max(x.view(N,-1), dim=1)[1] # (N)
        
        # get reference point
        ref = (cnn_feat * x)
        ref = ref.view(N,ref.shape[1],-1)
        ref = ref.sum(dim=2)
        
        return ref, ref_idx
        
    
class RN(nn.Module):
    def __init__(self, args):
        super(RN, self).__init__()
        self.args = args
        self.cnn = CNN(args) # (N,C,H,W)
        self.pos = self.get_positional_encoding(args) # (1,2,H,W)
        self.rn_g = RN_G(args)
        self.rn_f = RN_F(args)
        cls_ch = args.rn_f_chs.split(",")[-1]
        cls_ch = int(cls_ch[:-1]) if cls_ch[-1].lower() == "d" else int(cls_ch)
        self.classifier = nn.Linear(cls_ch, args.num_cat)
        if args.rn_extension:
            self.ref_finder = RefFinder(args)
        
        weight_init(self.modules())
        self.to(args.device)
        
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

    def forward(self, img, qst, debug=False):
        # img encoding w/ positional encoding
        x = self.cnn(img)
            
        if self.args.rn_extension:
            ref, ref_idx = self.ref_finder(x, qst)  # (N,C), (N)
            
        x = torch.cat([x, self.pos.repeat(x.shape[0],1,1,1)], dim=1)
        N,C,H,W = x.shape
        
        if self.args.rn_extension:
            # ref point w/ positional encoding
            x_pos = self.pos[:,0,:,:].view(-1)[ref_idx].unsqueeze(1)
            y_pos = self.pos[:,1,:,:].view(-1)[ref_idx].unsqueeze(1)
            ref = torch.cat([ref, x_pos, y_pos], dim=1)
            
            # edge generation
            ref = ref.unsqueeze(2).unsqueeze(2).repeat(1,1,x.shape[2],x.shape[3])
            x = torch.cat([ref, x], dim=1)
            x = x.view(N,x.shape[1],-1).transpose(2,1)
            
        else:
            # edge generation
            x = x.permute(0,2,3,1)
            x = x.view(N,-1,C) # (N,HW,C)

            x_i = x.unsqueeze(2) # (N,HW,1,C)
            x_i = x_i.repeat(1,1,H*W,1) # (N,HW,HW,C); each-cell

            x_j = x.unsqueeze(1) # (N,1,HW,C)
            x_j = x_j.repeat(1,H*W,1,1) # (N,HW,HW,C); all-cell

            x = torch.cat([x_i, x_j], dim=3)
            x = x.view(N,-1,C*2)
        
        # edge conditioning w/ question
        qst = qst.unsqueeze(1)
        qst = qst.repeat(1,x.shape[1],1)
        
        x = torch.cat([x, qst], dim=2)
        
        # RN - edge encoding
        x = x.view(-1, x.shape[-1])
        x = self.rn_g(x)
        x = x.view(N,-1,x.shape[-1])
        
        # RN - aggregate edge encoding
        x = x.sum(dim=1)
        
        # RN - graph encoding
        x = self.rn_f(x)
        
        # classifier
        x = self.classifier(x)
        cls_prob = F.log_softmax(x, dim=1)
        
        if debug:
            return cls_prob, rn_params
        else:
            return cls_prob, None
        
    def loss(self, pred, target):
        return F.nll_loss(pred, target)    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        