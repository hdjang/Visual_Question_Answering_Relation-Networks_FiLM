from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import pdb


def parse_opts():
    parser = argparse.ArgumentParser(description='relation-reasoning')
    parser.add_argument('--model', type=str, choices=['RN', 'FiLM'], 
                        help='model type')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay for optimizer')
    parser.add_argument('--optimizer', default='adam', type=str,
                        help='optimizer for training')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum for SGD')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--step_size', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr_gamma', type=float, default=1,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--lr_max', type=float, default=0.0005,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--cnn_feat_h', type=int, default=5,
                        help='resume from model stored')
    parser.add_argument('--cnn_feat_w', type=int, default=5,
                        help='resume from model stored')
    
    #--------------------------------------------------------------------------
    # RN cfg
    #--------------------------------------------------------------------------
    parser.add_argument('--exp_id', type=str,
                        help='experiment id')
    parser.add_argument('--data_dir', type=str, default='./data/sort-of-clevr.pickle',
                        help='dataset directory')
    parser.add_argument('--num_workers', type=int, default=20, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--phase', type=str,
                        help='input batch size for training (default: 64)')
    
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_cat', type=int, default=10,
                        help='random seed (default: 1)')
    parser.add_argument('--multi_step', type=str, default=None,
                        help='random seed (default: 1)')
    parser.add_argument('--resume_ckpt', type=str, default=None,
                        help='random seed (default: 1)')
    
    parser.add_argument('--use_mlp_bn', action="store_true", default=False,
                        help='random seed (default: 1)')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='resume from model stored')
    parser.add_argument('--device', type=str, default=None,
                        help='resume from model stored')
    parser.add_argument('--qst_dim', type=int, default=11,
                        help='resume from model stored')
    parser.add_argument('--cnn_chs', type=str, default="32,64,128,256",
                        help='resume from model stored')
    parser.add_argument('--rn_g_chs', type=str, default="2000,2000,2000,2000",
                        help='resume from model stored')
    parser.add_argument('--rn_f_chs', type=str, default="2000,1000,500,100",
                        help='resume from model stored')
    parser.add_argument('--no_img_norm', action='store_true', default=False,
                        help='resume from model stored')
    
    
    #--------------------------------------------------------------------------
    # FiLM cfg
    #--------------------------------------------------------------------------
    parser.add_argument('--film_cls_conv_chs', type=str, default="512",
                        help='momentum for SGD')
    parser.add_argument('--film_cls_mlp_chs', type=str, default="1024",
                        help='momentum for SGD')
    parser.add_argument('--film_cls_filter_size', type=int, default=1,
                        help='momentum for SGD')
    parser.add_argument('--film_num_module', type=int, default=4,
                        help='momentum for SGD')
    parser.add_argument('--film_module_dim', type=int, default=128,
                        help='momentum for SGD')
    parser.add_argument('--film_cnn_chs', type=str, default="128,128,128,128",
                        help='momentum for SGD')
    
    
    
    
    
    
    
    return parser.parse_args()