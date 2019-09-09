from __future__ import print_function
import argparse
import os
import pickle
import random
import numpy as np
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from dataset import get_dataset
from opts import parse_opts
from networks.RN import RN
from networks.FiLM import FiLM
from utils import Manifold_handler
from utils import Timer
import pdb


def eval_accuracy(pred, ans, sum_only=False):
    ans_ = pred.data.max(dim=1)[1]
    hit_sum = ans_.eq(ans).sum()
    if sum_only:
        return hit_sum
    else:
        return hit_sum * 100.0 / len(ans)
        
        
def train(args, model, train_loader, val_loader):
    
    model.train()
    
    # optimizer
    learning_rate = args.lr
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print("Optimizer: ", optimizer)
    
    
    # LR scheduling
    if args.multi_step:
        milestones = [int(s) for s in args.multi_step.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.lr_gamma)
    
    
    # exp directory
    if args.phase == 'train':
        exp_root = "./exp"
        exp_path = os.path.join(exp_root, args.exp_id)
        try:
            os.makedirs(exp_path)
        except:
            print('directory {} already exists'.format(exp_path))
    
    # resume model
    if args.resume_ckpt:
        checkpoint = torch.load(args.resume_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_epoch = checkpoint['epoch'] + 1
        print("\nModel resumed from {} epoch.\n".format(resume_epoch-1))
    else:
        resume_epoch = 1
    
    train_t0 = time.time()
    
    num_gpu = torch.cuda.device_count()
    iteration = 0
    for epoch in range(resume_epoch, args.epochs+1):
        # decay LR
        if args.lr_max > 0 and scheduler.get_lr()[0] < args.lr_max:
            scheduler.step()
        
        # train for a epoch
        for i, data in enumerate(train_loader):
            iteration += 1
            
            t0 = time.time()
            img, qst, ans = data
            if args.cuda:
                img = img.to(args.device)
                qst = qst.to(args.device)
                ans = ans.to(args.device)
            
            pred, _ = model(img, qst)
            if num_gpu > 1:
                loss = model.module.loss(pred, ans)
            else:
                loss = model.loss(pred, ans)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t1 = time.time()
            
            train_acc = eval_accuracy(pred, ans)
            if iteration % args.log_interval == 0:
                print('[train] iter: {} | epoch:{} [{}/{}] Loss: {:.4f} | VQA-Acc: {:.0f}% | LR: {} | time: {:.4f}'.format(iteration, epoch, i+1, len(train_loader), loss, train_acc, scheduler.get_lr(), t1-t0))
        
        # save checkpoint every epoch
        net = model.module if num_gpu > 1 else model
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(exp_path, args.model+"_epoch_{:02d}.pth".format(epoch)))
        print("model at {}-epoch is saved".format(epoch))
        
        # test every epoch
        if not args.skip_train_eval:
            test(args, model, val_loader)
    
    train_t1 = time.time()
    train_time = train_t1 - train_t0
    print("Total training time: {:.2f} min".format(train_time/60))
    
    return model


def test(args, model, val_loader):
    model.eval()
    
    # timer settup
    _t = {'timer': Timer(), 'misc': Timer()}
    
    # extracting tSNE source
    if args.phase == "test" and args.extract_manifold_source:
        manifold_handler = Manifold_handler(args)
        debug = True
    else:
        manifold_handler = None
        debug = False
        
    # evaluation
    acc_sum_rel = 0
    acc_sum_nonrel = 0
    acc_len_rel = 0
    acc_len_nonrel = 0
    for iteration, data in enumerate(val_loader):
        
        iteration += 1

        img_rel = data[0][0]
        qst_rel = data[1][0]
        ans_rel = data[2][0]
        img_nonrel = data[0][1]
        qst_nonrel = data[1][1]
        ans_nonrel = data[2][1]
        if args.cuda:
            img_rel = img_rel.to(args.device)
            qst_rel = qst_rel.to(args.device)
            ans_rel = ans_rel.to(args.device)
            img_nonrel = img_nonrel.to(args.device)
            qst_nonrel = qst_nonrel.to(args.device)
            ans_nonrel = ans_nonrel.to(args.device)
        
        _t['timer'].tic()
        
        pred_rel, model_param_rel = model(img_rel, qst_rel, debug)
        pred_nonrel, model_param_nonrel = model(img_nonrel, qst_nonrel, debug)
        
        forward_time = _t['timer'].toc()
        
        acc_sum_rel += eval_accuracy(pred_rel, ans_rel, sum_only=True)
        acc_sum_nonrel += eval_accuracy(pred_nonrel, ans_nonrel, sum_only=True)
        acc_len_rel += pred_rel.shape[0]
        acc_len_nonrel += pred_nonrel.shape[0]
        
        
        # manifold source accumulate
        if manifold_handler:
            manifold_src = {
                "qst_rel":qst_rel,
                "qst_nonrel":qst_nonrel,
                "model_param_rel":model_param_rel,
                "model_param_nonrel":model_param_nonrel,
                }
            manifold_handler.accumulate(manifold_src)
            
        print("[val] iter: {} | batch-size: {} | time: {:.5f}".format(iteration, args.batch_size, forward_time))
        
    val_acc_rel = acc_sum_rel * 100.0 / acc_len_rel
    val_acc_nonrel = acc_sum_nonrel * 100.0 / acc_len_nonrel
    print('\n[val] Rel-Acc: {:.2f}% | Non-Rel-Acc: {:.2f} | time: {:.5f} %\n'.format(val_acc_rel, val_acc_nonrel, forward_time))
    
    # manifold source save
    if manifold_handler:
        manifold_handler.terminate()

def main(args):
    
    # device setting
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # build model
    if args.model == "RN":
        model = RN(args)
    elif args.model == "FiLM":
        model = FiLM(args)
    print('\n',model)
    print('\nNumber of model parameters: {}'.format(sum([ p.data.nelement() for p in model.parameters()])))
    print('\n',args,'\n')
    num_gpu = torch.cuda.device_count()
    if num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu)))
    print("Model is loaded on {} GPUs.".format(num_gpu))
    
    # dataset
    train_set, val_set = get_dataset(args)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # train / test
    if args.phase == "train":
        model = train(args, model, train_loader, val_loader)
        print("Training is done!")
    elif args.phase == "test":
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("model loaded from a checkpoint: {}".format(args.checkpoint))
            test(args, model, val_loader)
            print("Testing is done!")
        else:
            print("ERROR: checkpoint does not exist!")
            return
    else:
        print("ERROR: wrong phase entered.")
        return
    
if __name__ == '__main__':
    args = parse_opts()
    main(args)