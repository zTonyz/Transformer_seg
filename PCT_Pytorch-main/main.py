from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset import PsbDataset
from model import Pct_seg
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

import time 

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args,io):
    train_loader = DataLoader(PsbDataset(partition='train',data_path=args.data_path,label_path=args.label_path,index=args.index),batch_size=args.batch_size,num_workers=0)
    test_loader = DataLoader(PsbDataset(partition='test',data_path=args.data_path,label_path=args.label_path,index=args.index),batch_size=args.batch_size,num_workers=0)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct_seg(out_c=args.out_c).to(device)
    if args.pre_train:
        model.load_state_dict(torch.load(args.model_path))
        print('load model successfully')
        #print(str(model))
        #model = nn.DataParallel(model)

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*20, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = torch.nn.CrossEntropyLoss()
    best_test_acc = 0

    for epoch in range(args.epochs):
        scheduler.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        #idx = 0
        total_time = 0.0
        correct=0.0
        num=0
        for data,label in train_loader:
            data, label = data.to(device), label.to(device)
            #print('data:',data.shape) #(2,3000,628)
            #print('label:',label.shape) #(2,3000)


            opt.zero_grad()
            start_time = time.time()
            pred = model(data)
            #print('pred',pred.shape) #(2,3000,8)
            pred=pred.permute(0,2,1)
            loss = criterion(pred, label)

            loss.backward()
            opt.step()
            end_time = time.time()
            total_time += (end_time - start_time)
            #print('pred',pred.shape)
            preds = pred.max(dim=1)[1]
            #print('preds',preds.shape) #(2,3000)
            count += 1*args.batch_size
            train_loss += loss.item()*args.batch_size
            num+=data.shape[0]*data.shape[1]
            correct+=(preds==label).sum().item()
            #print(correct)
            #train_true.append(label.cpu().numpy())
            #train_pred.append(preds.detach().cpu().numpy())

            
        #print ('train total time is',total_time)
        #train_true = np.concatenate(train_true)
        #train_pred = np.concatenate(train_pred)
        test_acc = correct*1.0/num
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch,
                                                            train_loss*1.0/count,

                                                            test_acc )
        io.cprint(outstr)
        '''
        if test_acc >= best_test_acc:
                best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.pth' % args.exp_name)
        '''
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        correct=0.0
        num=0
        total_time = 0.0
        with torch.no_grad():
            for data,label in test_loader:
                data, label = data.to(device), label.to(device)
                # print(idx,type(idx)) tensor
                # print(data.shape) (1,9408,628)

                opt.zero_grad()
                start_time = time.time()
                pred = model(data)
                #print(pred.shape)
                pred=pred.permute(0,2,1)
                #end_time = time.time()
                #total_time += (end_time - start_time)
                #loss = criterion(pred, label)
                preds = pred.max(dim=1)[1]

                #test_loss+=loss
                #count += 1*args.batch_size

                num+=data.shape[0]*data.shape[1]
                correct+=(preds==label).sum().item()
            #print ('test total time is', total_time)
            #test_true = np.concatenate(test_true)
            #test_pred = np.concatenate(test_pred)
            test_acc = correct*1.0/num
            #avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test %d, test acc: %.6f' % (epoch,test_acc)
            io.cprint(outstr)
            
            '''
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.pth' % args.exp_name)
            '''
    torch.save(model.state_dict(), 'checkpoints/%s/models/model.pth' % args.exp_name)

def test(args, io):
    train_loader = DataLoader(
        PsbDataset(partition='train', data_path=args.data_path, label_path=args.label_path, index=args.index),
        batch_size=1, num_workers=1)
    test_loader = DataLoader(
        PsbDataset(partition='test', data_path=args.data_path, label_path=args.label_path, index=args.index),
        batch_size=1, num_workers=1)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct_seg(out_c=args.out_c).to(device)
    model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1] 
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--out_c', type=int, default=8,
                        help='classes of this shape')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--label_path', type=str, default='E:/transf_mesh/PSB_new/seg_consistent/',
                        help='path of label')
    parser.add_argument('--data_path', type=str, default='E:/transf_mesh/features/',
                        help='path of dataset')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--index', type=int, default=2,
                        help='which class to train')
    parser.add_argument('--model_path', type=str, default='E:/transf_mesh/PCT_Pytorch-main/checkpoints/exp/models/model.pth', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--pre_train', type=bool, default=False,
                        help='use Pretrained model')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        #torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)