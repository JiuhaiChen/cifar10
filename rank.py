import copy
import numpy as np
import sys
import os
import pickle
import argparse
import itertools
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.transforms.functional as trnF
import torchvision.datasets as dset
import torch.nn.functional as F
from models.wrn_prime import WideResNet
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import random
import pdb
import gc # Garbage collector
import opencv_functional as cv2f
import cv2
import time
from PerturbDataset import PerturbDataset, PerturbDatasetCustom
# import ipdb
# from torch.autograd import Variable

parser = argparse.ArgumentParser(
    description='Test OOD with one classifier network. Metric used is the maximum softmax probability',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

# Test setup
parser.add_argument('--test_bs', type=int, default=500)
parser.add_argument('--in_dataset', choices=["CIFAR10", "CIFAR100"], default="CIFAR10")
parser.add_argument('--out_dataset', choices=["CIFAR10", "CIFAR100"], default="CIFAR100")
parser.add_argument('--prefetch', type=int, default=10, help='Pre-fetching threads.')

# Loading details
parser.add_argument('--architecture', type=str, default='wrn', choices=['wrn'], help='Choose architecture.')
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')
parser.add_argument('--model', '-l', type=str, required=True, help='Trained PyTorch model')
parser.add_argument('--vanilla', action='store_true')
parser.add_argument('--nchannel', default=128, type=int, help='number of last channel')

# Test-time training
parser.add_argument('--test-time-train', action='store_true')
parser.add_argument('--test-epochs', type=int, default=None)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(state)


def main():

    # Load the model
    num_class = 10
    net = None
    net = WideResNet(args.layers, num_class, args.widen_factor, dropRate=args.droprate)
  
    net.x_trans_head = nn.Linear(128, 3)
    net.y_trans_head = nn.Linear(128, 3)
    net.rot_head = nn.Linear(128, 4)
    
    if os.path.isfile(args.model):
        # net.load_state_dict(torch.load(args.model))
        net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.model).items()})
    else:
        raise Exception("Cannot find {0}".format(args.model))

    in_data = PerturbDataset(dset.CIFAR10('/home/jiuhai.chen/data', train=False, download=True), train_mode=False)

    train_loader_in = torch.utils.data.DataLoader(
    in_data,
    batch_size=args.test_bs,
    shuffle=False,
    num_workers=args.prefetch,
    pin_memory=False
    )
  

  
    Jacobian = []
    for x_tf_0, _, _, _, _, _, _, _ in tqdm(train_loader_in):
        
        batch_size = x_tf_0.shape[0]
        # x_tf_0 = x_tf_0.clone().detach().requires_grad_(True)
        x_tf_0 = x_tf_0.requires_grad_(True)
        logits, pen = net(x_tf_0)
        dims = pen.shape[1]
        # classification_smax = F.softmax(logits[:batch_size], dim=1)
        Jacobian_batch = torch.zeros(batch_size, 3, 32, 32, dims)

        for i in range(dims):
            grad_tensor = torch.zeros(pen.size())
            grad_tensor[:, i] = 1
            # logits.backward(grad_tensor, retain_graph=True)
            pen.backward(grad_tensor, retain_graph=True)
            with torch.no_grad():
                Jacobian_batch[:, :, :, :, i] = x_tf_0.grad.detach()

        Jacobian.append(Jacobian_batch)

    Jacobian = torch.cat(Jacobian, dim=0)
    torch.save(Jacobian, 'Jacobian_rot.pt')
    Jacobian = Jacobian.numpy()
    Jacobian_mean = np.mean(Jacobian, axis=0)
    print(np.linalg.matrix_rank(Jacobian_mean))
    

  
             





    

    

 
    


   

    

if __name__ == "__main__":
    main()
