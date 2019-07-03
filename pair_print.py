import argparse
import os
from pair_dataset import customDataset
from pp_model import PP
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default='./test.json', help="data")
parser.add_argument("--labelroot", type=str, default='./label_saved.json', help="label")
opt = parser.parse_args()

dataset = customDataset(opt.dataroot,opt.labelroot)
