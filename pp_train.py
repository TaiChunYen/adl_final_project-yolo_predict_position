import argparse
import os
from pdataset import customDataset
from pp_model import PP
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--dataroot", type=str, default='./test.json', help="data")
parser.add_argument("--labelroot", type=str, default='./label_saved.json', help="label")
opt = parser.parse_args()

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

dataset = customDataset(opt.dataroot,opt.labelroot)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.n_cpu)

model = PP().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss()

count = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        outputs=model(imgs.cuda())
        loss = criterion(outputs, labels.cuda())

        optimizer.zero_grad()
        print(i, loss.item())
        loss.backward()
        optimizer.step()
        
    count+=1        
    if count>=50:
        torch.save(model.state_dict(),'./model-%s.pkl'%str(epoch))
        count=0

torch.save(model.state_dict(),'./model-last.pkl')


















