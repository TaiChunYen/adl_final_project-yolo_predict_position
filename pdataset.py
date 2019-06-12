from torch.utils.data.dataset import Dataset
import os
import json
import torch
import pdb

class customDataset(Dataset):
    def __init__(self,root,lab_root,transform=None):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        self.root = root
        self.lab_root = lab_root
        self.transform = transform
        self.yolo = []
        self.label = []
        self.len = 0
        #pair = []#

        f = open(self.root, "r")
        jf = json.loads(f.read())
        f2 = open(self.lab_root, "r")
        jf2 = json.loads(f2.read())
        
        for i in range(len(jf)):
            if jf2[jf[i]['filename'][-13:-3]+'png'] != None:
                self.yolo.append(jf[i]['output'][-4:])
                self.label.append(jf2[jf[i]['filename'][-13:-3]+'png'][0])
                #temp=jf[i]#
                #temp['label']=jf2[jf[i]['filename'][-13:-3]+'png'][0]#
                #pair.append(temp)#

        self.len = len(self.yolo)
        self.yolo = torch.FloatTensor(self.yolo)   
        self.label = torch.FloatTensor(self.label)
        #pdb.set_trace()
        #with open('./pair.json','w') as f3:#
            #pdb.set_trace()
            #data_arr=json.dumps(pair)#
            #f3.write(data_arr)#
            #f3.close#

    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        return self.yolo[index], self.label[index]


    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return self.len
