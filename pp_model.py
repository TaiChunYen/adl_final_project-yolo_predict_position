import torch
import torch.nn.functional as F

class PP(torch.nn.Module):

    def __init__(self):
        super(PP, self).__init__()
        self.input = torch.nn.Linear(4,512)
        self.pp = torch.nn.Linear(512,512)
        self.output = torch.nn.Linear(512,3)

    def forward(self, bbox):
        out = F.relu(self.input(bbox))
        out2 = F.relu(self.pp(out))
        out3 = self.output(out2)

        return out3
