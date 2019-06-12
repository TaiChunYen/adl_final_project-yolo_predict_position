import torch

class PP(torch.nn.Module):

    def __init__(self):
        super(PP, self).__init__()
        self.input = torch.nn.Linear(4,512)
        self.pp = torch.nn.Linear(512,512)
        self.output = torch.nn.Linear(512,3)

    def forward(self, bbox):
        out = self.input(bbox)
        out2 = self.pp(out)
        out3 = self.output(out2)

        return out3
