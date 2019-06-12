from pp_model import PP
import torch

model = PP()
model.load_state_dict(torch.load('./model-999.pkl'))
model.eval()

inputbb = [540, 575, 644, 726]
inputbb = torch.FloatTensor(inputbb)

out = model(inputbb)
print(out)
