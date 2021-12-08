import torch
from torch.nn.functional import softmax

def CrossEntropyLoss(prediction, label):
    softmax = torch.nn.Softmax(dim=1)
    return torch.log(softmax(prediction)[0][label])*torch.tensor(-1)