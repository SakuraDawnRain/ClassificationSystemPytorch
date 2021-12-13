import torch
from torch.nn.functional import softmax

def CrossEntropyLoss(prediction, label):
    softmax = torch.nn.Softmax(dim=1)
    # return torch.log(softmax(prediction)[0][label])*torch.tensor(-1)
    return torch.sum(torch.mul(torch.log(softmax(prediction).view(-1)[label]), torch.tensor(-1)))

def KappaLoss(prediction, label):
    pass

def QWK(matrix):
    pass