from torch import nn


def freeze_model(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
