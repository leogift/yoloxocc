import torch.nn as nn
import math

def initialize_regression_weights(module, std=1e-1):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_classification_weights(module, prior_prob=1e-2):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                b = m.bias.view(1, -1)
                b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
                m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
