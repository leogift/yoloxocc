import torch.nn as nn
import math

def initialize_weights(module, prior_prob=1e-2):
    for _, m in module.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.zeros_(m.bias)
            m.eps = 1e-3
            m.momentum = 0.03
