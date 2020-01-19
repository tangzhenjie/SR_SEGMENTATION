import torch.nn as nn
import torch
input = torch.randn(50, 5)
target = torch.ones(50, 1)
tep = target.view(-1, 1).repeat(1, 5)
tep1 = tep >= 2
log_p = input[tep1]
i = 0

# 直接去掉尾数