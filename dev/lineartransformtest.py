import torch
import torch.nn as nn
import numpy as np


a = np.array([[.9,.0,.0,.0,.0],
            [.8,.0,.0,.0,.0],
            [.1,.0,.0,.0,.0],
            [.6,.0,.0,.0,.0],
            [.6,.0,.0,.0,.0]])

a = torch.from_numpy(a).float()


shift_embedding = nn.Linear(5,32)

print(shift_embedding(a).shape)