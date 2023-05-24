import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm

def process_HE(out):
    logpx = F.log_softmax(out, dim=1).tolist()
    logpx = np.array(logpx)
    out_HE = torch.tensor(-np.sum(np.multiply(logpx, np.exp(logpx)), axis = 1)).unsqueeze(1)
    return out_HE

