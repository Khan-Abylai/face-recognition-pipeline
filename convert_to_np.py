import numpy as np
import torch
import argparse
import torch
import torch.nn as nn
import numpy as np
from backbones import get_model


parser = argparse.ArgumentParser()
parser.add_argument('--weights_path', type=str, default='./weights/model.pt')
parser.add_argument('--network', type=str, default='r100')
parser.add_argument('--out_path', type=str, default='./weights/model.np')

args = parser.parse_args()
net = get_model(args.network, fp16=False)
net.load_state_dict(torch.load(args.weights_path))
net.eval()
net.cpu()

print(net)

s_dict = net.state_dict()
total = 0
t = 'num_batches_tracked'
np_weights = np.array([], dtype=np.float32)
for k, v in s_dict.items():
    if k[-len(t):] == t:
        continue

    total += v.numel()
    v_reshape = v.reshape(-1)
    np_v = v_reshape.data.numpy()
    np_weights = np.concatenate((np_weights, np_v))

print(total)
print(np_weights.shape)
print(np_weights)
print(np_weights.dtype)

np_weights.tofile(args.out_path)
