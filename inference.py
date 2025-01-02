import argparse
import time

import cv2
import numpy as np
import torch
from backbones import get_model
from sklearn import preprocessing
from numpy.linalg import norm

@torch.no_grad()
def inference(weight, name, img):

    img = cv2.imread(img)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval().cpu()
    feat = net(img).numpy()
    # print(feat)
    stop = 1
    return feat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='/face_data/out_combined/model.pt')
    args = parser.parse_args()
    img_path1 = "images/arm2_aligned.jpg"
    img_path2 = "images/arm1_aligned.jpg"
    t1 = time.time()
    emb1 = inference(args.weight, args.network, img_path1)
    print("torch inf time:", time.time() - t1)

    emb2 = inference(args.weight, args.network, img_path2)
    emb1_n = emb1.reshape(1, -1)
    emb2_n = emb2.reshape(1, -1)
    emb1_n = preprocessing.normalize(emb1_n).flatten()
    emb2_n = preprocessing.normalize(emb2_n).flatten()
    result = np.dot(emb1_n, emb2_n.T)

    print("similarity:", result)