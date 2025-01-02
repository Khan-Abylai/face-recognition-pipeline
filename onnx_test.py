import cv2
import torch
import numpy as np
import onnx
import onnxruntime
from sklearn import preprocessing
import time
def img2tensor(image):
    img  = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float().cuda()
    img.div_(255).sub_(0.5).div_(0.5)
    return img

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_model = onnx.load("weights/model_r100.onnx")
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession("weights/model_r100.onnx")


image = cv2.imread('images/pers2/1.jpg')
img = img2tensor(image)
t1 = time.time()
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
emb1 = ort_session.run(None, ort_inputs)[0] # array
print("onnx inf time:", time.time() - t1)
emb1_n = emb1.reshape(1, -1)

image2 = cv2.imread('images/pers2/2.jpg')
img = img2tensor(image2)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img)}
emb2 = ort_session.run(None, ort_inputs)[0] # array
emb2_n = emb2.reshape(1, -1)

emb1_n = preprocessing.normalize(emb1_n).flatten()
emb2_n = preprocessing.normalize(emb2_n).flatten()
result = np.dot(emb1_n, emb2_n.T)

# similarity_score = np.sum(emb1_n * emb2_n, -1)
# score = similarity_score.flatten()


print("similarity:", result)
