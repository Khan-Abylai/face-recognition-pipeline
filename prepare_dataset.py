import os 
from glob import glob
import shutil

path1 = "/home/yeleussinova/data_SSD/face_recognition/dataset/combined_images"
path2 = "/home/yeleussinova/data_SSD/face_recognition/dataset/faces_megafacetrain_112x112/out_images"
out_path = "/home/yeleussinova/data_SSD/face_recognition/dataset/combined_images"

print("files: ", len(glob(os.path.join(path1, "*/*"))))
# file1 = os.listdir(path1)
# file2 = os.listdir(path2)
#
# file_n = [int(x) for x in file1]
# max_id = max(file_n)
# print(max_id)

# for x, y in zip(range(113221+1, 113221+1+len(file2)), file2):
#     print(x, y)
#     new_path = os.path.join(out_path, str(x).zfill(6))
#     files_to_copy = glob(os.path.join(path2, y, "*"))
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)
#     for file_ in files_to_copy:
#         shutil.copy(file_, new_path)

# folders = os.listdir(out_path)
# for folder in folders:
#     new_path = os.path.join(out_path, folder)
#     files = glob(os.path.join(path2, folder, "*"))
#     if not os.path.exists(new_path):
#         os.makedirs(new_path)
#     for f in files:
#         shutil.copy(f, new_path)
#
# print(len(file1) + len(file2))
#
# print(len(glob("/home/yeleussinova/data_SSD/face_recognition/dataset/combined_images/*/*")))