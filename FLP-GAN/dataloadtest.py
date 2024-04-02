import pickle
# import skip
from data.dataset import FaceTextDataset
from tools.config import cfg, cfg_from_file

# dt = FaceTextDataset(cfg.DATAPATH, 64, cfg.MAXLEN, split="train")
# img, cap, cap_len, masks, landmarks, key, bboxs, segMap = dt[1]
# print(cap)
# print(img[0].numpy().shape)
#
# path = '/nfs/users/wangkun/FLG-GAN/data/数据集/处理后/faces/train/filenames.pickle'
# with open(path, 'rb') as f:
#     dt = pickle.load(f)
# print(len(dt))
# h = pickle.load(open(path, 'rb'))
# face_captions = {}
# for key in h.keys():
#     face_captions[key] = h[key]
#
# training_image_list = [key for key in face_captions]
# print("training image list", training_image_list)

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# import numpy as np
# path = '/nfs/users/wangkun/FLG-GAN/data/数据集/处理后/faces/train/filenames.pickle'
# with open(path, 'rb') as f:
#     dt = pickle.load(f)
# path = '/nfs/users/wangkun/FLG-GAN/data/数据集/处理后/faces/train/char-CNN-RNN-embeddings.pickle'
# with open(path, 'rb') as f:
#     dt_ = pickle.load(f)
#     dt_ = np.array(dt_)
# print(dt, dt_)
# import nltk
# nltk.download('averaged_perceptron_tagger')

import cv2

img = cv2.imread('/nfs/users/wangkun/FLG-GAN/code/ktmp/6/6_fake_segmap.png')
height, width, _ = img.shape

for i in range(height):
    for j in range(width):
        # img[i,j] is the RGB pixel at position (i, j)
        # check if it's [0, 0, 0] and replace with [255, 255, 255] if so
        if img[i, j].sum() == 0:
            img[i, j] = [255, 255, 255]
cv2.imwrite('/nfs/users/wangkun/FLG-GAN/code/ktmp/6/6_fake_segmap_w.png', img)