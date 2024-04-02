import os
import pickle
import time
import numpy as np
from PIL.Image import Image
import torch
from torch.autograd import Variable
import argparse
from torch.utils.data import DataLoader
from PIL import Image
import multiprocessing
import logging
import nltk
from gensim.models import KeyedVectors
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import cv2

nltk.download('stopwords')
from data.dataset import FaceTextDataset
from models.encoders import RNN_ENCODER, FACE_ENCODER
from models.netG import G_NET
from tools.config import cfg, cfg_from_file
from models.LMGan import LMGen
from tools.losses import words_loss
from tools.visualizations import build_roiattn_images, editShapes, bSplineAndSeg

model = None


def main():
    parser = argparse.ArgumentParser("FLG-GAN")
    parser.add_argument(
        "-gpu_id",
        type=str,
        default="2",
        dest="gpu_id"
    )
    parser.add_argument(
        "-cfg",
        default="./cfgs/faceVal.yml",
        dest="cfg"
    )
    parser.add_argument(
        "-sample",
        type=bool,
        default=True,
        dest="sample"
    )
    parser.add_argument(
        "-shape",
        type=bool,
        default=False,
        dest="shape"
    )
    args = parser.parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger("test")
    logger.info(cfg)
    cfg.LOG = logger

    testDataset = FaceTextDataset(cfg.DATAPATH, cfg.TREE.BASE_SIZE, cfg.MAXLEN, split="train")
    testDataloader = DataLoader(testDataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=8)
    print(len(testDataset))
    # idx2word, word2idx = loadDict(os.path.join(cfg.DATAPATH, "captions.pickle"))
    idx2word, word2idx = testDataset.idx2word, testDataset.word2idx
    vocab_size = len(idx2word)

    # load models
    TextEncoder = RNN_ENCODER(vocab_size)
    state_dict = torch.load(cfg.ENCODER_PATH, map_location=lambda storage, loc: storage)
    print('Text', cfg.ENCODER_PATH)
    TextEncoder.load_state_dict(state_dict)
    TextEncoder = TextEncoder.to(cfg.device)
    TextEncoder.eval()
    print('test-TextEncoder load finished')
    caption_vectors = {}
    for step, data in enumerate(testDataloader, 0):
        start = time.time()
        imgs, caps, cap_lens, masks, landmarks, keys, bboxs, segMaps = data
        batch_size = masks.shape[0]
        noise = Variable(torch.FloatTensor(batch_size, cfg.MODEL.Z_DIM)).to(cfg.device)
        cap_lens, sorted_cap_indices = \
            torch.sort(cap_lens, 0, True)
        caps = Variable(caps[sorted_cap_indices].squeeze()).to(cfg.device)
        hidden = TextEncoder.init_hidden(batch_size)
        words_embs, sent_emb = TextEncoder(caps[:, 1:], cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        # print(keys[0], sent_emb[0].size())
        for i in range(batch_size):
            caption_vectors[keys[i]] = sent_emb[i].cpu().numpy()
    with open('/nfs/users/wangkun/FLG-GAN/data/数据集/处理后/faces/caption_vectors/trn_caption_vectors.pickle', 'wb') as f:
        pickle.dump(caption_vectors, f)





def loadDict(filepath):
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
        trn_captions, test_captions = x[0], x[1]
        idx2word, word2idx = x[2], x[3]
        del x
    return idx2word, word2idx


def cap2tensor(cap, idx2Mask, word2idx):
    cap = tokenize(cap)
    cap_new = []
    for word in cap:
        idx = word2idx.get(word, -1)
        if idx == -1:
            print("word not in vocab: " + word)
            continue
        cap_new.append(idx)
    if len(cap_new) <= 0:
        print("empty caption")
        return None, None, None
    maxLen = cfg.MAXLEN - 1
    x = np.zeros(maxLen, dtype='int64')
    x_len = len(cap_new)
    num_words = x_len
    if num_words <= maxLen:
        x[:num_words] = cap_new
    else:
        cap_new = np.asarray(cap_new).astype('int64')
        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:maxLen]
        ix = np.sort(ix)
        x = cap_new[ix]
        x_len = maxLen
    masks = [idx2Mask[_x] for _x in x]
    masks = np.asarray(masks).astype('int64')
    cap = torch.from_numpy(x)
    masks = torch.from_numpy(masks)
    return cap, masks, torch.tensor([x_len])


def tokenize(cap):
    cap = cap.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cap.lower())
    if len(tokens) == 0:
        print(cap)
        print("no tokens")
    tokens_new = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0:
            tokens_new.append(t)
    return tokens_new


def getIdx2Masks(word2idx, idx2word):
    idx2Mask = [0] * len(word2idx)
    idx2Mask[0] = 1
    stop_words = set(stopwords.words('english'))
    for word in stop_words:
        if word2idx.get(word, None) is not None:
            idx2Mask[word2idx[word]] = 1
    return idx2Mask


if __name__ == "__main__":
    main()
