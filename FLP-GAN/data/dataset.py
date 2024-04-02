from operator import index
import cv2
from numpy.lib.shape_base import split
from torch._C import dtype
from torch.utils import data
import os
import math
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from PIL import Image, ImageOps
from torchvision import transforms
import numpy as np
import torch

from config import cfg

FACIAL_LANDMARKS_68_IDXS_FLIP = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 27, 26, 25,
                                 24, 23, 22, 21, 20, 19, 18, 28, 29, 30, 31, 36, 35, 34, 33, 32, 46, 45, 44, 43,
                                 48, 47, 40, 39, 38, 37, 42, 41, 55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56,
                                 65, 64, 63, 62, 61, 68, 67, 66]


class FaceTextDataset(data.Dataset):
    def __init__(self, data_dir, imsize, num_words, is4gen=False, split="train"):
        self.data_dir = data_dir
        self.split = split
        self.imsize = imsize
        self.is4gen = is4gen
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.embeddings_num = cfg.EMBEDDING_NUM
        self.num_words = num_words
        self.filenames, self.captions, self.idx2word, self.word2idx, self.vocab_size, self.files4gen = \
            self.loadData(data_dir, split)
        self.landmarks, self.genIdx = \
            self.load_landmarks(data_dir, self.filenames, cfg.MAXSHIFT)
        self.facial_bboxs = self.load_bboxs(data_dir, self.filenames)
        print(len(self.genIdx))
        self.idx2Mask = self.getIdx2Masks(self.word2idx)

    # stop words mask
    def getIdx2Masks(self, word2idx):
        idx2Mask = [0] * len(word2idx)
        idx2Mask[0] = 1
        stop_words = set(stopwords.words('english'))
        for word in stop_words:
            if word2idx.get(word, None) is not None:
                idx2Mask[word2idx[word]] = 1

        return idx2Mask

    # split words
    def loadCaption(self, filenames):
        all_captions = []
        for filename in filenames:
            cap_path = os.path.join(self.data_dir, "text", filename + ".txt")
            with open(cap_path, "r") as f:
                captions = f.readlines()
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    if len(tokens) == 0:
                        print(cap)
                        # cfg.LOG.info("len = 0", cap)
                        continue
                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    all_captions.append(all_captions[-1])
        return all_captions

    # load raw caption / pickle file
    def loadData(self, data_dir, split):
        trn_filenames = self.load_filenames(data_dir, "train")
        test_filenames = self.load_filenames(data_dir, "test")
        filepath = os.path.join(data_dir, "captions.pickle")
        if not os.path.isfile(filepath):  ## first time, parse and save as pickle
            trn_captions = self.loadCaption(trn_filenames)
            test_captions = self.loadCaption(test_filenames)
            trn_captions, test_captions, idx2word, word2idx, vocab_size = self.build_dictionary(trn_captions,
                                                                                                test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([trn_captions, test_captions,
                             idx2word, word2idx], f, protocol=2)
                # cfg.LOG.info('Save to: ', filepath)
                print('Save to: ', filepath)
        else:  ## already exists, just load
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                trn_captions, test_captions = x[0], x[1]
                idx2word, word2idx = x[2], x[3]
                del x
                vocab_size = len(idx2word)
                # cfg.LOG.info('Load from: %s' % filepath)
                # cfg.LOG.info("vocab: %d" % vocab_size)
                print('Load from: ', filepath)
                print("vocab:", vocab_size)
        if split == "train":
            filenames = trn_filenames
            captions = trn_captions
        else:
            filenames = test_filenames
            captions = test_captions
        files4gen = filenames
        if os.path.exists(os.path.join(data_dir, "filenames_good.pickle")):
            with open(os.path.join(data_dir, "filenames_good.pickle"), "rb") as f:
                files4gen = pickle.load(f)
        return filenames, captions, idx2word, word2idx, vocab_size, files4gen

    # make dictionary
    def build_dictionary(self, trn_captions, test_captions):
        idx2word = dict()
        word2idx = dict()
        word2idx["[pad]"] = 0
        idx2word[0] = "[pad]"
        cnt = 1
        idx2word[1] = "[CLS]"
        word2idx["[CLS]"] = 1
        trn_captions_new = []
        for cap in trn_captions:
            cap_new = [1]
            for word in cap:
                idx = word2idx.get(word, -1)
                if idx == -1:
                    cnt += 1
                    word2idx[word] = cnt
                    idx2word[cnt] = word
                    idx = cnt
                cap_new.append(idx)
            trn_captions_new.append(cap_new)

        test_captions_new = []
        for cap in test_captions:
            cap_new = [1]
            for word in cap:
                idx = word2idx.get(word, -1)
                if idx == -1:
                    cnt += 1
                    word2idx[word] = cnt
                    idx2word[cnt] = word
                    idx = cnt
                cap_new.append(idx)
            test_captions_new.append(cap_new)

        return trn_captions_new, test_captions_new, idx2word, word2idx, cnt

    def load_filenames(self, data_dir, split):
        path = os.path.join(data_dir, split, "filenames.pickle")
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                filenames = pickle.load(f)
            # cfg.LOG.info("Load filenames from: %s (%d)" % (path, len(filenames)))
            print("Load filenames from: %s (%d)" % (path, len(filenames)))
        else:
            filenames = []
            # cfg.LOG.info("No filenames.pickle")
            print("No filenames.pickle")
        return filenames

    def load_landmarks(self, data_dir, filenames, maxShift):
        landmarks = []
        genIdx = []
        path = os.path.join(data_dir, "landmarks.pickle")
        with open(path, "rb") as f:
            data = pickle.load(f)
        for i, filename in enumerate(filenames):
            landmark = data[filename + '.png']
            landmarks.append(landmark)
            if filename in self.files4gen:
                genIdx.append(i)
            # if landmark[45][0] - landmark[36][0] > 96 and abs(landmark[30][0] - landmark[27][0]) <= 5:
            #     genIdx.append(i)
        return landmarks, genIdx

    def load_bboxs(self, data_dir, filenames):
        facial_bboxs = []
        path = os.path.join(data_dir, "facial_bbox.pickle")
        with open(path, "rb") as f:
            data = pickle.load(f)
        for i, filename in enumerate(filenames):
            bboxs = data[filename + '.png']
            facial_bboxs.append(bboxs)
        return facial_bboxs

    def get_img(self, img_path, imsize):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((304, 304))
        hc = math.ceil(np.random.uniform(1e-2, 304 - 256))
        wc = math.ceil(np.random.uniform(1e-2, 304 - 256))
        img = img.crop((wc, hc, wc + 256, hc + 256))
        flipped = np.random.uniform() > 0.5
        if flipped:
            img = ImageOps.mirror(img)
        img = self.transform(img)
        img_set = []
        for i in range(cfg.TREE.BRANCH_NUM):
            t_img = transforms.Resize(imsize)(img)
            imsize *= 2
            img_set.append(t_img)
        return img_set, hc, wc, flipped

    def get_caption(self, sent_ix):
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        num_words = len(sent_caption)
        x = np.zeros((self.num_words, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.num_words:
            x[:num_words, 0] = sent_caption
        else:
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.num_words]
            ix = np.sort(ix)
            ix[0] = 0
            x[:, 0] = sent_caption[ix]
            x_len = self.num_words
        masks = [self.idx2Mask[_x[0]] for _x in x]
        masks = np.asarray(masks).astype('int64')
        return x, x_len - 1, masks

    def __getitem__(self, index):
        cap_idx = np.random.randint(0, self.embeddings_num)
        if self.is4gen:
            index = self.genIdx[index]
            key = self.filenames[index]
            cap_idx = index * self.embeddings_num + cap_idx
            cap, cap_len, masks = self.get_caption(cap_idx)
            landmarks = self.landmarks[index]
            if self.split != 'test':  # trainning data augment
                hc = math.ceil(np.random.uniform(1e-2, 304 - 256))
                wc = math.ceil(np.random.uniform(1e-2, 304 - 256))
                factor = 304 / 256.0
                landmarks = landmarks * factor
                for i, landmark in enumerate(landmarks):
                    # crop
                    landmarks[i][0] = min(256, max(1, landmarks[i][0] - wc))
                    landmarks[i][1] = min(256, max(1, landmarks[i][1] - hc))
                flipped = np.random.uniform() > 0.5
                if flipped:
                    landmarks = self.flip_landmarks(landmarks)
            return cap, cap_len, masks, landmarks, key
        else:
            if cfg.SEG_FLAG and self.split == "train":
                index = self.genIdx[index]
            key = self.filenames[index]
            img_path = os.path.join(self.data_dir, "imgs", key + ".png")
            img, hc, wc, flipped = self.get_img(img_path, self.imsize)
            cap_idx = index * self.embeddings_num + cap_idx
            cap, cap_len, masks = self.get_caption(cap_idx)
            landmarks = self.landmarks[index]
            bboxs = np.array(self.facial_bboxs[index])
            segMap, landmarks, bboxs = self.transformLocs(landmarks, bboxs, key, hc, wc, flipped)
            if sum(bboxs[4]) == 0:
                bboxs[4] = (max(0, bboxs[0][0] - 80), bboxs[0][1] - 25, max(50, bboxs[1][0] - 35), bboxs[1][3] + 25)
            return img, cap, cap_len, masks, landmarks, key, bboxs, segMap

    # randomly transform landmarks
    def transformLocs(self, landmarks, bboxs, key, hc, wc, flipped):
        segMap = np.load(os.path.join(self.data_dir, "segmaps", key + ".npy")).astype(np.float32)
        if self.split != 'test':
            segMap = cv2.resize(segMap, (304, 304)) / 255.0
            segMap = segMap[hc:hc + 256, wc:wc + 256, :]
            factor = 304 / 256.0
            # scale
            landmarks = landmarks * factor
            bboxs = bboxs * factor
            for i, landmark in enumerate(landmarks):
                # crop  
                landmarks[i][0] = min(256, max(1, landmarks[i][0] - wc))
                landmarks[i][1] = min(256, max(1, landmarks[i][1] - hc))
            if flipped:
                segMap = cv2.flip(segMap, 1)
                landmarks = self.flip_landmarks(landmarks)
            for i, box in enumerate(bboxs):
                # crop
                bboxs[i][0] = min(256, max(0, bboxs[i][0] - hc))
                bboxs[i][1] = min(256, max(0, bboxs[i][1] - wc))
                bboxs[i][2] = min(256, max(0, bboxs[i][2] - hc))
                bboxs[i][3] = min(256, max(0, bboxs[i][3] - wc))
                if flipped:
                    bboxs[i][1] = 256 - bboxs[i][1]
                    bboxs[i][3] = 256 - bboxs[i][3]
                    bboxs[i][1], bboxs[i][3] = bboxs[i][3], bboxs[i][1]
        return segMap.transpose((2, 0, 1)), landmarks, bboxs.astype(np.float32)

    # horizontal flip landmarks 
    def flip_landmarks(self, landmarks):
        landmarks[:, 0] = 256 - landmarks[:, 0]
        landmarks_points_flipped = np.zeros(landmarks.shape, dtype=np.float32)
        for i in range(len(FACIAL_LANDMARKS_68_IDXS_FLIP)):
            landmarks_points_flipped[i, 0] = landmarks[FACIAL_LANDMARKS_68_IDXS_FLIP[i] - 1, 0]
            landmarks_points_flipped[i, 1] = landmarks[FACIAL_LANDMARKS_68_IDXS_FLIP[i] - 1, 1]
        return landmarks_points_flipped

    def __len__(self):
        if (self.is4gen or cfg.SEG_FLAG) and (self.split == 'train' or cfg.LM_FLAG):
            return len(self.genIdx)
        return len(self.filenames)
