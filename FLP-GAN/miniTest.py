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
# import cv2
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
from data.dataset import FaceTextDataset
from models.encoders import RNN_ENCODER, FACE_ENCODER
from models.netG import G_NET
from tools.config import cfg, cfg_from_file
from models.LMGan import LMGen
from tools.losses import words_loss
from tools.visualizations import build_roiattn_images, editShapes, bSplineAndSeg, build_lm_points_images

model = None


def main():
    parser = argparse.ArgumentParser("FLG-GAN")
    parser.add_argument(
        "-gpu_id",
        type=str,
        default="0",
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
        default=False,
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

    testDataset = FaceTextDataset(cfg.DATAPATH, cfg.TREE.BASE_SIZE, cfg.MAXLEN, split="test")
    testDataloader = DataLoader(testDataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=8)

    # idx2word, word2idx = loadDict(os.path.join(cfg.DATAPATH, "captions.pickle"))
    idx2word, word2idx = testDataset.idx2word, testDataset.word2idx
    vocab_size = len(idx2word)

    idx2Mask = getIdx2Masks(word2idx, idx2word)

    # load models
    TextEncoder = RNN_ENCODER(vocab_size, cfg=cfg)
    state_dict = torch.load(cfg.ENCODER_PATH, map_location=lambda storage, loc: storage)
    print('Text', cfg.ENCODER_PATH)
    TextEncoder.load_state_dict(state_dict)
    TextEncoder = TextEncoder.to(cfg.device)
    TextEncoder.eval()
    print('test-TextEncoder load finished')

    ImageEncoder = FACE_ENCODER(cfg.MODEL.E_DIM, cfg)
    img_encoder_path = cfg.ENCODER_PATH.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    ImageEncoder.load_state_dict(state_dict)
    ImageEncoder = ImageEncoder.to(cfg.device)
    ImageEncoder.eval()

    LmG = LMGen()
    state_dict = torch.load(cfg.LMG_PATH, map_location=lambda storage, loc: storage)
    LmG.load_state_dict(state_dict)
    LmG = LmG.to(cfg.device)
    LmG.eval()

    netG = G_NET(cfg)
    print('netG', cfg.NETG_PATH)
    state_dict = torch.load(cfg.NETG_PATH, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG = netG.to(cfg.device)
    netG.eval()
    print("Model load finished")
    if args.sample:
        netG = torch.nn.DataParallel(netG)
        LmG = torch.nn.DataParallel(LmG)
        manualSeed = 200
        np.random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)
        p = multiprocessing.Pool()
        p_conn, c_conn = multiprocessing.Pipe()
        seg_time = 0.0
        save_time = 0.0
        LmG_time = 0.0
        ImG_time = 0.0
        # s_tmp = cfg.NETG_PATH[:cfg.NETG_PATH.rfind('.pth')]
        s_tmp = '/data/wangkun/project/FLG-GAN/output/t1'
        save_dir = '%s/%s' % (s_tmp, 'ablation/onlylmg')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for i in range(10):
            for step, data in enumerate(testDataloader, 0):
                start = time.time()
                imgs, caps, cap_lens, masks, landmarks, keys, bboxs, segMaps = data
                batch_size = masks.shape[0]
                noise = Variable(torch.FloatTensor(batch_size, cfg.MODEL.Z_DIM)).to(cfg.device)
                cap_lens, sorted_cap_indices = \
                    torch.sort(cap_lens, 0, True)
                imgs = [imgs[0][i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
                imgs = Variable(torch.cat(imgs, 0)).to(cfg.device)
                masks = [masks[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
                masks = Variable(torch.cat(masks, 0)).to(cfg.device)
                bboxs = [bboxs[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
                bboxs = Variable(torch.cat(bboxs, 0)).to(cfg.device)
                caps = Variable(caps[sorted_cap_indices].squeeze()).to(cfg.device)
                keys = [keys[i] for i in sorted_cap_indices.numpy()]

                start = time.time()

                hidden = TextEncoder.init_hidden(batch_size)
                words_embs, sent_emb = TextEncoder(caps[:, 1:], cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                start = time.time()

                #######################################################
                # (2) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                if not os.path.isdir("%s/segMaps" % (save_dir)):
                    os.makedirs("%s/segMaps" % (save_dir))
                if not os.path.isdir("%s/single" % (save_dir)):
                    os.makedirs("%s/single" % (save_dir))
                if cfg.SEG_FLAG:
                    LMs, scores = LmG(sent_emb, words_embs, noise, masks[:, 1:])
                    LMs = LMs.reshape(batch_size, -1, 2)
                    LMs = LMs * 128 + 128
                    # deltaL = LMs[:,:,1].min()
                    # deltaR = -LMs[:,:,1].max() + 256
                    # delta = deltaL - deltaR
                    # LMs[:,:,1] += delta
                    LmG_time += time.time() - start
                    start = time.time()
                    segMapList = []
                    params = []
                    for j in range(batch_size):
                        params.append(LMs[j].detach().cpu().numpy())
                    segMapsList = list(p.imap(bSplineAndSeg, params))
                    for j in range(batch_size):
                        segMaps = segMapsList[j].astype(np.float32)
                        # segImg = segMaps.copy()
                        segMaps = segMaps.transpose((2, 0, 1))
                        # segImg[:, :, 2] += segImg[:, :, 3]
                        # segImg[:, :, 1] += segImg[:, :, 3]
                        # segImg = Image.fromarray(np.uint8(segImg[:,:,:3]))
                        # segImg.save("%s/segMaps/%s_%d.png" % (save_dir, keys[j], i))
                        segMaps = torch.from_numpy(segMaps / 255.0).unsqueeze(0)
                        segMaps = Variable(segMaps).to(cfg.device)
                        segMapList.append(segMaps)
                    segMaps = torch.cat(segMapList, 0)
                    seg_time += time.time() - start
                    start = time.time()
                    fake_imgs, _, c_code, mu, logvar = netG(noise, sent_emb, words_embs, masks[:, 1:], segMaps)
                    ImG_time += time.time() - start
                else:
                    fake_imgs, _, c_code, mu, logvar = netG(noise, sent_emb, words_embs, masks[:, 1:])

                start = time.time()
                for j in range(batch_size):
                    s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    k = -1
                    # for k in range(len(fake_imgs)):
                    im = fake_imgs[k][j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    fullpath = '%s_%d.png' % (s_tmp, i)
                    im.save(fullpath)
                save_time += time.time() - start
        print("lmG: %fs" % LmG_time)
        print("seg: %fs" % seg_time)
        print("imG: %fs" % ImG_time)
        print("save: %fs" % save_time)
    elif args.shape:
        global model
        model = KeyedVectors.load_word2vec_format(
            '/data/wangkun/project/FLG-GAN/faces/glove.twitter.27B.200d.bin.gz', binary=True)
        noise = Variable(torch.FloatTensor(1, cfg.MODEL.Z_DIM)).to(cfg.device)
        save_dir = "/data/wangkun/project/FLG-GAN/output/t1"
        while (True):
            cap = input("input caption:")
            name = ""
            cap, masks, cap_len = cap2tensor(cap, idx2Mask, word2idx)
            if cap is None or masks is None or cap_len is None:
                continue
            cap = Variable(cap).unsqueeze(0).to(cfg.device)
            masks = Variable(masks).unsqueeze(0).to(cfg.device)
            noise.data.normal_(0, 1)
            hidden = TextEncoder.init_hidden(1)
            words_emb, sent_emb = TextEncoder(cap, cap_len, hidden)

            LMs, scores = LmG(sent_emb, words_emb, noise, masks)

            LMs = LMs.reshape(1, -1, 2)
            LMs = LMs * 128 + 128
            LMs = LMs[0].detach().cpu().numpy()

            # self-att weights
            # t_score, idxs = torch.sort(scores, 1, True)
            # f = open(os.path.join(save_dir, "%s_lm_scores.txt" % name), "w")
            # f.write(str(scores[0]) + '\n')
            # for j in range(cap_len):
            #     f.write(idx2word[int(cap[0][j])] + ' ')
            # f.write("\n")
            # f.write(str(masks[0][1:].detach().cpu()) + "\n")
            # f.write(str(t_score[0]) + '\n')
            # for idx in idxs[0]:
            #     f.write(idx2word[int(cap[0][int(idx)])] + ' ')
            # f.write("\n")
            # f.close()

            batch_size = 1
            # make bboxs
            bboxs = np.zeros((1, 5, 4), dtype=np.float32)
            bboxs[0][0] = (LMs[11][1] - 10, LMs[10][0] - 10, LMs[14][1] + 15, LMs[13][0] + 15)
            bboxs[0][1] = (LMs[17][1] - 10, LMs[16][0] - 10, LMs[20][1] + 15, LMs[19][0] + 15)
            bboxs[0][2] = (LMs[7][1] - 15, LMs[8][0] - 15, LMs[9][1] + 25, LMs[9][0] + 25)
            bboxs[0][3] = (LMs[23][1] - 25, LMs[22][0] - 25, LMs[26][1] + 25, LMs[25][0] + 25)
            bboxs[0][4] = (
            max(0, bboxs[0][0][0] - 80), bboxs[0][0][1] - 15, max(40, bboxs[0][1][0] - 50), bboxs[0][1][3] + 15)
            bboxs = bboxs * cfg.ATT_SZE / 256
            bboxs = Variable(torch.from_numpy(bboxs)).to(cfg.device)
            bboxs_ori = bboxs
            bboxs = bboxs.reshape(5, 4)
            batch_idxs = torch.arange(batch_size).reshape(batch_size, 1).repeat(1, 5).reshape(batch_size * 5, 1).to(
                cfg.device)
            bboxs = torch.cat((batch_idxs, bboxs), 1)
            roi_masks = Variable(torch.zeros((1, 5), dtype=bool)).to(cfg.device)

            segMaps = bSplineAndSeg(LMs).astype(np.float32)
            segImg = segMaps.copy()
            segMaps = segMaps.transpose((2, 0, 1))
            segImg[:, :, 2] += segImg[:, :, 3]
            segImg[:, :, 1] += segImg[:, :, 3]
            segImg = Image.fromarray(np.uint8(segImg[:, :, :3]))
            segImg.save(os.path.join(save_dir, "%s_fake_segmap.png") % name)
            segMaps = torch.from_numpy(segMaps / 255.0).unsqueeze(0)
            segMaps = Variable(segMaps).to(cfg.device)
            fake_imgs, _, c_code, mu, logvar = netG(noise, sent_emb, words_emb, masks, segMaps)

            imgs = fake_imgs[1].detach()

            if cfg.ROI_FLAG:
                region_features, _ = ImageEncoder(imgs, bboxs)
                _, _, att_maps = words_loss(region_features.detach(), words_emb.detach(),
                                    None, cap_len, masks, roi_masks, 1)
                img_set, _ = \
                        build_roiattn_images(cfg, imgs.cpu(), cap, idx2word, att_maps, cfg.ATT_SZE, bboxs_ori.cpu(), segMaps[:, :3].detach().cpu())
                if img_set is not None:
                    im = Image.fromarray(img_set)
                    im.save(os.path.join(save_dir, "%s_attnVis.png" % name))

            for i in range(3):
                fake_imgs[i][0].add_(1).div_(2).mul_(255)
                img = fake_imgs[i][0].detach().cpu().data.numpy()
                img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(np.uint8(img))
                img.save(os.path.join(save_dir, "%s_fake_img_%d.png" % (name, i)))

            # edit
            feedback = input("input feedback:")
            weight = input("input weight:")
            try:
                weight = float(weight)
            except:
                weight = 5
            tLMs = editShapes(LMs, feedback, model, weight)
            segMaps = bSplineAndSeg(tLMs).astype(np.float32)
            segImg = segMaps.copy()
            segMaps = segMaps.transpose((2, 0, 1))
            segImg[:, :, 2] += segImg[:, :, 3]
            segImg[:, :, 1] += segImg[:, :, 3]
            segImg = Image.fromarray(np.uint8(segImg[:, :, :3]))
            segImg.save(os.path.join(save_dir, "%s_fake_segmap_edit.png" % name))
            segMaps = torch.from_numpy(segMaps / 255.0).unsqueeze(0)
            segMaps = Variable(segMaps).to(cfg.device)
            fake_imgs, _, _, mu, logvar = netG(noise, sent_emb, words_emb, masks, segMaps, c_code)
            fake_imgs[2][0].add_(1).div_(2).mul_(255)
            fake_img = fake_imgs[2][0].detach().cpu().data.numpy()
            fake_img = np.transpose(fake_img, (1, 2, 0))
            fake_img = Image.fromarray(np.uint8(fake_img))
            fake_img.save(os.path.join(save_dir, "%s_fake_img_edit.png" % name))
    else:
        print('Face Image Generation Test:')
        noise = Variable(torch.FloatTensor(1, cfg.MODEL.Z_DIM)).to(cfg.device)
        save_dir = "/data/wangkun/project/FLG-GAN/output/t1"
        while (True):
            cap = input("input caption:")
            # name = "6"
            name = cap
            cap, masks, cap_len = cap2tensor(cap, idx2Mask, word2idx)
            if cap is None or masks is None or cap_len is None:
                continue
            cap = Variable(cap).unsqueeze(0).to(cfg.device)
            masks = Variable(masks).unsqueeze(0).to(cfg.device)
            noise.data.normal_(0, 1)
            hidden = TextEncoder.init_hidden(1)
            words_emb, sent_emb = TextEncoder(cap, cap_len, hidden)

            LMs, scores = LmG(sent_emb, words_emb, noise, masks)
            lmImg = build_lm_points_images(LMs)
            lmImg.save(os.path.join(save_dir, "%s_fake_lm.png") % name)
            LMs = LMs.reshape(1, -1, 2)
            LMs = LMs * 128 + 128
            LMs = LMs[0].detach().cpu().numpy()
            # self-att weights
            t_score, idxs = torch.sort(scores, 1, True)
            f = open(os.path.join(save_dir, "%s_lm_scores.txt" % name), "w")
            f.write(str(scores[0]) + '\n')
            for j in range(cap_len):
                f.write(idx2word[int(cap[0][j])] + ' ')
            f.write("\n")
            f.write(str(masks[0][1:].detach().cpu()) + "\n")
            f.write(str(t_score[0]) + '\n')
            for idx in idxs[0]:
                f.write(idx2word[int(cap[0][int(idx)])] + ' ')
            f.write("\n")
            f.close()
            batch_size = 1
            # make bboxs
            bboxs = np.zeros((1, 5, 4), dtype=np.float32)
            bboxs[0][0] = (LMs[11][1] - 10, LMs[10][0] - 10, LMs[14][1] + 15, LMs[13][0] + 15)
            bboxs[0][1] = (LMs[17][1] - 10, LMs[16][0] - 10, LMs[20][1] + 15, LMs[19][0] + 15)
            bboxs[0][2] = (LMs[7][1] - 15, LMs[8][0] - 15, LMs[9][1] + 25, LMs[9][0] + 25)
            bboxs[0][3] = (LMs[23][1] - 25, LMs[22][0] - 25, LMs[26][1] + 25, LMs[25][0] + 25)
            bboxs[0][4] = (
            max(0, bboxs[0][0][0] - 80), bboxs[0][0][1] - 15, max(40, bboxs[0][1][0] - 50), bboxs[0][1][3] + 15)
            bboxs = bboxs * cfg.ATT_SZE / 256
            bboxs = Variable(torch.from_numpy(bboxs)).to(cfg.device)
            bboxs_ori = bboxs
            bboxs = bboxs.reshape(5, 4)
            batch_idxs = torch.arange(batch_size).reshape(batch_size, 1).repeat(1, 5).reshape(batch_size * 5, 1).to(
                cfg.device)
            bboxs = torch.cat((batch_idxs, bboxs), 1)
            roi_masks = Variable(torch.zeros((1, 5), dtype=bool)).to(cfg.device)

            # lmImg = LMs.copy()
            # print(lmImg.shape)
            # lmImg[:, :, 2] += lmImg[:, :, 3]
            # lmImg[:, :, 1] += lmImg[:, :, 3]
            # lmImg = Image.fromarray(np.uint8(lmImg[:, :, :3]))
            # lmImg.save(os.path.join(save_dir, "%s_fake_lm.png") % name)

            segMaps = bSplineAndSeg(LMs).astype(np.float32)
            segImg = segMaps.copy()
            segMaps = segMaps.transpose((2, 0, 1))
            segImg[:, :, 2] += segImg[:, :, 3]
            segImg[:, :, 1] += segImg[:, :, 3]
            segImg = Image.fromarray(np.uint8(segImg[:, :, :3]))
            segImg.save(os.path.join(save_dir, "%s_fake_segmap.png") % name)
            segMaps = torch.from_numpy(segMaps / 255.0).unsqueeze(0)
            segMaps = Variable(segMaps).to(cfg.device)
            fake_imgs, _, c_code, mu, logvar = netG(noise, sent_emb, words_emb, masks, segMaps)

            imgs = fake_imgs[1].detach()

            if cfg.ROI_FLAG:
                region_features, _ = ImageEncoder(imgs, bboxs)
                _, _, att_maps = words_loss(region_features.detach(), words_emb.detach(),
                                            None, cap_len, masks, roi_masks, 1)
                img_set, _ = \
                    build_roiattn_images(cfg, imgs.cpu(), cap, idx2word, att_maps, cfg.ATT_SZE, bboxs_ori.cpu(),
                                         segMaps[:, :3].detach().cpu())
                if img_set is not None:
                    im = Image.fromarray(img_set)
                    im.save(os.path.join(save_dir, "%s_attnVis.png" % name))

            for i in range(3):
                fake_imgs[i][0].add_(1).div_(2).mul_(255)
                img = fake_imgs[i][0].detach().cpu().data.numpy()
                img = np.transpose(img, (1, 2, 0))
                img = Image.fromarray(np.uint8(img))
                img.save(os.path.join(save_dir, "%s_fake_img_%d.png" % (name, i)))

            # edit
            cap = input("edit caption:")
            cap, masks, cap_len = cap2tensor(cap, idx2Mask, word2idx)
            if cap is None or masks is None or cap_len is None:
                continue
            cap = Variable(cap).unsqueeze(0).to(cfg.device)
            masks = Variable(masks).unsqueeze(0).to(cfg.device)
            hidden = TextEncoder.init_hidden(1)
            words_emb, sent_emb = TextEncoder(cap, cap_len, hidden)
            LMs, scores = LmG(sent_emb, words_emb, noise, masks)
            LMs = LMs.reshape(1, -1, 2)
            LMs = LMs * 128 + 128
            LMs = LMs[0].detach().cpu().numpy()

            batch_size = 1
            # make bboxs
            bboxs = np.zeros((1, 5, 4), dtype=np.float32)
            bboxs[0][0] = (LMs[11][1] - 10, LMs[10][0] - 10, LMs[14][1] + 15, LMs[13][0] + 15)
            bboxs[0][1] = (LMs[17][1] - 10, LMs[16][0] - 10, LMs[20][1] + 15, LMs[19][0] + 15)
            bboxs[0][2] = (LMs[7][1] - 15, LMs[8][0] - 15, LMs[9][1] + 25, LMs[9][0] + 25)
            bboxs[0][3] = (LMs[23][1] - 25, LMs[22][0] - 25, LMs[26][1] + 25, LMs[25][0] + 25)
            bboxs[0][4] = (
                max(0, bboxs[0][0][0] - 80), bboxs[0][0][1] - 15, max(40, bboxs[0][1][0] - 50), bboxs[0][1][3] + 15)
            bboxs = bboxs * cfg.ATT_SZE / 256
            bboxs = Variable(torch.from_numpy(bboxs)).to(cfg.device)
            bboxs_ori = bboxs
            bboxs = bboxs.reshape(5, 4)
            batch_idxs = torch.arange(batch_size).reshape(batch_size, 1).repeat(1, 5).reshape(batch_size * 5, 1).to(
                cfg.device)
            bboxs = torch.cat((batch_idxs, bboxs), 1)
            roi_masks = Variable(torch.zeros((1, 5), dtype=bool)).to(cfg.device)

            segMaps = bSplineAndSeg(LMs).astype(np.float32)
            segImg = segMaps.copy()
            segMaps = segMaps.transpose((2, 0, 1))
            segImg[:, :, 2] += segImg[:, :, 3]
            segImg[:, :, 1] += segImg[:, :, 3]
            segImg = Image.fromarray(np.uint8(segImg[:, :, :3]))
            segImg.save(os.path.join(save_dir, "%s_fake_segmap_edit.png") % name)
            segMaps = torch.from_numpy(segMaps / 255.0).unsqueeze(0)
            segMaps = Variable(segMaps).to(cfg.device)
            fake_imgs, _, _, mu, logvar = netG(noise, sent_emb, words_emb, masks, segMaps)
            fake_imgs[2][0].add_(1).div_(2).mul_(255)
            fake_img = fake_imgs[2][0].detach().cpu().data.numpy()
            fake_img = np.transpose(fake_img, (1, 2, 0))
            fake_img = Image.fromarray(np.uint8(fake_img))
            fake_img.save(os.path.join(save_dir, "%s_fake_img_edit.png") % name)


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
