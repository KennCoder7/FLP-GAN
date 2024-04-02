import numpy as np
from numpy.lib.type_check import real
import torch
import cv2
import os
import torch.nn as nn
import skimage.transform
from PIL import Image, ImageDraw, ImageFont
from scipy import interpolate
import matplotlib.pyplot as plt
import collections
import nltk
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# from config import cfg

# For visualization ################################################
COLOR_DIC = {0: [128, 64, 128], 1: [244, 35, 232],
             2: [70, 70, 70], 3: [102, 102, 156],
             4: [190, 153, 153], 5: [153, 153, 153],
             6: [250, 170, 30], 7: [220, 220, 0],
             8: [107, 142, 35], 9: [152, 251, 152],
             10: [70, 130, 180], 11: [220, 20, 60],
             12: [255, 0, 0], 13: [0, 0, 142],
             14: [119, 11, 32], 15: [0, 60, 100],
             16: [0, 80, 100], 17: [0, 0, 230],
             18: [0, 0, 70], 19: [0, 0, 0]}

FONT_MAX = 50

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]


def build_lm_images(landmarks):
    convas = np.zeros((512, 512, 3), dtype=np.uint8)
    landmarks = landmarks.reshape(4, -1, 2)
    landmarks = landmarks * 128 + 128
    # landmarks = landmarks * 256
    convas = Image.fromarray(convas).convert('RGB')
    draw = ImageDraw.Draw(convas)

    dx = [0, 0, 256, 256]
    dy = [0, 256, 0, 256]
    for i, landmark in enumerate(landmarks):
        img = bSplineAndSeg(landmark)
        img[:, :, 2] += img[:, :, 3]
        img[:, :, 1] += img[:, :, 3]
        img = Image.fromarray(img[:, :, :3])
        convas.paste(img, (dx[i], dy[i], dx[i] + 256, dy[i] + 256))
    # for i, landmark in enumerate(landmarks):
    #     for point in landmark:
    #         draw.ellipse((dx[i] + point[0] - 3, dy[i] + point[1] - 3, dx[i] + point[0] + 3, dy[i] + point[1] + 3), \
    #             fill=(255, 255, 255))
    draw.line((0, 256, 512, 256), fill=(255, 0, 0), width=3)
    draw.line((256, 0, 256, 512), fill=(255, 0, 0), width=3)

    return convas


def build_lm_points_images(landmarks):
    convas = np.zeros((256, 256, 3), dtype=np.uint8)
    landmarks = landmarks.reshape(1, -1, 2)
    landmarks = landmarks * 128 + 128
    # landmarks = landmarks * 256
    convas = Image.fromarray(convas).convert('RGB')
    draw = ImageDraw.Draw(convas)
    left_eye = [10, 11, 12, 13, 14, 15, 10]
    right_eye = [16, 17, 18, 19, 20, 21, 16]
    nose = [7, 8, 9, 7]
    mouth = [22, 23, 24, 25, 26, 27, 22]
    face = list(range(7))
    colors = ['r', 'r', 'g', 'b', 'b']

    dx = [0, 0, 256, 256]
    dy = [0, 256, 0, 256]
    # for i, landmark in enumerate(landmarks):
    #     img = bSplineAndSeg(landmark)
    #     img[:, :, 2] += img[:, :, 3]
    #     img[:, :, 1] += img[:, :, 3]
    #     img = Image.fromarray(img[:, :, :3])
    #     convas.paste(img, (dx[i], dy[i], dx[i] + 256, dy[i] + 256))
    for i, landmark in enumerate(landmarks):
        for j, point in enumerate(landmark):
            if j in left_eye:
                color = (255,0,0)
            elif j in right_eye:
                color = (255,0,0)
            elif j in nose:
                color = (0,255,0)
            elif j in mouth:
                color = (0,0,255)
            else:
                color = (66,255,255)
            draw.ellipse((dx[i] + point[0] - 3, dy[i] + point[1] - 3, dx[i] + point[0] + 3, dy[i] + point[1] + 3),
                         fill=color)
    # draw.line((0, 256, 512, 256), fill=(255, 0, 0), width=3)
    # draw.line((256, 0, 256, 512), fill=(255, 0, 0), width=3)

    return convas


def build_super_images(cfg, real_imgs, captions, ixtoword,
                       attn_maps, att_sze, pre_imgs=None, locs=None):
    max_word_num = cfg.MAXLEN - 1
    batch_size = real_imgs.size(0)
    nvis = min(8, batch_size)
    real_imgs = real_imgs[:nvis]
    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = real_imgs.size(2)
    if real_imgs.size(2) < 256:
        locs = None
    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i % 20]

    real_imgs = \
        nn.Upsample(size=(vis_size, vis_size), mode='bilinear', align_corners=True)(real_imgs)
    # [-1, 1] --> [0, 1]
    # real_imgs[:,0].mul_(std[0]).add_(mean[0])
    # real_imgs[:,1].mul_(std[1]).add_(mean[1])
    # real_imgs[:,2].mul_(std[2]).add_(mean[2])
    # real_imgs.mul_(255)
    real_imgs.add_(1).div_(2).mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1)).copy()
    if locs is not None:
        for i in range(nvis):
            for landmark in locs[i]:
                cv2.circle(real_imgs[i], (int(landmark[0] * vis_size), int(landmark[1] * vis_size)), 1, (0, 255, 0), 3)
    if pre_imgs is not None:
        pre_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear', align_corners=True)(pre_imgs)
        # [-1, 1] --> [0, 1]
        # pre_imgs[:,0].mul_(std[0]).add_(mean[0])
        # pre_imgs[:,1].mul_(std[1]).add_(mean[1])
        # pre_imgs[:,2].mul_(std[2]).add_(mean[2])
        # pre_imgs.mul_(255)
        pre_imgs.add_(1).div_(2).mul_(255)
        pre_imgs = pre_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        pre_imgs = np.transpose(pre_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        # 1 x word_num x 17 x 17
        attn = attn_maps[i].cpu().view(1, -1, att_sze, att_sze)
        # --> 1 x 1 x 17 x 17
        attn_max = attn.max(dim=1, keepdim=True)
        attn = torch.cat([attn_max[0], attn], 1)
        #
        attn = attn.view(-1, 1, att_sze, att_sze)
        attn = attn.repeat(1, 3, 1, 1).data.numpy()
        # n x c x h x w --> n x h x w x c
        attn = np.transpose(attn, (0, 2, 3, 1))
        np.save(os.path.join(cfg.IMG_DIR, "max%d.npy" % i), attn_max[1])
        np.save(os.path.join(cfg.IMG_DIR, "image%d.npy" % i), attn)
        num_attn = attn.shape[0]
        #
        img = real_imgs[i]
        lrI = img
        if pre_imgs is not None:
            lrI = pre_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            one_map = attn[j]
            if (vis_size // att_sze) > 1:
                one_map = \
                    skimage.transform.pyramid_expand(one_map, sigma=20,
                                                     upscale=vis_size // att_sze, channel_axis=-1)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                if j == 0:
                    one_map = (one_map - minVglobal) / (maxVglobal - minVglobal)
                else:
                    one_map = (one_map - one_map.min()) / (one_map.max() - one_map.min() + 1e-9)
                one_map *= 255
                #
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)
                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        return img_set, sentences
    else:
        return None


def drawCaption(convas, captions, ixtoword, vis_size, off1=2, off2=2):
    num = captions.size(0)
    img_txt = Image.fromarray(convas)
    fnt = ImageFont.truetype('/home/lchen/code/AttnGAN/eval/FreeMono.ttf', int(50 * vis_size / 272))
    # get a drawing context
    d = ImageDraw.Draw(img_txt)
    sentence_list = []
    for i in range(num):
        cap = captions[i].data.cpu().numpy()
        sentence = []
        for j in range(len(cap)):
            if cap[j] == 0:
                break
            word = ixtoword[cap[j]].encode('ascii', 'ignore').decode('ascii')
            d.text(((j + off1) * (vis_size + off2), i * FONT_MAX), '%d:%s' % (j, word[:6]),
                   font=fnt, fill=(255, 255, 255, 255))
            sentence.append(word)
        sentence_list.append(sentence)
    return img_txt, sentence_list


def build_roiattn_images(cfg, real_imgs, captions, ixtoword, \
                         attn_maps, att_sze, bboxs, pre_imgs=None):
    max_word_num = cfg.MAXLEN
    batch_size = real_imgs.size(0)
    nvis = min(8, batch_size)
    real_imgs = real_imgs[:nvis]

    if att_sze == 17:
        vis_size = att_sze * 16
    else:
        vis_size = att_sze * 8
    delta = vis_size // att_sze
    # vis_size = real_imgs.size(2)

    text_convas = \
        np.ones([batch_size * FONT_MAX,
                 (max_word_num + 2) * (vis_size + 2), 3],
                dtype=np.uint8)

    for i in range(max_word_num):
        istart = (i + 2) * (vis_size + 2)
        iend = (i + 3) * (vis_size + 2)
        text_convas[:, istart:iend, :] = COLOR_DIC[i % 20]

    real_imgs = \
        nn.functional.interpolate(real_imgs, size=(vis_size, vis_size), mode='bilinear', align_corners=True)
    # [-1, 1] --> [0, 1]
    real_imgs.add_(1).div_(2).mul_(255)
    # real_imgs[:,0].mul_(std[0]).add_(mean[0])
    # real_imgs[:,1].mul_(std[1]).add_(mean[1])
    # real_imgs[:,2].mul_(std[2]).add_(mean[2])
    # real_imgs.mul_(255)
    real_imgs = real_imgs.data.numpy()
    # b x c x h x w --> b x h x w x c
    real_imgs = np.transpose(real_imgs, (0, 2, 3, 1))
    if pre_imgs is not None:
        pre_imgs = \
            nn.Upsample(size=(vis_size, vis_size), mode='bilinear', align_corners=True)(pre_imgs)
        # [-1, 1] --> [0, 1]
        # pre_imgs[:,0].mul_(std[0]).add_(mean[0])
        # pre_imgs[:,1].mul_(std[1]).add_(mean[1])
        # pre_imgs[:,2].mul_(std[2]).add_(mean[2])
        # pre_imgs.mul_(255)
        pre_imgs.add_(1).div_(2).mul_(255)
        pre_imgs = pre_imgs.data.numpy()
        # b x c x h x w --> b x h x w x c
        pre_imgs = np.transpose(pre_imgs, (0, 2, 3, 1))
    pad_sze = real_imgs.shape
    middle_pad = np.zeros([pad_sze[2], 2, 3])
    post_pad = np.zeros([pad_sze[1], pad_sze[2], 3])

    # batch x seq_len x 17 x 17 --> batch x 1 x 17 x 17
    seq_len = max_word_num
    img_set = []
    num = nvis  # len(attn_maps)

    text_map, sentences = \
        drawCaption(text_convas, captions, ixtoword, vis_size)
    text_map = np.asarray(text_map).astype(np.uint8)

    bUpdate = 1
    for i in range(num):
        attn = attn_maps[i].cpu()  # 1 x word_num x 5
        attn_max = attn.max(dim=1, keepdim=True)  # 1 x 1 x 5
        attn = torch.cat([attn_max[0], attn], 1)  # 1 x word_num + 1 x 5
        attn = attn.squeeze().data.numpy()  # word_num + 1 x 5
        # np.save(os.path.join(cfg.IMG_DIR, "max%d.npy" % i), attn_max[1])
        # np.save(os.path.join(cfg.IMG_DIR, "image%d.npy" % i), attn)
        num_attn = attn.shape[0]  # word num
        #
        img = real_imgs[i]
        lrI = img
        if pre_imgs is not None:
            lrI = pre_imgs[i]
        row = [lrI, middle_pad]
        row_merge = [img, middle_pad]
        row_beforeNorm = []
        minVglobal, maxVglobal = 1, 0
        for j in range(num_attn):
            roi_map = attn[j]
            one_map = np.zeros((att_sze, att_sze, 3))
            boxs = bboxs[i] * vis_size / att_sze
            boxs = boxs.data.numpy().astype('int')
            boxs = boxs // delta
            boxs = np.clip(boxs, 0, att_sze - 1)
            for idx in range(0, 5):
                for ti in range(boxs[idx][0], boxs[idx][2]):
                    for tj in range(boxs[idx][1], boxs[idx][3]):
                        one_map[ti][tj][:] = max(one_map[ti][tj][0], roi_map[idx])
            one_map = \
                skimage.transform.pyramid_expand(one_map, sigma=20,
                                                 upscale=vis_size // att_sze, channel_axis=-1)
            row_beforeNorm.append(one_map)
            minV = one_map.min()
            maxV = one_map.max()
            if minVglobal > minV:
                minVglobal = minV
            if maxVglobal < maxV:
                maxVglobal = maxV
        for j in range(seq_len + 1):
            if j < num_attn:
                one_map = row_beforeNorm[j]
                if j == 0:
                    one_map = (one_map - minVglobal) / (maxVglobal - minVglobal + 1e-9)
                else:
                    # one_map = (one_map - one_map.min()) / (one_map.max() - one_map.min() + 1e-9)
                    one_map = (one_map - one_map.min()) / (maxVglobal - minVglobal + 1e-9)
                one_map *= 255
                one_map = one_map.astype('uint8')
                # 
                PIL_im = Image.fromarray(np.uint8(img))
                PIL_att = Image.fromarray(np.uint8(one_map))
                merged = \
                    Image.new('RGBA', (vis_size, vis_size), (0, 0, 0, 0))
                mask = Image.new('L', (vis_size, vis_size), (210))
                merged.paste(PIL_im, (0, 0))
                merged.paste(PIL_att, (0, 0), mask)

                merged = np.array(merged)[:, :, :3]
            else:
                one_map = post_pad
                merged = post_pad
            row.append(one_map)
            row.append(middle_pad)
            #
            row_merge.append(merged)
            row_merge.append(middle_pad)
        row = np.concatenate(row, 1)
        row_merge = np.concatenate(row_merge, 1)
        txt = text_map[i * FONT_MAX: (i + 1) * FONT_MAX]
        if txt.shape[1] != row.shape[1]:
            print('txt', txt.shape, 'row', row.shape)
            bUpdate = 0
            break
        row = np.concatenate([txt, row, row_merge], 0)
        img_set.append(row)
    if bUpdate:
        img_set = np.concatenate(img_set, 0)
        img_set = img_set.astype(np.uint8)
        im = Image.fromarray(img_set)
        return img_set, sentences
    else:
        return None


def bSplineAndSeg(landmarks):
    """ turn landmarks into segment map
        Args:
            landmasks: 28 point coordinate
        Returns:
            data: 4-channel face segment map
    """
    plt.switch_backend('agg')
    plt.style.use('dark_background')
    fig = plt.figure()

    # region index
    left_eye = [10, 11, 12, 13, 14, 15, 10]
    right_eye = [16, 17, 18, 19, 20, 21, 16]
    nose = [7, 8, 9, 7]
    mouth = [22, 23, 24, 25, 26, 27, 22]
    face = list(range(7))
    regions = [left_eye, right_eye, nose, mouth, face]

    dims = [0, 0, 1, 2, 3]
    c_dims = [0, 0, 1, 2, 2]
    colors = ['r', 'r', 'g', 'b', 'b']
    image = np.zeros((256, 256, 4), np.uint8)
    history = np.zeros((256, 256, 3))

    # plot background
    plt.style.use('dark_background')
    fig = plt.figure()
    # size setting
    fig.set_size_inches(256 / 100.0, 256 / 100.0)
    plt.axis((0, 256, 0, 256))
    # remove axis
    plt.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    # plot curve for each region
    for idx, region in enumerate(regions):
        x = []
        y = []
        for i in region:
            x.append(landmarks[i][0])
            y.append(256 - landmarks[i][1])
            # b-spline
        tck, u = interpolate.splprep([x, y], k=3, s=0)
        u = np.linspace(0, 1, num=50, endpoint=True)
        out = interpolate.splev(u, tck)
        plt.plot(out[0], out[1], colors[idx])
        fig.canvas.draw()
        # draw image
        buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((256, 256, 3))
        buf = buf - history
        buf = buf.astype(np.uint8)
        history = buf
        image[:, :, dims[idx]] += buf[:, :, c_dims[idx]]
    plt.cla()
    plt.close("all")
    # fill area inside curve
    data = fill(image)
    return data


# fill holes BFS
def fill(m_data):
    """ fill holes inside curve
        Args:
            m_data: image contains five curves
        Returns:
            data: 4-channel face segment map
    """
    n, m = 256, 256
    mask = m_data > 0
    m_data[mask] = 255
    data = m_data.tolist()
    for d in range(3):
        que = collections.deque()
        for i in range(256):
            if data[i][0][d] == 0:
                data[i][0][d] = 2
                que.append((i, 0))
            if data[i][m - 1][d] == 0:
                data[i][m - 1][d] = 2
                que.append((i, m - 1))
        for i in range(m - 1):
            if data[0][i][d] == 0:
                data[0][i][d] = 2
                que.append((0, i))
            if data[n - 1][i][d] == 0:
                data[n - 1][i][d] = 2
                que.append((n - 1, i))
        while que:
            x, y = que.popleft()
            for mx, my in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= mx < n and 0 <= my < m and data[mx][my][d] == 0:
                    data[mx][my][d] = 2
                    que.append((mx, my))
    data = np.asarray(data, dtype=np.uint8)
    m1 = data[:, :, :3] == 2
    m2 = data[:, :, :3] == 0
    data[:, :, :3][m1] = 0
    data[:, :, :3][m2] = 255
    return data


def editShapes(landmarks, feedback, model, weight):
    # regionIdxs = {
    #     "face": [[0, 1, 2, 3, 4, 5, 6]],
    #     "nose": [[7, 8, 9]],
    #     "eye": [[10, 11, 12, 15, 14, 13], [16, 17, 18, 21, 20, 19]],
    #     "mouth": [[22, 23, 24, 27, 26, 25]]
    # }
    # drawLandmarks(landmarks, "lm_ori")
    delta = weight
    deltaK = weight
    changeRules = {
        "face": {
            "longer": [(2, 'down', delta), (3, 'down', delta), (4, 'down', delta)],
            "shorter": [(2, 'up', delta), (3, 'up', delta), (4, 'up', delta)],
            "wider": [(0, 6, '+', deltaK), (1, 5, '+', deltaK), (2, 4, '+', deltaK)],
            "narrower": [(0, 6, '-', deltaK), (1, 5, '-', deltaK), (2, 4, '-', deltaK)],
            "bigger": [(0, 6, '+', deltaK), (1, 5, '+', deltaK), (2, 4, '+', deltaK), (2, 'down', deltaK),
                       (3, 'down', delta), (4, 'down', delta)],
            "smaller": [(0, 6, '-', deltaK), (1, 5, '-', deltaK), (2, 4, '-', deltaK), (2, 'up', delta),
                        (3, 'up', delta), (4, 'up', delta)]
        },
        "nose": {
            "longer": [(7, 'up', delta)],
            "shorter": [(7, 'down', delta)],
            "wider": [(8, 9, '+', deltaK)],
            "narrower": [(8, 9, '-', deltaK)],
            "bigger": [(7, 'up', delta), (8, 9, '+', deltaK)],
            "smaller": [(7, 'down', delta), (8, 9, '-', deltaK)]
        },
        "eye": {
            "longer": [(10, 13, '+', deltaK), (16, 19, '+', deltaK)],
            "shorter": [(10, 13, '-', deltaK), (16, 19, '-', deltaK)],
            "wider": [(10, 13, '+', deltaK), (16, 19, '+', deltaK)],
            "narrower": [(10, 13, '-', deltaK), (16, 19, '-', deltaK), ],
            "bigger": [(10, 13, '+', deltaK), (11, 15, '+', deltaK), (12, 14, '+', deltaK),
                       (16, 19, '+', deltaK), (17, 21, '+', deltaK), (18, 20, '+', deltaK), ],
            "smaller": [(10, 13, '-', deltaK), (11, 15, '-', deltaK), (12, 14, '-', deltaK),
                        (16, 19, '-', deltaK), (17, 21, '-', deltaK), (18, 20, '-', deltaK), ],
        },
        "mouth": {
            "thinner": [(23, 26, '-', delta), (24, 27, '-', delta)],
            "thicker": [(23, 26, '+', delta), (24, 27, '+', delta)],
            "wider": [(22, 25, '+', delta)],
            "narrower": [(22, 25, '-', delta)],
            "bigger": [(22, 25, '+', delta), (23, 26, '+', delta), (24, 27, '+', delta)],
            "smaller": [(22, 25, '-', delta), (23, 26, '-', delta), (24, 27, '-', delta)],
        }
    }
    nj = nltkPos(feedback)
    changes = getChanges(nj, model)
    for i, change in enumerate(changes):
        region = change[0]
        changeType = change[1]
        rules = changeRules[region][changeType]
        for rule in rules:
            if len(rule) > 3:
                p1 = rule[0]
                p2 = rule[1]
                if landmarks[p1][0] > landmarks[p2][0]:
                    p1, p2 = p2, p1
                deltaY = landmarks[p2][1] - landmarks[p1][1]
                deltaX = landmarks[p2][0] - landmarks[p1][0]
                k = deltaY / deltaX
                arc = math.atan(k)
                sin = math.sin(arc)
                cos = math.cos(arc)
                delta = rule[3]
                if rule[2] == '+':
                    landmarks[p1][0] -= delta * cos
                    landmarks[p2][0] += delta * cos
                    landmarks[p1][1] -= delta * sin
                    landmarks[p2][1] += delta * sin
                elif rule[2] == '-':
                    landmarks[p1][0] += delta * cos
                    landmarks[p2][0] -= delta * cos
                    landmarks[p1][1] += delta * sin
                    landmarks[p2][1] -= delta * sin
            else:
                if rule[1] == 'up':
                    landmarks[rule[0]][1] -= rule[2]
                elif rule[1] == 'down':
                    landmarks[rule[0]][1] += rule[2]
    # drawLandmarks(landmarks, "lm_edit")
    return landmarks


def nltkPos(feedback):
    separators = set([',', '.'])
    # document = input("input feedback:")
    # document = 'bigger eyes, shorter face, smaller nose, thinner and wider lips.'
    # document = "bigger eyes"
    sentences = nltk.sent_tokenize(feedback)
    nn = []
    jj = []
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        print(words)
        pos_tags = nltk.pos_tag(words)
        print(pos_tags)
        for item in pos_tags:
            if item[1].startswith('JJ') or item[0].endswith('er'):
                jj.append(item[0])
            elif item[1].startswith('NN'):
                nn.append(item[0])
            elif item[1] in separators:
                if len(nn) > len(jj):
                    jj.append(jj[-1])
                elif len(nn) < len(jj):
                    nn.append(nn[-1])
    nj = list(zip(nn, jj))
    print(nj)
    return nj


def getChanges(nj, model):
    targets = ["face", "eye", "nose", "mouth"]
    changeTypes = ["bigger", "smaller", "wider", "thinner", "thicker", "longer", "narrower", "shorter"]
    changes = []
    for i, item in enumerate(nj):
        # if item[0] not in model.vocab or item[1] not in model.vocab:
        #     continue
        targetItem = ""
        changeItem = ""
        maxi = 0.0
        for target in targets:
            sim = model.similarity(item[0], target)
            if sim > maxi:
                maxi = sim
                targetItem = target
        maxi = 0.0
        for changeType in changeTypes:
            sim = model.similarity(item[1], changeType)
            if sim > maxi:
                maxi = sim
                changeItem = changeType
        changes.append((targetItem, changeItem))
    print(changes)
    return changes


def drawLandmarks(landmarks, name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    t_img = np.zeros((255, 255, 3), dtype=np.uint8)
    for i, point in enumerate(landmarks):
        cv2.putText(t_img, str(i), (int(point[0]), int(point[1])), font, 0.3, (255, 0, 0), 1)
    cv2.imwrite(name + ".png", t_img)
