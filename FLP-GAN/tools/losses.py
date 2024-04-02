import torch
import torch.nn as nn
import numpy as np
# from torch.nns.loss import L1Loss

from attention import func_attention
from config import cfg


# ##################Loss for matching text-image###################
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def sent_loss(cnn_code, rnn_code, labels, batch_size, eps=1e-8):
    """compute sentence loss
        Args:
            cnn_code: face features.
            rnn_code: caption feature.
            labels: match label
        Returns:
            loss0: text/image cross entropy loss
            loss1: image/text cross entropy loss
    """

    ### Batch cosine_similarity
    # --> 1 x batch_size x nef
    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    # cnn_code_norm / rnn_code_norm: 1 x batch_size x 1
    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    # scores* / norm*: 1 x batch_size x batch_size
    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * cfg.TRAIN.SMOOTH.GAMMA3  # match probability matrix

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)

    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(scores0, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1


def words_loss(img_features, words_emb, labels,
               cap_lens, masks, imMasks, batch_size):
    """ compute word loss
        Args:
            words_emb(query): (batch x nef x seq_len)
            img_features(context, value): (batch x nef x 5)
            masks: stopword masks (batch x seq_len)
            imMasks: roi masks (batch x 5)    
        Returns:
            loss0: text/image cross entropy loss
            loss1: image/text cross entropy loss
            att_maps: roi-word attention map
    """
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 5
        context = img_features

        mask = masks[i, :words_num]  # stopword masks

        ### word-region attention ###
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1, \
                                          mask, imMasks)
        ### weiContext: each words represented by weight sum of five region feature

        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        row_sim = row_sim.view(batch_size, words_num)

        # match probability
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


# def compute_lens(landmarks):
#     l_eye_len = abs(landmarks[:, 11] - landmarks[:, 12]).unsqueeze(1)
#     r_eye_len = abs(landmarks[:, 13] - landmarks[:, 14]).unsqueeze(1)
#     nose_w = abs(landmarks[:, 8] - landmarks[:, 10]).unsqueeze(1)
#     nose_h = abs(landmarks[:, 7] - landmarks[:, 9]).unsqueeze(1)
#     mouth_w = abs(landmarks[:, 15] - landmarks[:, 16]).unsqueeze(1)
#     lens = torch.cat([l_eye_len, r_eye_len, nose_w, nose_h, mouth_w], 1)
#     lens = torch.bmm(lens, lens.transpose(1, 2))

#     return lens

# ##################Loss for G and Ds##############################
def discriminator_loss(netD, real_imgs, fake_imgs, conditions,
                       real_labels, fake_labels, segMaps=None):
    # Forward
    real_features, seg_features = netD(real_imgs, segMaps)
    fake_features, _ = netD(fake_imgs.detach())
    # loss
    #
    cond_real_logits = netD.COND_DNET(real_features, conditions)
    cond_real_errD = nn.BCELoss()(cond_real_logits, real_labels)
    cond_fake_logits = netD.COND_DNET(fake_features, conditions)
    cond_fake_errD = nn.BCELoss()(cond_fake_logits, fake_labels)
    #
    batch_size = real_features.size(0)
    cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
    cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])
    #
    if cfg.SEGD_FLAG:
        seg_real_logits = netD.SEG_DNET(real_features, seg_features)
        seg_real_errD = nn.BCELoss()(seg_real_logits, real_labels)
        seg_fake_logits = netD.SEG_DNET(fake_features, seg_features)
        seg_fake_errD = nn.BCELoss()(seg_fake_logits, fake_labels)

    if netD.UNCOND_DNET is not None:
        real_logits = netD.UNCOND_DNET(real_features)
        fake_logits = netD.UNCOND_DNET(fake_features)
        real_errD = nn.BCELoss()(real_logits, real_labels)
        fake_errD = nn.BCELoss()(fake_logits, fake_labels)
        errD = ((real_errD + cond_real_errD) / 2. +
                (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)
        if cfg.SEGD_FLAG:
            errD += (seg_real_errD + seg_fake_errD) / 4
    else:
        errD = cond_real_errD + (cond_fake_errD + cond_wrong_errD) / 2.
    log = 'Real_Acc: {:.4f} Fake_Acc: {:.4f} '.format(torch.mean(real_logits).item(), torch.mean(fake_logits).item())
    return errD, log


def generator_loss(netsD, image_encoder, fake_imgs, real_labels, words_embs,
                   sent_emb, match_labels, masks, roi_masks, bboxs, cap_lens, segmaps=None):
    numDs = len(netsD)
    batch_size = real_labels.size(0)
    logs = ''
    # Forward
    errG_total = 0
    for i in range(numDs):
        features, seg_feat = netsD[i](fake_imgs[i], segmaps)
        cond_logits = netsD[i].COND_DNET(features, sent_emb)
        cond_errG = nn.BCELoss()(cond_logits, real_labels)
        g_loss = cond_errG
        if netsD[i].UNCOND_DNET is not None:
            logits = netsD[i].UNCOND_DNET(features)
            uncond_errG = nn.BCELoss()(logits, real_labels)
            g_loss += uncond_errG
        if netsD[i].SEG_DNET is not None:
            logits = netsD[i].SEG_DNET(features, seg_feat)
            seg_errG = nn.BCELoss()(logits, real_labels)
            g_loss += seg_errG

        errG_total += g_loss
        logs += 'g_loss%d: %.2f ' % (i, g_loss.item())

        # Ranking loss
        if i == (numDs - 1):
            # words_features: batch_size x nef x 17 x 17
            # sent_code: batch_size x nef
            if cfg.ROI_FLAG:
                roi_features, img_code = image_encoder(fake_imgs[i], bboxs)
                w_loss0, w_loss1, _ = words_loss(roi_features, words_embs, match_labels,
                                                 cap_lens, masks, roi_masks, batch_size)
            else:
                region_features, img_code = image_encoder(fake_imgs[i], None)
                w_loss0, w_loss1, _ = grid_words_loss(region_features, words_embs,
                                                      match_labels, cap_lens, None, None, batch_size)
            w_loss = (w_loss0 + w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            s_loss0, s_loss1 = sent_loss(img_code, sent_emb,
                                         match_labels, batch_size)
            s_loss = (s_loss0 + s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA

            # ablation exp
            errG_total += w_loss + s_loss
            logs += 'w_loss: %.2f s_loss: %.2f ' % (w_loss, s_loss.item())

    return errG_total, logs


def grid_words_loss(img_features, words_emb, labels, cap_lens, masks, imMasks, batch_size):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
        masks: batch x seq_len
        imMasks: batch x 17 x 17
    """
    att_maps = []
    similarities = []
    cap_lens = cap_lens.data.tolist()
    for i in range(batch_size):
        # Get the i-th text description
        words_num = cap_lens[i]
        # -> 1 x nef x words_num
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()
        # -> batch_size x nef x words_num
        word = word.repeat(batch_size, 1, 1)
        # batch x nef x 17*17
        context = img_features

        """
            word(query): batch x nef x words_num
            context: batch x nef x 17 x 17
            weiContext: batch x nef x words_num
            attn: batch x words_num x 17 x 17
        """
        mask = None
        if masks is not None:
            mask = masks[i, :words_num]
        weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1, \
                                          mask, imMasks)
        # weiContext, attn = func_attention(word, context, cfg.TRAIN.SMOOTH.GAMMA1)
        att_maps.append(attn[i].unsqueeze(0).contiguous())
        # --> batch_size x words_num x nef
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        # --> batch_size*words_num x nef
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        # -->batch_size*words_num
        row_sim = cosine_similarity(word, weiContext)
        # --> batch_size x words_num
        # mask = mask.repeat(batch_size, 1)
        row_sim = row_sim.view(batch_size, words_num)
        # row_sim.data.masked_fill_(mask.bool(), -float('inf'))

        # Eq. (10)
        row_sim.mul_(cfg.TRAIN.SMOOTH.GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        # --> 1 x batch_size
        # similarities(i, j): the similarity between the i-th image and the j-th text description
        similarities.append(row_sim)

    # batch_size x batch_size
    similarities = torch.cat(similarities, 1)
    similarities = similarities * cfg.TRAIN.SMOOTH.GAMMA3
    similarities1 = similarities.transpose(0, 1)
    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    else:
        loss0, loss1 = None, None
    return loss0, loss1, att_maps


##################################################################
def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def landmark_loss(LmDetector, imgs, landmarks):
    L1Loss = torch.nn.L1Loss().to(cfg.device)
    batch_size = imgs.shape[0]
    imgs = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)(imgs)
    imgs.add_(1).div_(2)
    std = torch.tensor([0.229, 0.224, 0.225]).to(cfg.device).view(-1, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(cfg.device).view(-1, 1, 1)
    imgs.sub_(mean).div_(std)
    _, locs, scores = LmDetector(imgs)
    locs = locs.view(batch_size, -1, 2)
    locs /= 224
    lm_idxs = torch.tensor(
        [2, 3, 5, 7, 8, 9, 11, 13, 14, 28, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 52, 54, 56,
         58]).to(cfg.device)
    landmarks = landmarks.index_select(1, lm_idxs)
    landmarks /= 256.0
    locs = locs.index_select(1, lm_idxs)
    errL1 = cfg.TRAIN.SMOOTH.GAMMA4 * L1Loss(locs, landmarks)

    return errL1, locs
