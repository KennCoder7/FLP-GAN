import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from PIL import Image
import time

from config import cfg
from losses import sent_loss, words_loss, generator_loss, discriminator_loss, KL_loss, grid_words_loss, \
    landmark_loss
from visualizations import build_lm_images, build_roiattn_images, build_super_images
from weight import copy_G_params, load_params
from cpm_vgg16 import cpm_vgg16


def trainEncoders(TextEncoder, ImageEncoder, optimizer, scheduler, trnDataloader, valDataloader, idx2word):
    TextEncoder = TextEncoder.to(cfg.device)
    ImageEncoder = ImageEncoder.to(cfg.device)
    best_sl = (float('inf'), float('inf'))
    best_wl = (float('inf'), float('inf'))
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        TextEncoder.train()
        ImageEncoder.train()
        s_total_loss0 = 0
        s_total_loss1 = 0
        w_total_loss0 = 0
        w_total_loss1 = 0
        start_time = time.time()
        for iteration, data in enumerate(trnDataloader):
            imgs, caps, cap_lens, masks, landmarks, keys, bboxs, segMaps = data
            batch_size = masks.shape[0]
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

            labels = Variable(torch.LongTensor(range(batch_size))).to(cfg.device)
            # mask rois for faces without hair appearance
            roi_masks = bboxs.sum(2) == 0

            bboxs = bboxs * 17 / 256
            bboxs_ori = bboxs
            bboxs = bboxs.reshape(batch_size * 5, 4)

            batch_idxs = torch.arange(batch_size).reshape(batch_size, 1).repeat(1, 5).reshape(batch_size * 5, 1).to(
                cfg.device)
            bboxs = torch.cat((batch_idxs, bboxs), 1)

            # roi_features: batch_size x nef x 5
            # img_code: batch_size x nef
            if cfg.ROI_FLAG:
                roi_features, img_code = ImageEncoder(imgs, bboxs)
            else:
                roi_features, img_code = ImageEncoder(imgs, None)

            # words_emb: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            hidden = TextEncoder.init_hidden(batch_size)
            words_emb, sent_emb = TextEncoder(caps[:, 1:], cap_lens, hidden)

            ######### calulate loss #############
            if cfg.ROI_FLAG:
                w_loss0, w_loss1, attn_maps = words_loss(roi_features, words_emb, labels,
                                                         cap_lens, masks[:, 1:], roi_masks, batch_size)
            else:
                w_loss0, w_loss1, attn_maps = grid_words_loss(roi_features, words_emb, labels,
                                                              cap_lens, None, None, batch_size)

            w_total_loss0 += w_loss0.data
            w_total_loss1 += w_loss1.data
            loss = w_loss0 + w_loss1

            s_loss0, s_loss1 = \
                sent_loss(img_code, sent_emb, labels, batch_size)
            loss += s_loss0 + s_loss1
            s_total_loss0 += s_loss0.data
            s_total_loss1 += s_loss1.data

            ######### update model #############
            loss.backward()
            torch.nn.utils.clip_grad_norm_(TextEncoder.parameters(), 0.5)
            optimizer.step()

            if iteration > 0 and iteration % 20 == 0:
                s_cur_loss0 = s_total_loss0.item() / 20
                s_cur_loss1 = s_total_loss1.item() / 20

                w_cur_loss0 = w_total_loss0.item() / 20
                w_cur_loss1 = w_total_loss1.item() / 20

                # cfg.LOG.info('| epoch {:3d} | {:5d}/{:5d} batches | '
                #              's_loss {:5.2f} {:5.2f} | '
                #              'w_loss {:5.2f} {:5.2f}'
                #              .format(epoch, iteration, len(trnDataloader),
                #                      s_cur_loss0, s_cur_loss1,
                #                      w_cur_loss0, w_cur_loss1))
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                             's_loss {:5.2f} {:5.2f} | '
                             'w_loss {:5.2f} {:5.2f}'
                             .format(epoch, iteration, len(trnDataloader),
                                     s_cur_loss0, s_cur_loss1,
                                     w_cur_loss0, w_cur_loss1))
                s_total_loss0 = 0
                s_total_loss1 = 0
                w_total_loss0 = 0
                w_total_loss1 = 0

        ####### save current imgs
        if epoch != 0 and epoch % cfg.TRAIN.SAVE_PERIOD == 0:
            if cfg.ROI_FLAG:
                img_set, _ = \
                    build_roiattn_images(cfg, imgs.cpu(), caps[:, 1:], idx2word, attn_maps, cfg.ATT_SZE, bboxs_ori.cpu())
            else:
                img_set, _ = \
                    build_super_images(cfg, imgs.cpu(), caps[:, 1:], idx2word, attn_maps, att_sze)

            if img_set is not None:
                im = Image.fromarray(img_set)
                fullpath = '%s/attention_maps%d.png' % (cfg.IMG_DIR, epoch)
                im.save(fullpath)
        if optimizer.param_groups[0]["lr"] > 0.00001:
            scheduler.step()
        #################  eval  ###############
        TextEncoder.eval()
        ImageEncoder.eval()
        s_total_loss = 0
        w_total_loss = 0
        for iteration, data in enumerate(valDataloader):
            imgs, caps, cap_lens, masks, landmarks, keys, bboxs, segMaps = data
            batch_size = masks.shape[0]
            labels = Variable(torch.LongTensor(range(batch_size))).to(cfg.device)

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

            # mask rois for faces without hair appearance
            roi_masks = bboxs.sum(2) == 0

            bboxs_ori = bboxs
            bboxs = bboxs.reshape(batch_size * 5, 4)
            bboxs = bboxs * 17 / 256
            batch_idxs = torch.arange(batch_size).reshape(batch_size, 1).repeat(1, 5).reshape(batch_size * 5, 1).to(
                cfg.device)
            bboxs = torch.cat((batch_idxs, bboxs), 1)

            # roi_features: batch_size x nef x 5
            # img_code: batch_size x nef
            if cfg.ROI_FLAG:
                roi_features, img_code = ImageEncoder(imgs, bboxs)
            else:
                roi_features, img_code = ImageEncoder(imgs, None)

            nef, att_sze = roi_features.size(1), roi_features.size(2)

            # words_emb: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            hidden = TextEncoder.init_hidden(batch_size)
            words_emb, sent_emb = TextEncoder(caps[:, 1:], cap_lens, hidden)

            if cfg.ROI_FLAG:
                w_loss0, w_loss1, attn_maps = words_loss(roi_features, words_emb, labels,
                                                         cap_lens, masks[:, 1:], roi_masks, batch_size)
            else:
                w_loss0, w_loss1, attn_maps = grid_words_loss(roi_features, words_emb, labels,
                                                              cap_lens, None, None, batch_size)
            w_total_loss += (w_loss0 + w_loss1).data

            s_loss0, s_loss1 = \
                sent_loss(img_code, sent_emb, labels, batch_size)
            s_total_loss += (s_loss0 + s_loss1).data

        elapsed = time.time() - start_time
        s_cur_loss = s_total_loss.item() / iteration
        w_cur_loss = w_total_loss.item() / iteration
        # cfg.LOG.info('| end epoch {:3d} | valid loss s/w {:5.2f} {:5.2f}' \
        #              '| {:.2f} s/epoch | lr {:.5f}| ' \
        #              .format(epoch, s_cur_loss, w_cur_loss, elapsed, optimizer.param_groups[0]["lr"]))
        print('| end epoch {:3d} | valid loss s/w {:5.2f} {:5.2f}' \
                     '| {:.2f} s/epoch | lr {:.5f}| ' \
                     .format(epoch, s_cur_loss, w_cur_loss, elapsed, optimizer.param_groups[0]["lr"]))

        ########## save models ##########
        if epoch != 0 and epoch % cfg.TRAIN.SAVE_PERIOD == 0:
            torch.save(ImageEncoder.state_dict(),
                       '%s/image_encoder%d_%.2f_%.2f.pth' % (cfg.MODEL_DIR, epoch, s_cur_loss, w_cur_loss))
            torch.save(TextEncoder.state_dict(),
                       '%s/text_encoder%d_%.2f_%.2f.pth' % (cfg.MODEL_DIR, epoch, s_cur_loss, w_cur_loss))
        if s_cur_loss < best_sl[0]:
            torch.save(ImageEncoder.state_dict(),
                       '%s/image_encoder_bestSL.pth' % cfg.MODEL_DIR)
            torch.save(TextEncoder.state_dict(),
                       '%s/text_encoder_bestSL.pth' % cfg.MODEL_DIR)
            best_sl = (s_cur_loss, w_cur_loss)
        if w_cur_loss < best_wl[1]:
            torch.save(ImageEncoder.state_dict(),
                       '%s/image_encoder_bestWL.pth' % cfg.MODEL_DIR)
            torch.save(TextEncoder.state_dict(),
                       '%s/text_encoder_bestWL.pth' % cfg.MODEL_DIR)
            best_wl = (s_cur_loss, w_cur_loss)
        # cfg.LOG.info('| end epoch {:3d} | best valid loss s/w {:5.2f} {:5.2f} | {:5.2f} {:5.2f}'
        #              .format(epoch, best_sl[0], best_sl[1], best_wl[0], best_wl[1]))
        print('| end epoch {:3d} | best valid loss s/w {:5.2f} {:5.2f} | {:5.2f} {:5.2f}'
                     .format(epoch, best_sl[0], best_sl[1], best_wl[0], best_wl[1]))


def trainLMGAN(TextEncoder, LMGen, LMDis, optimizerG, optimizerD, schedulerG, schedulerD, \
               trnDataloader, valDataloader, idx2word):
    criterion = torch.nn.BCELoss().to(cfg.device)
    TextEncoder = TextEncoder.to(cfg.device)
    TextEncoder.eval()
    LMGen = LMGen.to(cfg.device)
    LMDis = LMDis.to(cfg.device)
    L1Loss = torch.nn.L1Loss().to(cfg.device)
    bestLoss = float('inf')
    lm_idxs = torch.tensor(
        [3, 5, 7, 8, 9, 11, 13, 28, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 52, 54, 56, 58]).to(
        cfg.device)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        LMGen.train()
        LMDis.train()
        errD_total = 0
        errG_total = 0
        errL1_total = 0
        start_time = time.time()
        for iteration, data in enumerate(trnDataloader):
            caps, cap_lens, masks, landmarks, keys = data
            bs = caps.shape[0]

            noise = Variable(torch.FloatTensor(bs, cfg.MODEL.Z_DIM)).to(cfg.device)

            cap_lens, sorted_cap_indices = \
                torch.sort(cap_lens, 0, True)

            masks = [masks[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
            masks = Variable(torch.cat(masks, 0)).to(cfg.device)
            caps = Variable(caps[sorted_cap_indices].squeeze()).to(cfg.device)
            keys = [keys[i] for i in sorted_cap_indices.numpy()]
            landmarks = [landmarks[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
            landmarks = Variable(torch.cat(landmarks, 0)).to(cfg.device)
            # --> bs x 28 x 2
            landmarks = landmarks.index_select(1, lm_idxs)
            landmarks = (landmarks - 128) / 128
            # landmarks = landmarks / 256

            fake_label = Variable(torch.FloatTensor(bs).fill_(0)).to(cfg.device)
            real_label = Variable(torch.FloatTensor(bs).fill_(1)).to(cfg.device)
            fake_label = Variable(torch.FloatTensor(bs).data.uniform_(0, 0.1)).to(cfg.device)
            real_label = Variable(torch.FloatTensor(bs).data.uniform_(0.9, 1)).to(cfg.device)

            # forward G
            hidden = TextEncoder.init_hidden(bs)
            words_emb, sent_emb = TextEncoder(caps[:, 1:], cap_lens, hidden)

            noise.data.normal_(0, 1)
            LMs, scores = LMGen(sent_emb, words_emb, noise, masks[:, 1:])

            LMDis.zero_grad()

            ############### wgan implement ###########
            # outputF = LMDis(LMs.detach(), sent_emb)
            # validF = torch.mean(outputF)
            # outputR = LMDis(landmarks.view(bs, -1), sent_emb)
            # validR = -torch.mean(outputR)
            # errD = validF + validR

            # validF_total += validF.data
            # validR_total += validR.data

            # for vannila gan
            # cond_wrong_logits = netD.COND_DNET(real_features[:(batch_size - 1)], conditions[1:batch_size])
            # cond_wrong_errD = nn.BCELoss()(cond_wrong_logits, fake_labels[1:batch_size])

            # if netD.UNCOND_DNET is not None:
            #     real_logits = netD.UNCOND_DNET(real_features)
            #     fake_logits = netD.UNCOND_DNET(fake_features)
            #     real_errD = nn.BCELoss()(real_logits, real_labels)
            #     fake_errD = nn.BCELoss()(fake_logits, fake_labels)
            #     errD = ((real_errD + cond_real_errD) / 2. +
            #             (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)

            # forward D
            real_l_code = LMDis(landmarks.view(bs, -1))
            fake_l_code = LMDis(LMs.detach())

            ######## losses ###########
            # cond real loss
            cond_real_logits = LMDis.cond_judge(real_l_code, sent_emb)
            cond_real_errD = criterion(cond_real_logits, real_label)
            # cond fake loss
            cond_fake_logits = LMDis.cond_judge(fake_l_code, sent_emb)
            cond_fake_errD = criterion(cond_fake_logits, fake_label)
            # match loss
            cond_wrong_logits = LMDis.cond_judge(real_l_code[:(bs - 1)], sent_emb[1:bs])
            cond_wrong_errD = criterion(cond_wrong_logits, fake_label[1:bs])

            # uncond real/fake loss
            real_logits = LMDis.uncond_judge(real_l_code)
            fake_logits = LMDis.uncond_judge(fake_l_code)
            real_errD = criterion(real_logits, real_label)
            fake_errD = criterion(fake_logits, fake_label)
            errD = ((real_errD + cond_real_errD) / 2. +
                    (fake_errD + cond_fake_errD + cond_wrong_errD) / 3.)

            # backward and update D
            errD_total += errD.data
            errD.backward()
            optimizerD.step()

            # Clip weights of discriminator
            if cfg.TRAIN.IF_CLIP:
                for p in LMDis.parameters():
                    p.data.clamp_(-cfg.GAN.CLIP, cfg.GAN.CLIP)

            # update G network every cfg.GAN.N_CRITIC steps
            if iteration % cfg.GAN.N_CRITIC == 0:
                LMGen.zero_grad()
                ## forward G
                LMs, scores = LMGen(sent_emb, words_emb, noise, masks[:, 1:])

                # L1 Loss
                errL1 = L1Loss(landmarks.view(bs, -1), LMs)

                ## forward D
                l_code = LMDis(LMs)
                # cond loss
                cond_logits = LMDis.cond_judge(l_code, sent_emb)
                cond_errG = criterion(cond_logits, real_label)
                # uncond loss
                uncond_logits = LMDis.cond_judge(l_code)
                uncond_errG = criterion(uncond_logits, real_label)
                errG = cond_errG + uncond_errG
                errG += errL1

                errG_total += errG.data
                errL1_total += errL1.data
                # backward and update G
                errG.backward()
                optimizerG.step()

            if iteration > 0 and iteration % 20 == 0:
                # cfg.LOG.info(
                #     '| epoch {:3d} | {:5d}/{:5d} batches | errD {:8.7f} r_acc {:6.5f} f_acc {:6.5f} | errG {:8.7f} {:8.7f} | lr {:6.5f}' \
                #         .format(epoch, iteration, len(trnDataloader), errD_total / iteration, torch.mean(real_logits), \
                #                 torch.mean(fake_logits), errG_total / iteration, errL1_total / iteration,
                #                 optimizerD.param_groups[0]["lr"]))
                print(
                    '| epoch {:3d} | {:5d}/{:5d} batches | errD {:8.7f} r_acc {:6.5f} f_acc {:6.5f} | errG {:8.7f} {:8.7f} | lr {:6.5f}' \
                        .format(epoch, iteration, len(trnDataloader), errD_total / iteration, torch.mean(real_logits), \
                                torch.mean(fake_logits), errG_total / iteration, errL1_total / iteration,
                                optimizerD.param_groups[0]["lr"]))
        schedulerD.step()
        schedulerG.step()

        # savings
        if epoch > 0 and epoch % cfg.TRAIN.SAVE_PERIOD == 0:
            torch.save(LMGen.state_dict(), '%s/netG%d.pth' % (cfg.MODEL_DIR, epoch))
            torch.save(LMDis.state_dict(), '%s/netD%d.pth' % (cfg.MODEL_DIR, epoch))

            # save landmarks imgs
            im = build_lm_images(LMs[:4].detach().cpu().numpy())
            fullpath = '%s/lm_maps_f%d.png' % (cfg.IMG_DIR, epoch)
            im.save(fullpath)

            im = build_lm_images(landmarks[:4].detach().cpu().numpy())
            fullpath = '%s/lm_maps_r%d.png' % (cfg.IMG_DIR, epoch)
            im.save(fullpath)

            score = scores[:4].detach().cpu()

            # save self-attn weights
            t_score, idxs = torch.sort(score, 1, True)
            f = open('%s/lm_scoress%d.txt' % (cfg.IMG_DIR, epoch), "w")
            for i in range(4):
                f.write(str(score[i]) + '\n')
                for j in range(cfg.MAXLEN - 1):
                    f.write(idx2word[int(caps[i][j + 1])] + ' ')
                f.write("\n")
                f.write(str(masks[i][1:].detach().cpu()) + "\n")
                f.write(str(t_score[i]) + '\n')
                for idx in idxs[i]:
                    f.write(idx2word[int(caps[i][int(idx) + 1])] + ' ')
                f.write("\n")
            f.close()
        ######### eval #########
        LMGen.eval()
        MSELoss = torch.nn.MSELoss(reduction="sum")
        Loss = 0
        cnt = 0
        for iteration, data in enumerate(valDataloader):
            caps, cap_lens, masks, landmarks, keys = data
            bs = caps.shape[0]
            cnt += bs
            noise = Variable(torch.FloatTensor(bs, cfg.MODEL.Z_DIM)).to(cfg.device)

            cap_lens, sorted_cap_indices = \
                torch.sort(cap_lens, 0, True)

            masks = [masks[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
            masks = Variable(torch.cat(masks, 0)).to(cfg.device)
            caps = Variable(caps[sorted_cap_indices].squeeze()).to(cfg.device)
            keys = [keys[i] for i in sorted_cap_indices.numpy()]
            landmarks = [landmarks[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
            landmarks = Variable(torch.cat(landmarks, 0)).to(cfg.device)
            # --> bs x 28 x 2
            landmarks = landmarks.index_select(1, lm_idxs)
            landmarks = (landmarks - 128) / 128

            hidden = TextEncoder.init_hidden(bs)
            words_emb, sent_emb = TextEncoder(caps[:, 1:], cap_lens, hidden)

            noise.data.normal_(0, 1)
            LMs, scores = LMGen(sent_emb, words_emb, noise, masks[:, 1:])
            Loss += MSELoss(LMs * 128 + 128, landmarks.view(-1, len(lm_idxs) * 2) * 128 + 128) / (2 * len(lm_idxs))

        Loss = Loss / cnt
        if Loss < bestLoss:
            bestLoss = Loss
            torch.save(LMGen.state_dict(), '%s/netGBestMSE.pth' % (cfg.MODEL_DIR))
        elapsed = time.time() - start_time
        # cfg.LOG.info('| end epoch {:3d} | valid MSELoss / Best {:5.3f} / {:5.3f} | {:.2f} s/epoch' \
        #              .format(epoch, Loss, bestLoss, elapsed))
        print('| end epoch {:3d} | valid MSELoss / Best {:5.3f} / {:5.3f} | {:.2f} s/epoch' \
                     .format(epoch, Loss, bestLoss, elapsed))


def trainImGAN(ImageEncoder, TextEncoder, netG, netsD, optimizerG, optimizersD,
               trnDataloader, testDataloader, idx2word, _epoch, cfg):
    ######## pretrained Landmark detector for Landmark loss ######
    # LmDetector = cpm_vgg16(69)
    # state = torch.load("/nfs/users/wangkun/FLG-GAN/data/数据集/处理工具/pytorch_face_landmark-master/checkpoint/cpm_vgg16-epoch-049-050.pth")
    # LmDetector.load_state_dict(state)
    # LmDetector = LmDetector.to(cfg.device)
    # LmDetector.eval()

    TextEncoder = TextEncoder.to(cfg.device)
    ImageEncoder = ImageEncoder.to(cfg.device)
    TextEncoder.eval()
    ImageEncoder.eval()
    netG = netG.to(cfg.device)
    for i in range(len(netsD)):
        netsD[i] = netsD[i].to(cfg.device)
    ######## train
    avg_param_G = copy_G_params(netG)
    for epoch in range(_epoch, _epoch + cfg.TRAIN.MAX_EPOCH):
        netG.train()
        start_time = time.time()
        for iteration, data in enumerate(trnDataloader):
            imgs, caps, cap_lens, masks, landmarks, keys, bboxs, segMaps = data
            batch_size = masks.shape[0]
            cap_lens, sorted_cap_indices = \
                torch.sort(cap_lens, 0, True)
            for i in range(len(imgs)):
                imgs[i] = imgs[i][sorted_cap_indices]
                imgs[i] = Variable(imgs[i]).to(cfg.device)
            masks = [masks[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
            masks = Variable(torch.cat(masks, 0)).to(cfg.device)
            bboxs = [bboxs[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
            bboxs = Variable(torch.cat(bboxs, 0)).to(cfg.device)
            caps = Variable(caps[sorted_cap_indices].squeeze()).to(cfg.device)
            keys = [keys[i] for i in sorted_cap_indices.numpy()]
            segMaps = [segMaps[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
            segMaps = Variable(torch.cat(segMaps, 0)).to(cfg.device)
            landmarks = [landmarks[i].unsqueeze(0) for i in sorted_cap_indices.numpy()]
            landmarks = Variable(torch.cat(landmarks, 0)).to(cfg.device)

            # labels
            real_labels = Variable(torch.FloatTensor(batch_size).fill_(1)).to(cfg.device)
            fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0)).to(cfg.device)
            match_labels = Variable(torch.LongTensor(range(batch_size))).to(cfg.device)
            noise = Variable(torch.FloatTensor(batch_size, cfg.MODEL.Z_DIM)).to(cfg.device)
            fixed_noise = Variable(torch.FloatTensor(batch_size, cfg.MODEL.Z_DIM)).normal_(0, 1).to(cfg.device)

            # mask 5th roi for faces without hair appearance
            roi_masks = bboxs.sum(2) == 0

            bboxs = bboxs * 17 / 256
            bboxs_ori = bboxs
            bboxs = bboxs.reshape(batch_size * 5, 4)

            batch_idxs = torch.arange(batch_size).reshape(batch_size, 1).repeat(1, 5).reshape(batch_size * 5, 1).to(
                cfg.device)
            bboxs = torch.cat((batch_idxs, bboxs), 1)

            #######################################################
            # extract text features
            ######################################################
            hidden = TextEncoder.init_hidden(batch_size)
            words_embs, sent_embs = TextEncoder(caps[:, 1:], cap_lens, hidden)
            words_embs, sent_embs = words_embs.detach(), sent_embs.detach()

            #######################################################
            # Generate fake images
            ######################################################
            noise.data.normal_(0, 1)
            fake_imgs, _, c_code, mu, logvar = netG(noise, sent_embs, words_embs, masks[:, 1:], segMaps)

            #######################################################
            # Update D network
            ######################################################
            errD_total = 0
            D_logs = ''
            for i in range(len(netsD)):
                netsD[i].zero_grad()
                if cfg.SEGD_FLAG:
                    errD, log = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                   sent_embs, real_labels, fake_labels, segMaps)
                else:
                    errD, log = discriminator_loss(netsD[i], imgs[i], fake_imgs[i],
                                                   sent_embs, real_labels, fake_labels)
                # backward and update parameters
                errD.backward()
                optimizersD[i].step()
                errD_total += errD.item()
                D_logs += 'errD%d: %.2f ' % (i, errD.item())
                D_logs += log

            netG.zero_grad()
            if cfg.SEGD_FLAG:
                errG_total, G_logs = generator_loss(netsD, ImageEncoder, fake_imgs, real_labels, words_embs,
                                                    sent_embs, match_labels, masks[:, 1:], roi_masks, bboxs, cap_lens,
                                                    segMaps)
            else:
                errG_total, G_logs = generator_loss(netsD, ImageEncoder, fake_imgs, real_labels, words_embs,
                                                    sent_embs, match_labels, masks[:, 1:], roi_masks, bboxs, cap_lens)
            # ablation no smoothing loss
            kl_loss = KL_loss(mu, logvar)
            errG_total += kl_loss
            G_logs += 'kl_loss: %.2f ' % kl_loss.item()
            locs = None
            ####  Landmark loss ####
            # if epoch >= cfg.LM_LOSS_EPOCH:
            #     lm_loss, locs = landmark_loss(LmDetector, fake_imgs[-1], landmarks)
            #     errG_total += lm_loss
            #     G_logs += 'lm_loss: %.2f ' % lm_loss.item()
            # backward and update parameters
            errG_total.backward()
            optimizerG.step()
            for p, avg_p in zip(netG.parameters(), avg_param_G):
                avg_p.mul_(0.999).add_(p.data, alpha=0.001)
            if iteration > 0 and iteration % 60 == 0:
                # cfg.LOG.info('| epoch {:3d} | {:5d}/{:5d} batches | '.format(epoch, iteration, len(trnDataloader)) +
                #              D_logs + ' | ' + G_logs + ' | Loss D:{:5.2f} | Loss G: {:5.2f} '.format(errD_total,
                #                                                                                      errG_total.item()))
                print('| epoch {:3d} | {:5d}/{:5d} batches | '.format(epoch, iteration, len(trnDataloader)) +
                             D_logs + ' | ' + G_logs + ' | Loss D:{:5.2f} | Loss G: {:5.2f} '.format(errD_total,
                                                                                                     errG_total.item()))

        if epoch % cfg.TRAIN.SAVE_PERIOD == 0:
            ## save models
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save(netG.state_dict(), '%s/netG_avg_epoch_%d.pth' % (cfg.MODEL_DIR, epoch))
            # for i in range(len(netsD)):
            #     netD = netsD[i]
            #     torch.save(netD.state_dict(), '%s/netD_epoch_%d_%d.pth' % (cfg.MODEL_DIR, epoch, i))
            #############
            # save images
            #############
            saveImages(cfg, netG, ImageEncoder, fixed_noise, sent_embs, words_embs, caps, masks, roi_masks, bboxs, bboxs_ori,
                       cap_lens,
                       idx2word, epoch, batch_size, segMaps, name="avg")
            load_params(netG, backup_para)
        elapsed = time.time() - start_time
        # cfg.LOG.info('| end epoch {:3d} | {:.2f} s/epoch | lr {:.5f}| ' \
        #              .format(epoch, elapsed, optimizerG.param_groups[0]["lr"]))
        print('| end epoch {:3d} | {:.2f} s/epoch | lr {:.5f}| ' \
                     .format(epoch, elapsed, optimizerG.param_groups[0]["lr"]))


def saveImages(cfg, netG, ImageEncoder, fixed_noise, sent_embs, words_embs, caps, masks, roi_masks, bboxs, bboxs_ori,
               cap_lens,
               idx2word, epoch, batch_size, segmaps, name="", locs=None):
    fake_imgs, att_maps, c_code, _, _ = netG(fixed_noise, sent_embs, words_embs, masks[:, 1:], segmaps)
    # for G att maps
    for i in range(len(att_maps)):
        if len(fake_imgs) > 1:
            imgs = fake_imgs[i + 1].detach().cpu()
            pre_imgs = fake_imgs[i].detach().cpu()
        else:
            imgs = fake_imgs[0].detach().cpu()
            pre_imgs = None
        if locs is not None:
            locs = locs.detach().cpu()
        img_set, _ = \
            build_super_images(cfg, imgs.cpu(), caps[:, 1:], idx2word, att_maps[i], att_maps[i].size(2), pre_imgs=pre_imgs,
                               locs=locs)
        if img_set is not None:
            im = Image.fromarray(img_set)
            fullpath = '%s/G_%s_maps%d_stage%d.png' % (cfg.IMG_DIR, name, epoch, i + 1)
            im.save(fullpath)
    # for pretrain att
    imgs = fake_imgs[-1].detach()
    img_set = None
    segImgs = None
    if cfg.SEG_FLAG:
        segImgs = segmaps.detach().cpu()
        segImgs[:, 2, :] += segImgs[:, 3, :]
        segImgs[:, 1, :] += segImgs[:, 3, :]
        segImgs = segImgs[:, :3, :]
        segImgs.mul_(2).sub_(1)
    if cfg.ROI_FLAG:
        region_features, _ = ImageEncoder(imgs, bboxs)
        _, _, att_maps = words_loss(region_features.detach(), words_embs.detach(),
                                    None, cap_lens, masks[:, 1:], roi_masks, batch_size)
        img_set, _ = \
            build_roiattn_images(cfg, imgs.cpu(), caps[:, 1:], idx2word, att_maps, cfg.ATT_SZE, bboxs_ori.cpu(), segImgs)
    else:
        region_features, _ = ImageEncoder(imgs, None)
        att_sze = region_features.size(2)

        _, _, att_maps = grid_words_loss(region_features.detach(),
                                         words_embs.detach(),
                                         None, cap_lens,
                                         None, None, batch_size)
        img_set, _ = \
            build_super_images(cfg, imgs.cpu(), caps[:, 1:], idx2word, att_maps, att_sze, segImgs)
    if img_set is not None:
        im = Image.fromarray(img_set)
        fullpath = '%s/D_%s_map%d.png' % (cfg.IMG_DIR, name, epoch)
        im.save(fullpath)
