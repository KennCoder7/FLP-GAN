from PIL.Image import NONE
import torch
import torch.nn as nn
from torch.autograd import Variable

from blocks import GLU, upBlock, conv3x3, ResBlock, SegBlock


# from config import cfg


# ############## G networks ###################
class CA_NET(nn.Module):
    def __init__(self, cfg):
        super(CA_NET, self).__init__()
        self.cfg = cfg
        self.t_dim = cfg.MODEL.E_DIM
        self.c_dim = cfg.MODEL.C_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 4, bias=True)
        self.relu = GLU()


    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.cfg.device)
        eps = Variable(eps)
        return eps.mul(std).add_(mu)  # sample condition vector

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, ncf, cfg):
        super(INIT_STAGE_G, self).__init__()
        self.cfg = cfg
        self.gf_dim = ngf
        self.in_dim = cfg.MODEL.Z_DIM + ncf

        self.define_module()


    def define_module(self):
        nz, ngf = self.in_dim, self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        if self.cfg.SEG_FLAG:
            self.segBlock = SegBlock(ngf)

    def forward(self, z_code, c_code, segMaps=None):
        """Stage-I.
        Args:
            z_code: noise vector
            c_code: condition vector
            segMaps: segment map
        Returns:
           out_code64: feature map ngf/16 x 64 x 64
        """
        c_z_code = torch.cat((c_code, z_code), 1)
        # state size ngf x 4 x 4
        out_code = self.fc(c_z_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        if self.cfg.SEG_FLAG and segMaps is not None:
            out_code = self.segBlock(out_code, segMaps)
        # state size ngf/2 x 8 x 8
        out_code = self.upsample1(out_code)
        # state size ngf/4 x 16 x 16
        out_code = self.upsample2(out_code)
        # state size ngf/8 x 32 x 32
        out_code32 = self.upsample3(out_code)
        # state size ngf/16 x 64 x 64
        out_code64 = self.upsample4(out_code32)

        return out_code64


class Memory(nn.Module):
    def __init__(self):
        super(Memory, self).__init__()

    def forward(self, query, key, value, wMask):
        """ memory response
            Args:
                query: feature of previous generated image 
                key: key of memory slots
                value: value of memory slots
                wMask: stopword masks
            Returns:
                weightedContext: context feature
                weight: pixel-word attention map
        """
        ih, iw = query.size(2), query.size(3)
        queryL = ih * iw
        batch_size, sourceL = key.size(0), key.size(2)

        # --> batch x queryL x idf
        q = query.view(batch_size, -1, queryL)
        qT = torch.transpose(q, 1, 2).contiguous()
        kT = key

        # Get weight
        # (batch x queryL x idf)(batch x idf x sourceL)-->batch x queryL x sourceL
        weight = torch.bmm(qT, kT)

        # --> batch*queryL x sourceL
        weight = weight.view(batch_size * queryL, sourceL)

        # mask stop words
        # batch_size x sourceL --> batch_size*queryL x sourceL
        wMask = wMask.repeat(queryL, 1)
        weight.data.masked_fill_(wMask.bool(), -float('inf'))

        weight = torch.nn.functional.softmax(weight, dim=1)

        # --> batch x queryL x sourceL
        weight = weight.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        weight = torch.transpose(weight, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL) --> batch x idf x queryL
        weightedContext = torch.bmm(value, weight)  #
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        weight = weight.view(batch_size, -1, ih, iw)

        return weightedContext, weight


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, nef, ncf, cfg):
        super(NEXT_STAGE_G, self).__init__()
        self.cfg = cfg
        self.gf_dim = ngf
        self.ef_dim = nef
        self.cf_dim = ncf
        self.num_residual = cfg.MODEL.RES_NUM
        self.define_module()


    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.cfg.MODEL.RES_NUM):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim

        ### memory writing gate
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.A = nn.Linear(self.ef_dim, 1, bias=False)
        self.B = nn.Linear(self.gf_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        ### transform image feature
        self.M_r = nn.Sequential(
            nn.Conv1d(ngf, ngf * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        ### transform word feature
        self.M_w = nn.Sequential(
            nn.Conv1d(self.ef_dim, ngf * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        ### get memory slot keys
        self.key = nn.Sequential(
            nn.Conv1d(ngf * 2, ngf, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        ### get memory slot values
        self.value = nn.Sequential(
            nn.Conv1d(ngf * 2, ngf, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        ### memory query
        self.memory_operation = Memory()
        ### memory response gate
        self.response_gate = nn.Sequential(
            nn.Conv2d(self.gf_dim * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        self.residual = self._make_layer(ResBlock, ngf * 2)
        self.upsample = upBlock(ngf * 2, ngf)
        ### fuse segment map
        if self.cfg.SEG_FLAG:
            self.segBlock = SegBlock(ngf * 2)

    def forward(self, h_code, c_code, word_embs, wMask, segMaps=None):
        """ memory response
            Args:
                h_code: feature of previous generated image   (batch x idf x ih x iw (queryL=ihxiw))
                c_code: condition vector   (batch x idf x queryL)
                word_embs: word features    (batch x cdf x sourceL (sourceL=seq_len))     
                wMask: stopword masks
                segMaps: face segment maps
            Returns:
                out_code: context feature
                att: pixel-word attention map
        """
        #### Memory Writing ####
        word_embs_T = torch.transpose(word_embs, 1, 2).contiguous()
        # batch x sourceL x cdf 
        h_code_avg = self.avg(h_code).detach()
        h_code_avg = h_code_avg.squeeze(3)
        # batch x idf x 1
        h_code_avg_T = torch.transpose(h_code_avg, 1, 2).contiguous()
        # batch x 1 x idf
        ### writing gate for image and word ###
        gate1 = torch.transpose(self.A(word_embs_T), 1, 2).contiguous()
        # batch x 1 x sourceL 
        gate2 = self.B(h_code_avg_T).repeat(1, 1, word_embs.size(2))
        # batch x 1 x sourceL
        writing_gate = torch.sigmoid(gate1 + gate2)  # eq(7) in DMGAN paper
        h_code_avg = h_code_avg.repeat(1, 1, word_embs.size(2))
        # batch x idf x sourceL
        memory = self.M_w(word_embs) * writing_gate + self.M_r(h_code_avg) * (1 - writing_gate)

        #### Key Addressing and Value Reading ###
        key = self.key(memory)
        value = self.value(memory)
        memory_out, att = self.memory_operation(h_code, key, value, wMask)

        ##### gated Response ####
        response_gate = self.response_gate(torch.cat((h_code, memory_out), 1))
        h_code_new = h_code * (1 - response_gate) + response_gate * memory_out
        h_code_new = torch.cat((h_code_new, h_code_new), 1)  # update image feature

        if self.cfg.SEG_FLAG and segMaps is not None:  # fuse face segment maps
            h_code_new = self.segBlock(h_code_new, segMaps)

        out_code = self.residual(h_code_new)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code, att


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),  # conv to 3-chanel
            nn.Tanh()  # normalize
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self, cfg):
        super(G_NET, self).__init__()
        self.cfg = cfg
        ngf = cfg.MODEL.GF_DIM
        nef = cfg.MODEL.E_DIM
        ncf = cfg.MODEL.C_DIM
        self.ca_net = CA_NET(cfg)
        if cfg.SEG_FLAG:
            self.segBlock = SegBlock(ngf)
            self.residual = ResBlock(ngf)
        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(ngf * 16, ncf, cfg)
            self.img_net1 = GET_IMAGE_G(ngf)
        # gf x 64 x 64
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G(ngf, nef, ncf, cfg)
            self.img_net2 = GET_IMAGE_G(ngf)
        if cfg.TREE.BRANCH_NUM > 2:
            self.h_net3 = NEXT_STAGE_G(ngf, nef, ncf, cfg)
            self.img_net3 = GET_IMAGE_G(ngf)


    def forward(self, z_code, sent_emb, word_embs, mask, segMaps=None, cCode=None):
        """ 3stage GAN
            Args:
                z_code: noise vector
                sent_emb: sentence feature 
                word_embs: word features 
                mask: stopword masks
                segMaps: face segment maps
                (cCode): condition vector when edit
            Returns:
                fake_imgs: generated imgs list (64x64, 128x128, 256x256)
                att_maps: pixel-word attention map
                c_code: condition vector sampled when genereting these imgs
                mu,logvar:  CA distribution parameter
        """
        fake_imgs = []
        att_maps = []
        c_code, mu, logvar = self.ca_net(sent_emb)  # condition augment
        if cCode is not None:  # when editing, no need to use new codintion vector
            c_code = cCode
        if self.cfg.TREE.BRANCH_NUM > 0:  # stage 1
            h_code1 = self.h_net1(z_code, c_code, segMaps)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if self.cfg.TREE.BRANCH_NUM > 1:  # stage 2
            h_code2, att1 = \
                self.h_net2(h_code1, c_code, word_embs, mask, segMaps)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
            if att1 is not None:
                att_maps.append(att1)
        if self.cfg.TREE.BRANCH_NUM > 2:  # stage 3
            h_code3, att2 = \
                self.h_net3(h_code2, c_code, word_embs, mask, segMaps)
            if self.cfg.SEG_FLAG and segMaps is not None:  # one more segment fuse layer
                h_code3 = self.segBlock(h_code3, segMaps)
                h_code3 = self.residual(h_code3)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
            if att2 is not None:
                att_maps.append(att2)

        return fake_imgs, att_maps, c_code, mu, logvar
