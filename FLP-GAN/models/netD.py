import torch
import torch.nn as nn
# from config import cfg

from blocks import Block3x3_leakRelu, downBlock, encode_image_by_16times


class D_GET_LOGITS(nn.Module):
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        if self.bcondition:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + nef, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        """ discrimination
            Args:
                h_code: image feature
                (c_code): sentence feature
            Returns:
                output: condition/uncondition validity
        """
        if self.bcondition and c_code is not None:
            # conditioning output
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((h_code, c_code), 1)  # concat image feature and sentence feature
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)  # jointly process
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)
        return output.view(-1)


### segment discrimination ###
class SEG_GET_LOGITS(nn.Module):
    def __init__(self, ndf):
        super(SEG_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.jointConv = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.outlogits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

    def forward(self, h_code, seg_code):
        # state size (2*ngf) x 4 x 4
        h_s_code = torch.cat((h_code, seg_code), 1)
        # state size ngf x in_size x in_size
        h_s_code = self.jointConv(h_s_code)

        output = self.outlogits(h_s_code)
        return output.view(-1)


### segment discrimination ###
class SEG_NET(nn.Module):
    def __init__(self, cfg):
        super(SEG_NET, self).__init__()
        ndf = cfg.MODEL.DF_DIM
        self.seg_code_s16 = encode_image_by_16times(ndf, 3)
        self.seg_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.seg_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.seg_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.seg_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

    def forward(self, segmaps):
        segmaps = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True)(segmaps[:, :3])
        x_code4 = self.seg_code_s16(segmaps)
        return x_code4


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, cfg, b_jcu=True, SegNet=None):
        super(D_NET64, self).__init__()
        ndf = cfg.MODEL.DF_DIM
        nef = cfg.MODEL.E_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.SEG_DNET = None
        if cfg.SEGD_FLAG and SegNet is not None:
            self.SEG_NET = SegNet
            self.SEG_DNET = SEG_GET_LOGITS(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)  ## hold a uncondition Dnet
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)  ## hold a condition Dnet
        self.cfg = cfg
    def forward(self, img, segmaps=None):
        """ discrimination
            Args:
                img: image tensor
                (segmaps): segment maps
            Returns:
               x_code4: image feature
               seg_code: segment maps feature
        """
        x_code4 = self.img_code_s16(img)  # 4 x 4 x 8df
        seg_code = None
        if self.cfg.SEGD_FLAG and self.SEG_NET is not None and segmaps is not None:
            seg_code = self.SEG_NET(segmaps)
        return x_code4, seg_code


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self, cfg, b_jcu=True, SegNet=None):
        super(D_NET128, self).__init__()
        ndf = cfg.MODEL.DF_DIM
        nef = cfg.MODEL.E_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.SEG_DNET = None
        if cfg.SEGD_FLAG and SegNet is not None:
            self.SEG_NET = SegNet
            self.SEG_DNET = SEG_GET_LOGITS(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)  ## hold a uncondition Dnet
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)  ## hold a condition Dnet
        self.cfg = cfg


    def forward(self, img, segmaps=None):
        """ discrimination
            Args:
                img: image tensor
                (segmaps): segment maps
            Returns:
               x_code4: image feature
               seg_code: segment maps feature
        """
        x_code8 = self.img_code_s16(img)  # 8 x 8 x 8df
        x_code4 = self.img_code_s32(x_code8)  # 4 x 4 x 16df
        x_code4 = self.img_code_s32_1(x_code4)  # 4 x 4 x 8df
        seg_code = None
        if self.cfg.SEGD_FLAG and self.SEG_NET is not None and segmaps is not None:
            seg_code = self.SEG_NET(segmaps)
        return x_code4, seg_code


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self, cfg, b_jcu=True, SegNet=None):
        super(D_NET256, self).__init__()
        ndf = cfg.MODEL.DF_DIM
        nef = cfg.MODEL.E_DIM
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.SEG_DNET = None
        if cfg.SEGD_FLAG and SegNet is not None:
            self.SEG_NET = SegNet
            self.SEG_DNET = SEG_GET_LOGITS(ndf)
        if b_jcu:
            self.UNCOND_DNET = D_GET_LOGITS(ndf, nef, bcondition=False)  ## hold a uncondition Dnet
        else:
            self.UNCOND_DNET = None
        self.COND_DNET = D_GET_LOGITS(ndf, nef, bcondition=True)  ## hold a uncondition Dnet
        self.cfg = cfg

    def forward(self, img, segmaps=None):
        """ discrimination
            Args:
                img: image tensor
                (segmaps): segment maps
            Returns:
               x_code4: image feature
               seg_code: segment maps feature
        """
        x_code16 = self.img_code_s16(img)
        x_code8 = self.img_code_s32(x_code16)
        x_code4 = self.img_code_s64(x_code8)
        x_code4 = self.img_code_s64_1(x_code4)
        x_code4 = self.img_code_s64_2(x_code4)
        seg_code = None
        if self.cfg.SEGD_FLAG and self.SEG_NET is not None and segmaps is not None:
            seg_code = self.SEG_NET(segmaps)
        return x_code4, seg_code
