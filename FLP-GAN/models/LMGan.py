import torch
import torch.nn as nn
import torch.nn.functional as F

from spectral import SpectralNorm
from config import cfg


class LMGen(nn.Module):
    def __init__(self, dropout=0.2):
        super(LMGen, self).__init__()
        self.nzf = cfg.MODEL.Z_DIM
        self.nef = cfg.MODEL.E_DIM
        self.dropout = dropout
        self.define_modules()

    def define_modules(self):
        # encode sent
        self.t_enc = nn.Sequential(
            nn.Linear(self.nef, self.nef * 2),
            nn.LeakyReLU(0.2),
        )
        # encode noise
        self.z_enc = nn.Sequential(
            nn.Linear(self.nzf, self.nef),
            nn.LeakyReLU(0.2),
        )
        # encode words
        self.w_enc = nn.Sequential(
            nn.Linear(self.nef, self.nef),
            nn.LeakyReLU(0.2),
        )
        # self-attn score layer for words
        self.scorer = nn.Linear(self.nef, 1)
        self.Drop = nn.Dropout(self.dropout)

        # predict landmarks
        self.MLP = nn.Sequential(
            nn.Linear(self.nef * 4, self.nef * 4),
            nn.BatchNorm1d(self.nef * 4),
            nn.LeakyReLU(0.2),
            nn.Linear(self.nef * 4, self.nef * 2),
            nn.BatchNorm1d(self.nef * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(self.nef * 2, self.nef),
            nn.BatchNorm1d(self.nef),
            nn.LeakyReLU(0.2),
            nn.Linear(self.nef, 56)
        )

    def forward(self, sent_embs, word_embs, noise, masks):
        """generate landmarks coordinate.
        Args:
            sent_embs: sentence feature.
            word_embs: word feature.
            noise: sampled noise
            masks: stopwords masks
        Returns:
            lms: landmarks coordinate.
            scores: self-attention weight for words
        """
        batch_size = sent_embs.shape[0]
        # encode input
        t = self.t_enc(sent_embs)
        z = self.z_enc(noise)
        word_embs = word_embs.transpose(1, 2).contiguous()
        word_embs = self.w_enc(word_embs)
        word_embs = self.Drop(word_embs)
        # self attn on words
        scores = self.scorer(word_embs.view(-1, self.nef))
        scores = scores.view(batch_size, cfg.MAXLEN - 1)
        # mask stopwords
        scores.data.masked_fill_(masks.bool(), -float("inf"))
        # get weights 
        scores = F.softmax(scores, dim=1)
        # weighted sum
        w = scores.unsqueeze(-1).expand_as(word_embs).mul(word_embs).sum(1)
        # concat sent/word/noise
        x = torch.cat((t, z, w), 1)
        # predict landmarks coordinate
        x = self.MLP(x)
        lms = torch.tanh(x)
        return lms, scores


class GET_LOGITS(nn.Module):
    def __init__(self, cond=False):
        super(GET_LOGITS, self).__init__()
        self.nef = cfg.MODEL.E_DIM
        self.ndf = cfg.MODEL.LM_DF_DIM
        self.nzf = cfg.MODEL.Z_DIM
        self.cond = cond
        # jointly process sent feature and landmarks feature
        if cond:
            self.join = nn.Sequential(
                SpectralNorm(nn.Linear(self.ndf * 2, self.ndf)),
                nn.LeakyReLU(0.2)
            )
        # encode sentence feature
        self.t_enc = nn.Sequential(
            SpectralNorm(nn.Linear(self.nef, self.ndf * 2)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(self.ndf * 2, self.ndf)),
            nn.LeakyReLU(0.2)
        )
        # jointly judge landmarks
        self.judge = nn.Sequential(
            SpectralNorm(nn.Linear(self.ndf, self.ndf // 2)),
            nn.LeakyReLU(0.2),
            nn.Linear(self.ndf // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, l_code, t_code=None):
        """evaluate landmarks validity.
        Args:
            l_code: landmarks feature encoded by LMDis.
            t_code: sentence feature.
        Returns:
            logits: landmarks validity.
        """
        if self.cond and t_code is not None:
            t_code = self.t_enc(t_code)
            l_t_code = torch.cat((l_code, t_code), 1)
            l_t_code = self.join(l_t_code)
        else:
            l_t_code = l_code
        logits = self.judge(l_t_code)
        return logits.view(-1)


class LMDis(nn.Module):
    def __init__(self):
        super(LMDis, self).__init__()
        self.ndf = cfg.MODEL.LM_DF_DIM
        self.define_modules()

    def define_modules(self):
        # landmarks encoder
        self.LM_enc = nn.Sequential(
            SpectralNorm(nn.Linear(56, self.ndf * 2)),
            nn.LeakyReLU(0.2),
            SpectralNorm(nn.Linear(self.ndf * 2, self.ndf)),
            nn.LeakyReLU(0.2)
        )
        self.cond_judge = GET_LOGITS(True)  ## judge according to caption relevance
        self.uncond_judge = GET_LOGITS(False)  ## judge landmarks itself

    def forward(self, LMs):
        # encode landmarks 
        l_code = self.LM_enc(LMs)
        return l_code
