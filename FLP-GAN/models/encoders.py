import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch
import torchvision
from torch.autograd import Variable
from torchvision import models
from facenet_pytorch import InceptionResnetV1
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from blocks import conv1x1, conv3x3
# from config import cfg

class FACE_ENCODER(nn.Module):
    def __init__(self, nef, cfg):
        super(FACE_ENCODER, self).__init__()
        self.cfg = cfg
        self.nef = nef      # embedding size 
        model = InceptionResnetV1(pretrained='vggface2')    # facenet backbone
        for param in model.parameters():        # fix backbone params
            param.requires_grad = False
        self.define_module(model)
        self.init_trainable_weights()

    
    def define_module(self, model):
        self.conv2d_1a = model.conv2d_1a
        self.conv2d_2a = model.conv2d_2a
        self.conv2d_2b = model.conv2d_2b
        self.maxpool_3a = model.maxpool_3a
        self.conv2d_3b = model.conv2d_3b
        self.conv2d_4a = model.conv2d_4a
        self.conv2d_4b = model.conv2d_4b
        self.repeat_1 = model.repeat_1
        self.mixed_6a = model.mixed_6a
        self.repeat_2 = model.repeat_2
        self.mixed_7a = model.mixed_7a
        self.repeat_3 = model.repeat_3
        self.block8 = model.block8
        self.avgpool_1a = model.avgpool_1a

        if self.cfg.ROI_FLAG:
            self.RoIAlign = torchvision.ops.RoIAlign(3, 1.0, -1) # for ROI attn
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.roi_fc = nn.Linear(896, self.nef)          # unified embed
        else:
            self.emb_features = conv1x1(896, self.nef)    # for grid attn, unified embed
        self.emb_cnn_code = nn.Linear(1792, self.nef)    # unified embed global feature 
        
    def forward(self, x, bboxs):
        """Extract Face Image features.
        Args:
            x: face imgs.
            bboxs: facial bounding boxs list.
        Returns:
            features: ROI features
            face_code: face features
        """
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)(x)
        
        x = self.conv2d_1a(x)
        # 149 x 149 x 32
        x = self.conv2d_2a(x)
        # 147 x 147 x 32
        x = self.conv2d_2b(x)
        # 147 x 147 x 64
        x = self.maxpool_3a(x)
        # 73 x 73 x 64
        x = self.conv2d_3b(x)
        # 73 x 73 x 80
        x = self.conv2d_4a(x)
        # 71 x 71 x 192
        x = self.conv2d_4b(x)
        # 35 x 35 x 256
        x = self.repeat_1(x)
        # 35 x 35 x 256
        
        x = self.mixed_6a(x)
        # 17 x 17 x 896
        x = self.repeat_2(x)
        # 17 x 17 x 896
        
        features = x    # copy for roi branch
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        # 8 x 8 x 1792
        x = self.avgpool_1a(x)
        x = x.view(x.size(0), -1)

        face_code = self.emb_cnn_code(x)

        # for roi att
        if self.cfg.ROI_FLAG:
            pooled_rois = self.RoIAlign(features, bboxs)
            # 3 x 3 x 896
            features = self.avgpool(pooled_rois)
            # k x 896 x 1
            features = torch.flatten(features, 1)
            # k x 896
            features = self.roi_fc(features)
            # k x nef
            features = torch.reshape(features, (x.shape[0], 5, self.nef))
            # n x 5 x nef
            features = torch.transpose(features, 2, 1)
            # n x nef x 5
        else:
            if features is not None:    # for grid att
                features = self.emb_features(features)

        return features, face_code

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif classname.find('BatchNorm') != -1:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, nhidden=256, 
                 drop_prob=0.5, nlayers=1, bidirectional=True, cfg=None):
        super(RNN_ENCODER, self).__init__()
        self.cfg = cfg
        self.n_steps = cfg.MAXLEN - 1
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # Number of recurrent layers
        self.bidirectional = bidirectional
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()


    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
                Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()))
    

    def forward(self, captions, cap_lens, hidden):
        """Extract captions features.
        Args:
            captions: face captions (word idx list).
            cap_lens: caption length list.
            hidden: initial hidden state
        Returns:
            words_emb: word features
            sent_emb: sentence features
        """
        # input: torch.LongTensor of size batch x n_steps
        # mebedding and drop out
        # --> emb: batch x n_steps x ninput
        emb = self.drop(self.encoder(captions))

        # pack a PackedSequence object
        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # output (batch, seq_len, hidden_size * num_directions)
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)      # sent feature & word feature
        # unpack PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # pad to max length
        if words_emb.shape[2] < self.cfg.MAXLEN - 1:
            padd = torch.zeros(words_emb.shape[0], words_emb.shape[1], self.cfg.MAXLEN - 1 - words_emb.shape[2]).to(self.cfg.device)
            words_emb = torch.cat((words_emb, padd), 2)
        # --> batch x num_directions*hidden_size
        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb
