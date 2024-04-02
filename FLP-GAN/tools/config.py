from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

# Training options
__C.TRAIN = edict()
__C.TRAIN.LR = 0.0002
__C.TRAIN.BETAS = (0.5, 0.999)
__C.TRAIN.DECAY_PERIOD = 10
__C.TRAIN.DECAY_RATE = 0.9
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 601
__C.TRAIN.IF_CLIP = False
__C.TRAIN.SAVE_PERIOD = 20
__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 4.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.LAMBDA = 5.0
__C.TRAIN.SMOOTH.GAMMA4 = 10.0

# Model options
__C.MODEL = edict()
__C.MODEL.E_DIM = 256
__C.MODEL.LM_DF_DIM = 128
__C.MODEL.Z_DIM = 100
__C.MODEL.C_DIM = 100
__C.MODEL.DF_DIM = 64
__C.MODEL.GF_DIM = 32
__C.MODEL.RES_NUM = 2

__C.GAN = edict()
__C.GAN.CLIP = 0.2
__C.GAN.N_CRITIC = 5

# Base options
__C.DATAPATH = "/data/wangkun/project/FLG-GAN/faces"
__C.MAXLEN = 81
__C.MAXSHIFT = 5.0
__C.EMBEDDING_NUM = 2
__C.DATASET_NAME = "FACE"
__C.ROI_FLAG = True
__C.SEG_FLAG = True
__C.SEGD_FLAG = False
__C.LM_FLAG = False
__C.LM_LOSS_EPOCH = 100
__C.ATT_SZE = 17
__C.CHECKPOINT_PATH = ""
__C.ENCODER_PATH = "/data/wangkun/project/FLG-GAN/output/FACE_pretrain/2021_01_12_18_47_19/Model/text_encoder_bestSL.pth"
__C.LMG_PATH = "/data/wangkun/project/FLG-GAN/output/FACE_LMGen/2021_01_18_17_06_11/Model/netD1800.pth"
__C.NETG_PATH = "/data/wangkun/project/FLG-GAN/output/FACE_FaceGen/2021_05_13_13_34_30/Model/netG_avg_epoch_660.pth"

__C.TREE = edict()
__C.TREE.BRANCH_NUM = 3
__C.TREE.BASE_SIZE = 299
__C.device = 'cuda'

# __C.LOG = edict()
# __C.LOG.info = ''


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b.keys():
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    _merge_a_into_b(yaml_cfg, __C)
