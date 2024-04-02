import os

from PIL.Image import Image
import torch
from torch import nn
import argparse
from torch._C import device
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import dateutil.tz

from data.dataset import FaceTextDataset
from models.encoders import FACE_ENCODER, RNN_ENCODER
from models.netG import G_NET
from models.netD import D_NET64, D_NET128, D_NET256, SEG_NET
from tools.config import cfg, cfg_from_file
from engine.trainer import trainImGAN
from tools.logger import setupLogger
from tools.weight import weights_init


def main():
    parser = argparse.ArgumentParser("FLG-GAN")
    parser.add_argument(
        "-gpu_id",
        type=str,
        default="4,5,6",
        dest="gpu_id"
    )
    parser.add_argument(
        "-cfg",
        default="./cfgs/faceGen.yml",
        dest="cfg"
    )

    args = parser.parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    # torch.cuda.set_device(args.gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s/%s' % \
                 (cfg.DATASET_NAME, "FaceGen", timestamp)
    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    cfg.IMG_DIR = image_dir
    cfg.MODEL_DIR = model_dir
    os.makedirs(model_dir)
    os.makedirs(image_dir)
    logger = setupLogger("FLG-GAN", output_dir)
    logger.info(cfg)
    cfg.LOG = logger

    ## Datasets
    trnDataset = FaceTextDataset(cfg.DATAPATH, cfg.TREE.BASE_SIZE, cfg.MAXLEN, split="train")
    testDataset = FaceTextDataset(cfg.DATAPATH, cfg.TREE.BASE_SIZE, cfg.MAXLEN, split="test")
    trnDataloader = DataLoader(trnDataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    testDataloader = DataLoader(testDataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4)

    ## models
    # TextEncoder = T_ENCODER(trnDataset.vocab_size, cfg.MODEL.E_DIM)
    TextEncoder = RNN_ENCODER(trnDataset.vocab_size, cfg=cfg)
    state_dict = torch.load(cfg.ENCODER_PATH, map_location=lambda storage, loc: storage)
    TextEncoder.load_state_dict(state_dict)
    for param in TextEncoder.parameters():
        param.requires_grad = False
    ImageEncoder = FACE_ENCODER(cfg.MODEL.E_DIM, cfg)
    # ImageEncoder = CNN_ENCODER(cfg.MODEL.E_DIM)
    img_encoder_path = cfg.ENCODER_PATH.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    ImageEncoder.load_state_dict(state_dict)
    for param in ImageEncoder.parameters():
        param.requires_grad = False

    netG = G_NET(cfg)
    netSEG = None
    if cfg.SEGD_FLAG:
        netSEG = SEG_NET(cfg)
    netsD = []
    if cfg.TREE.BRANCH_NUM > 0:
        netsD.append(D_NET64(cfg, SegNet=netSEG))
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128(cfg, SegNet=netSEG))
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256(cfg, SegNet=netSEG))

    netG.apply(weights_init)
    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
    epoch = 0
    if cfg.CHECKPOINT_PATH != "":
        state_dict = \
            torch.load(cfg.CHECKPOINT_PATH, map_location=lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load G from: ', cfg.CHECKPOINT_PATH)
        istart = cfg.CHECKPOINT_PATH.rfind('_') + 1
        iend = cfg.CHECKPOINT_PATH.rfind('.')
        epoch = int(cfg.CHECKPOINT_PATH[istart:iend])
        for i in range(len(netsD)):
            s_tmp = cfg.CHECKPOINT_PATH[:cfg.CHECKPOINT_PATH.rfind('/')]
            Dname = '%s/netD_epoch_%d_%d.pth' % (s_tmp, epoch, i)
            print('Load D from: ', Dname)
            state_dict = \
                torch.load(Dname, map_location=lambda storage, loc: storage)
            netsD[i].load_state_dict(state_dict)

    optimizerG = optim.Adam(netG.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS)
    optimizersD = []
    for i in range(len(netsD)):
        opt = optim.Adam(filter(lambda p: p.requires_grad, netsD[i].parameters()),
                         lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS)
        optimizersD.append(opt)

    # ImageEncoder = nn.DataParallel(ImageEncoder)
    # TextEncoder = nn.DataParallel(TextEncoder)
    # netG = nn.DataParallel(netG)
    # for i in range(len(netsD)):
    #     netsD[i] = nn.DataParallel(netsD[i])

    trainImGAN(ImageEncoder, TextEncoder, netG, netsD, optimizerG, optimizersD,
               trnDataloader, testDataloader, trnDataset.idx2word, epoch, cfg)


if __name__ == "__main__":
    main()
