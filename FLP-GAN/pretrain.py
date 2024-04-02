import os
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import pprint
import datetime
import dateutil.tz

from data.dataset import FaceTextDataset
from models.encoders import FACE_ENCODER, RNN_ENCODER
from tools.config import cfg, cfg_from_file
from engine.trainer import trainEncoders
from tools.logger import setupLogger


def main():
    parser = argparse.ArgumentParser("FLG-GAN")
    parser.add_argument(
        "-gpu_id",
        default="0",
        dest="gpu_id"
    )
    parser.add_argument(
        "-cfg",
        default="./cfgs/pretrain.yml",
        dest="cfg"
    )
    args = parser.parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    # torch.cuda.set_device(args.gpu_id)
    cuda_str = "cuda:" + args.gpu_id
    cfg.device = torch.device(cuda_str if torch.cuda.is_available() else "cpu")
    cfg.TRAIN.BATCH_SIZE = 48
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s/%s' % \
                 (cfg.DATASET_NAME, "pretrain", timestamp)
    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    cfg.IMG_DIR = image_dir
    cfg.MODEL_DIR = model_dir
    os.makedirs(model_dir)
    os.makedirs(image_dir)
    logger = setupLogger("pretrain encoders", output_dir)
    logger.info(cfg)
    cfg.LOG = logger

    ## Datasets
    trnDataset = FaceTextDataset(cfg.DATAPATH, cfg.TREE.BASE_SIZE, cfg.MAXLEN, split="train")
    testDataset = FaceTextDataset(cfg.DATAPATH, cfg.TREE.BASE_SIZE, cfg.MAXLEN, split="test")
    trnDataloader = DataLoader(trnDataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    testDataloader = DataLoader(testDataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4)

    ## models
    ImageEncoder = FACE_ENCODER(cfg.MODEL.E_DIM)
    TextEncoder = RNN_ENCODER(trnDataset.vocab_size)
    para = list(TextEncoder.parameters())
    for v in ImageEncoder.parameters():
        if v.requires_grad:
            para.append(v)
    ## optimizer 
    optimizer = torch.optim.Adam(para, lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS)
    # optimizer = torch.optim.SGD(para, lr=cfg.TRAIN.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.DECAY_PERIOD, gamma=cfg.TRAIN.DECAY_RATE)
    trainEncoders(TextEncoder, ImageEncoder, optimizer, scheduler,
                  trnDataloader, testDataloader, trnDataset.idx2word)


if __name__ == "__main__":
    main()
