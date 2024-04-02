import os
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
import dateutil.tz

from data.dataset import FaceTextDataset
from models.encoders import RNN_ENCODER
from models.LMGan import LMGen, LMDis
from tools.config import cfg, cfg_from_file
from engine.trainer import trainLMGAN
from tools.logger import setupLogger
from engine.trainer import trainLMGAN


def main():
    parser = argparse.ArgumentParser("Landmark GAN")
    parser.add_argument(
        "-gpu_id",
        type=int,
        default=0,
        dest="gpu_id"
    )
    parser.add_argument(
        "-cfg",
        default="./cfgs/lmGen.yml",
        dest="cfg"
    )

    args = parser.parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    # torch.cuda.set_device(args.gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s/%s' % \
                 (cfg.DATASET_NAME, "LMGen", timestamp)
    model_dir = os.path.join(output_dir, 'Model')
    image_dir = os.path.join(output_dir, 'Image')
    cfg.IMG_DIR = image_dir
    cfg.MODEL_DIR = model_dir
    os.makedirs(model_dir)
    os.makedirs(image_dir)
    logger = setupLogger("Landmark GAN", output_dir)
    logger.info(cfg)
    cfg.LOG = logger

    ## Datasets
    trnDataset = FaceTextDataset(cfg.DATAPATH, cfg.TREE.BASE_SIZE, cfg.MAXLEN, is4gen=True, split="train")
    testDataset = FaceTextDataset(cfg.DATAPATH, cfg.TREE.BASE_SIZE, cfg.MAXLEN, is4gen=True, split="test")
    trnDataloader = DataLoader(trnDataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=4)
    testDataloader = DataLoader(testDataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=4)

    ## models
    # TextEncoder = T_ENCODER(trnDataset.vocab_size, cfg.MODEL.E_DIM)
    TextEncoder = RNN_ENCODER(trnDataset.vocab_size)
    state_dict = torch.load(cfg.ENCODER_PATH, map_location=lambda storage, loc: storage)
    TextEncoder.load_state_dict(state_dict)
    for v in TextEncoder.parameters():
        v.requires_grad = False
    LmG = LMGen()
    LmD = LMDis()

    optimizerG = torch.optim.RMSprop(LmG.parameters(), lr=cfg.TRAIN.LR)
    optimizerD = torch.optim.RMSprop(LmD.parameters(), lr=cfg.TRAIN.LR)
    # optimizerG = torch.optim.Adam(LmG.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS)
    # optimizerD = torch.optim.Adam(LmD.parameters(), lr=cfg.TRAIN.LR, betas=cfg.TRAIN.BETAS)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=cfg.TRAIN.DECAY_PERIOD,
                                                 gamma=cfg.TRAIN.DECAY_RATE)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=cfg.TRAIN.DECAY_PERIOD,
                                                 gamma=cfg.TRAIN.DECAY_RATE)

    trainLMGAN(TextEncoder, LmG, LmD, optimizerG, optimizerD, schedulerG, schedulerD, \
               trnDataloader, testDataloader, trnDataset.idx2word)


if __name__ == "__main__":
    main()
