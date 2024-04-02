import imp
from lib2to3.pgen2 import token
import os
import pickle
import numpy as np
import io
from PIL.Image import Image
from requests import request
import torch
from torch.autograd import Variable
import argparse
from PIL import Image
import flask
import base64
from gensim.models import KeyedVectors
import time

from models.encoders import FACE_ENCODER, RNN_ENCODER
from models.netG import G_NET
from tools.config import cfg, cfg_from_file
from models.LMGan import LMGen, LMDis
from tools.visualizations import bSplineAndSeg, editShapes
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

app = flask.Flask(__name__)
idx2word = None
word2idx = None
TextEncoder = None
LmG = None
netG = None
idx2Mask = None
cacheDict = dict()
t = time.time()
print("load GloVe")
# model = KeyedVectors.load_word2vec_format(r'e:\glove.twitter.27B.200d.bin.gz', binary=True)
print("loaded. %f", time.time() - t)
with open(r"E:\celebA\captions.txt", "r") as f:
    captions = f.readlines()

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/getcap', methods=["GET"])
def getcap():
    cnt = len(captions)
    # print(captions)
    idx = np.random.randint(0, cnt)
    return captions[idx]

@app.route("/sample", methods=["POST"])
def sample():
    # Ensure a caption was properly uploaded to our endpoint.
    data = {"words_unseen": ""}
    if flask.request.method == 'POST':
        ip = flask.request.remote_addr
        state = dict()
        print(flask.request.form['caption'])
        if flask.request.form['caption'] is not None:
            cap_ = flask.request.form['caption']
            state['cap'] = cap_
            cap, masks, cap_len, words_unseen = cap2tensor(cap_, idx2Mask, word2idx)
            if words_unseen is not None and len(words_unseen) > 0:
                words_unseen = "words_unseen: " + words_unseen
            data["words_unseen"] = words_unseen
            if cap is None or masks is None or cap_len is None:
                data["words_unseen"] = words_unseen + "\tempty caption"
                return flask.jsonify(data)
            cap = Variable(cap).unsqueeze(0)
            masks = Variable(masks).unsqueeze(0)
            noise.data.normal_(0, 1)
            hidden = TextEncoder.init_hidden(1)
            words_emb, sent_emb = TextEncoder(cap, cap_len, hidden)
            LMs, scores = LmG(sent_emb, words_emb, noise, masks)
            LMs = LMs.reshape(1, -1, 2)
            LMs = LMs * 128 + 128
            LMs = LMs[0].detach().cpu().numpy()
            state['LMs'] = LMs
            segMaps = bSplineAndSeg(LMs).astype(np.float32)
            segImg = segMaps.copy()
            segMaps = segMaps.transpose((2, 0, 1))
            segImg[:, :, 2] += segImg[:, :, 3]
            segImg[:, :, 1] += segImg[:, :, 3]
            segImg = Image.fromarray(np.uint8(segImg[:,:,:3]))
            # target.paste(segImg, (344, 0, 600, 256))
            # segImg.save("fake_segmap.png")
            segMaps = torch.from_numpy(segMaps / 255.0).unsqueeze(0)
            segMaps = Variable(segMaps)
            fake_imgs, _, c_code, mu, logvar = netG(noise, sent_emb, words_emb, masks, segMaps)
            fake_imgs[2][0].add_(1).div_(2).mul_(255)
            fake_img = fake_imgs[2][0].detach().cpu().data.numpy()
            fake_img = np.transpose(fake_img, (1, 2, 0))
            fake_img = Image.fromarray(np.uint8(fake_img))
            # target.paste(fake_img, (0, 0, 256, 256))
            # fake_img.save("fake_img.png")
            imgByteArr = io.BytesIO()
            fake_img.save(imgByteArr, format="png")
            img_data = imgByteArr.getvalue()
            base64_data = base64.b64encode(img_data).decode("utf-8")
            data['base64img'] = base64_data
            state['base64img'] = base64_data
            state['c_code'] = c_code
            state['masks'] = masks
            state['sent_emb'] = sent_emb
            state['words_emb'] = words_emb
            state['noise'] = noise
            segByteArr = io.BytesIO()
            segImg.save(segByteArr, format="png")
            seg_data = segByteArr.getvalue()
            base64_data = base64.b64encode(seg_data).decode("utf-8")
            data['base64seg'] = base64_data
            data['desc'] = cap_
            cacheDict[ip] = state
    return flask.jsonify(data)

@app.route("/shape", methods=["POST"])
def shape():
    # Ensure a caption was properly uploaded to our endpoint.
    data = {"words_unseen": ""}
    if flask.request.method == 'POST':
        ip = flask.request.remote_addr
        if ip in cacheDict:
            lastState = cacheDict[ip]
            cap_ = flask.request.form['caption']
            print(cap_)
            LMs = lastState['LMs'].copy()
            sent_emb = lastState['sent_emb']
            words_emb = lastState['words_emb']
            masks = lastState['masks']
            c_code = lastState['c_code']
            tLMs = editShapes(LMs, cap_, model)
            segMaps = bSplineAndSeg(tLMs).astype(np.float32)
            segImg = segMaps.copy()
            segMaps = segMaps.transpose((2, 0, 1))
            segImg[:, :, 2] += segImg[:, :, 3]
            segImg[:, :, 1] += segImg[:, :, 3]
            segImg = Image.fromarray(np.uint8(segImg[:,:,:3]))
            segImg.save("fake_segmap.png")
            segMaps = torch.from_numpy(segMaps / 255.0).unsqueeze(0)
            segMaps = Variable(segMaps).to(cfg.device)
            fake_imgs, _, _, mu, logvar = netG(noise, sent_emb, words_emb, masks, segMaps, c_code)
            fake_imgs[2][0].add_(1).div_(2).mul_(255)
            fake_img = fake_imgs[2][0].detach().cpu().data.numpy()
            fake_img = np.transpose(fake_img, (1, 2, 0))
            fake_img = Image.fromarray(np.uint8(fake_img))
            data['base64before'] = lastState['base64img']
            imgByteArr = io.BytesIO()
            fake_img.save(imgByteArr, format="png")
            img_data = imgByteArr.getvalue()
            base64_data = base64.b64encode(img_data).decode("utf-8")
            data['base64after'] = base64_data
            data['desc'] = cap_
        else:
            data['words_unseen'] = "no image to edit"
    return flask.jsonify(data)

@app.route("/modify", methods=["POST"])
def modify():
    data = {"words_unseen": ""}
    if flask.request.method == 'POST':
        ip = flask.request.remote_addr
        if ip in cacheDict:
            lastState = cacheDict[ip]
            cap_ = flask.request.form['caption']
            print(cap_)
            LMs = lastState['LMs'].copy()
            segMaps = bSplineAndSeg(LMs).astype(np.float32)
            segMaps = segMaps.transpose((2, 0, 1))
            segMaps = torch.from_numpy(segMaps / 255.0).unsqueeze(0)
            segMaps = Variable(segMaps).to(cfg.device)

            cap, masks, cap_len, words_unseen = cap2tensor(cap_, idx2Mask, word2idx)
            if words_unseen is not None and len(words_unseen) > 0:
                words_unseen = "words_unseen: " + words_unseen
            cap = Variable(cap).unsqueeze(0)
            masks = Variable(masks).unsqueeze(0)
            
            data["words_unseen"] = words_unseen
            if cap is None or masks is None or cap_len is None:
                data["words_unseen"] = words_unseen + "\tempty caption"
                return flask.jsonify(data)
            noise = lastState['noise']
            hidden = TextEncoder.init_hidden(1)
            words_emb, sent_emb = TextEncoder(cap, cap_len, hidden)
            fake_imgs, _, _, mu, logvar = netG(noise, sent_emb, words_emb, masks, segMaps)
            fake_imgs[2][0].add_(1).div_(2).mul_(255)
            fake_img = fake_imgs[2][0].detach().cpu().data.numpy()
            fake_img = np.transpose(fake_img, (1, 2, 0))
            fake_img = Image.fromarray(np.uint8(fake_img))
            data['base64before'] = lastState['base64img']
            imgByteArr = io.BytesIO()
            fake_img.save(imgByteArr, format="png")
            img_data = imgByteArr.getvalue()
            base64_data = base64.b64encode(img_data).decode("utf-8")
            data['base64after'] = base64_data
            # cap diff
            lastCap = lastState['cap']
            data['desc'] = makeCapDiff(lastCap, cap_)
        else:
            data['words_unseen'] = "no image to edit"
    return flask.jsonify(data)

def main():
    parser = argparse.ArgumentParser("FLG-GAN")
    parser.add_argument(
        "-gpu_id",
        type=int,
        default=0,
        dest="gpu_id"
    )
    parser.add_argument(
        "-cfg",
        default="E:\celebA\FLG-Gan\code\cfgs\miniServer.yml",
        dest="cfg"
    )
    args = parser.parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    global idx2word, word2idx, idx2Mask
    idx2word, word2idx = loadDict(os.path.join(cfg.DATAPATH, "captions.pickle"))
    vocab_size = len(idx2word)
    idx2Mask = getIdx2Masks(word2idx, idx2word)

    # load models
    global TextEncoder, LmG, netG
    TextEncoder = RNN_ENCODER(vocab_size)
    state_dict = torch.load(cfg.ENCODER_PATH, map_location=lambda storage, loc: storage)
    TextEncoder.load_state_dict(state_dict)
    TextEncoder.eval()

    LmG = LMGen()
    state_dict = torch.load(cfg.LMG_PATH, map_location=lambda storage, loc: storage)
    LmG.load_state_dict(state_dict)
    LmG.eval()

    netG = G_NET()
    state_dict = torch.load(cfg.NETG_PATH, map_location=lambda storage, loc: storage)
    netG.load_state_dict(state_dict)
    netG.eval()

    global noise
    noise = Variable(torch.FloatTensor(1, cfg.MODEL.Z_DIM))
    
    app.run(host="121.248.51.115", port=5000)

def makeCapDiff(lastCap, cap):
    lastCap = tokenize(lastCap)
    cap = tokenize(cap)
    diffWords = []
    i, j = 0, 0
    while i < len(lastCap) and j < len(cap):
        while i < len(lastCap) and word2idx.get(lastCap[i], -1) == -1:
            i += 1
        while j < len(cap) and word2idx.get(cap[j], -1) == -1:
            j += 1
        if i >= len(lastCap) or j >= len(cap):
            break
        if lastCap[i] != cap[j]:
            diffWords.append("<u>" + cap[j] + "</u>")
        else:
            diffWords.append(cap[j])
        i += 1
        j += 1
    diffHTML = ' '.join(diffWords)
    print(diffHTML)
    return diffHTML
 
def loadDict(filepath):
    with open(filepath, 'rb') as f:
        x = pickle.load(f)
        trn_captions, test_captions = x[0], x[1]
        idx2word, word2idx = x[2], x[3]
        del x
    return idx2word, word2idx

def cap2tensor(cap, idx2Mask, word2idx):
    words_unseen = ""
    cap = tokenize(cap)
    cap_new = []
    for word in cap:
        idx = word2idx.get(word, -1)
        if idx == -1:
            words_unseen += word + ' '
            print("word not in vocab: " + word)
            continue
        cap_new.append(idx)
    if len(cap_new) <= 0:
        print("empty caption")
        return None, None, None, words_unseen
    maxLen = cfg.MAXLEN - 1
    x = np.zeros(maxLen, dtype='int64')
    x_len = len(cap_new)
    num_words = x_len
    if num_words <= maxLen:
        x[:num_words] = cap_new
    else:
        cap_new = np.asarray(cap_new).astype('int64')
        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:maxLen]
        ix = np.sort(ix)
        x = cap_new[ix]
        x_len = maxLen
    masks = [idx2Mask[_x] for _x in x]
    masks = np.asarray(masks).astype('int64')
    cap = torch.from_numpy(x)
    masks = torch.from_numpy(masks)
    return cap, masks, torch.tensor([x_len]), words_unseen

def tokenize(cap):
    cap = cap.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cap.lower())
    if len(tokens) == 0:
        print(cap)
        print("no tokens")
    tokens_new = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0:
            tokens_new.append(t)
    return tokens_new

def getIdx2Masks(word2idx, idx2word):
    idx2Mask = [0] * len(word2idx)
    idx2Mask[0] = 1
    stop_words = set(stopwords.words('english'))
    for word in stop_words:
        if word2idx.get(word, None) is not None:
            idx2Mask[word2idx[word]] = 1
    return idx2Mask

if __name__ == "__main__":
    main()