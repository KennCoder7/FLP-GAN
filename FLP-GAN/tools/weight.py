import torch.nn as nn
from copy import deepcopy


def weights_init(m):
    # orthogonal_
    # xavier_uniform_(
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if list(m.state_dict().keys())[0] == 'weight':
            nn.init.orthogonal_(m.weight.data, 1.0)
        elif list(m.state_dict().keys())[3] == 'weight_bar':
            nn.init.orthogonal_(m.weight_bar.data, 1.0)
        # nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten
