from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import numbers, math
import numpy as np


class VGG16_base(nn.Module):
    def __init__(self, pts_num):
        super(VGG16_base, self).__init__()

        self.downsample = 8
        self.pts_num = pts_num

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True))

        self.CPM_feature = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),  # CPM_1
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))  # CPM_2

        stage1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, kernel_size=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(512, pts_num, kernel_size=1, padding=0))
        stages = [stage1]
        for i in range(1, 3):
            stagex = nn.Sequential(
                nn.Conv2d(128 + pts_num, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=7, dilation=1, padding=3), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=1, padding=0), nn.ReLU(inplace=True),
                nn.Conv2d(128, pts_num, kernel_size=1, padding=0))
            stages.append(stagex)
        self.stages = nn.ModuleList(stages)

    # def specify_parameter(self, base_lr, base_weight_decay):
    #     params_dict = [
    #         {'params': get_parameters(self.features, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
    #         {'params': get_parameters(self.features, bias=True), 'lr': base_lr * 2, 'weight_decay': 0},
    #         {'params': get_parameters(self.CPM_feature, bias=False), 'lr': base_lr, 'weight_decay': base_weight_decay},
    #         {'params': get_parameters(self.CPM_feature, bias=True), 'lr': base_lr * 2, 'weight_decay': 0},
    #         ]
    #     for stage in self.stages:
    #         params_dict.append(
    #             {'params': get_parameters(stage, bias=False), 'lr': base_lr * 4, 'weight_decay': base_weight_decay})
    #         params_dict.append({'params': get_parameters(stage, bias=True), 'lr': base_lr * 8, 'weight_decay': 0})
    #     return params_dict

    # return : cpm-stages, locations
    def forward(self, inputs):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.size(0), inputs.size(1)
        batch_cpms, batch_locs, batch_scos = [], [], []

        feature = self.features(inputs)
        xfeature = self.CPM_feature(feature)
        for i in range(3):
            if i == 0:
                cpm = self.stages[i](xfeature)
            else:
                cpm = self.stages[i](torch.cat([xfeature, batch_cpms[i - 1]], 1))
            batch_cpms.append(cpm)

        # The location of the current batch
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(batch_cpms[-1][ibatch], 4, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

        return batch_cpms, batch_locs, batch_scos


# use vgg16 conv1_1 to conv4_4 as feature extracation
model_urls = 'https://download.pytorch.org/models/vgg16-397923af.pth'


def cpm_vgg16(pts):
    model = VGG16_base(pts)

    print('vgg16_base use pre-trained model')
    weights = model_zoo.load_url(model_urls)
    model.load_state_dict(weights, strict=False)
    return model


def find_tensor_peak_batch(heatmap, radius, downsample, threshold=0.000001):
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    assert radius > 0 and isinstance(radius, numbers.Number), 'The radius is not ok : {}'.format(radius)
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1. + 2. * x.data / (L - 1)

    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    # affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    # theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:, 0, 0] = (boxes[2] - boxes[0]) / 2
    affine_parameter[:, 0, 2] = (boxes[2] + boxes[0]) / 2
    affine_parameter[:, 1, 1] = (boxes[3] - boxes[1]) / 2
    affine_parameter[:, 1, 2] = (boxes[3] + boxes[1]) / 2
    # extract the sub-region heatmap
    theta = affine_parameter.to(heatmap.device)
    grid_size = torch.Size([num_pts, 1, radius * 2 + 1, radius * 2 + 1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = torch.arange(-radius, radius + 1).to(heatmap).view(1, 1, radius * 2 + 1)
    Y = torch.arange(-radius, radius + 1).to(heatmap).view(1, radius * 2 + 1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts, -1), 1)
    x = torch.sum((sub_feature * X).view(num_pts, -1), 1) / sum_region + index_w
    y = torch.sum((sub_feature * Y).view(num_pts, -1), 1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y], 1), score
