# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/05/05

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# ----------------------------------------------------------------------------
# -------------------------   Visual Feature Encoding       ------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 4, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 4, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class redefine_squeezenet(nn.Module):
    def __init__(self, ):
        super(redefine_squeezenet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
        )
        self.conv_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256),
        )
        # Final convolution is initialized differently from the rest
        final_conv = nn.Conv2d(512, 1000, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, img):
        low_feature = self.conv_1(img)
        high_feature = self.conv_2(low_feature)
        results = self.classifier(high_feature)
        return results

def convert_squeezenet(squeeze, net):#vgg16是pytorch自带的
    vgg_items = net.state_dict().items()
    vgg16_items = squeeze.items()

    pretrain_model = {}
    j = 0
    for k, v in net.state_dict().items():#按顺序依次填入
        v = list(vgg16_items)[j][1]
        k = list(vgg_items)[j][0]
        pretrain_model[k] = v
        j += 1
    return pretrain_model

class extract_by_squeezenet(nn.Module):
    def __init__(self, ):
        super(extract_by_squeezenet, self).__init__()

        net = redefine_squeezenet()

        self.conv_1 = net.conv_1
        self.conv_2 = net.conv_2

    def forward(self, img):
        low_feature = self.conv_1(img)
        high_feature = self.conv_2(low_feature)
        return low_feature, high_feature

class Pretrain_visual_extractor(nn.Module):
    def __init__(self, parms=None):
        super(Pretrain_visual_extractor, self).__init__()
        # backbone = resnet18(pretrained=True)
        # self.backbone = myResnet(backbone)
        self.backbone = extract_by_squeezenet()

        # lower and higher transformation
        self.lower_trans = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.higher_trans = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=512, out_channels=64, stride=1, kernel_size=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        # BCNN

        # image self attention
        self.ISA_1_c = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1)
        self.ISA_1_b = nn.BatchNorm2d(4)
        self.ISA_1_p = nn.PReLU()
        self.ISA_1_ca = ChannelAttention(4)
        self.ISA_1_sa = SpatialAttention()
        self.ISA_1_pool = nn.MaxPool2d(kernel_size=2)

        self.ISA_2_c = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
        self.ISA_2_b = nn.BatchNorm2d(8)
        self.ISA_2_p = nn.PReLU()
        self.ISA_2_ca = ChannelAttention(8)
        self.ISA_2_sa = SpatialAttention()

        self.fc = torch.nn.Linear(64*8, 10)

    def forward(self, img):
        N = img.size()[0]

        # backbone
        lower_feature, higher_feature = self.backbone(img)

        # lower and higher transformation
        lower_feature = self.lower_trans(lower_feature)
        higher_feature = self.higher_trans(higher_feature)

        # BCNN
        lower_feature = lower_feature.view(N, 64, 32*32)
        higher_feature = higher_feature.view(N, 64, 32*32)
        X = torch.bmm(lower_feature, torch.transpose(higher_feature, 1, 2)) / (32 ** 2)

        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = torch.unsqueeze(X, dim=1)

        # image self attention
        X = self.ISA_1_c(X)
        X = self.ISA_1_b(X)
        X = self.ISA_1_p(X)
        X = self.ISA_1_ca(X) * X
        X = self.ISA_1_sa(X) * X
        X = self.ISA_1_pool(X)

        X = self.ISA_2_c(X)
        X = self.ISA_2_b(X)
        X = self.ISA_2_p(X)
        X = self.ISA_2_ca(X) * X
        X = self.ISA_2_sa(X) * X

        X = X.view(N, -1)
        X = self.fc(X)

        return X

class ExtractFeature(nn.Module):
    def __init__(self):
        super(ExtractFeature, self).__init__()
        self.pre_train_extroctor = Pretrain_visual_extractor()

        self.backbone = self.pre_train_extroctor.backbone

        # lower and higher transformation
        self.lower_trans = self.pre_train_extroctor.lower_trans
        self.higher_trans = self.pre_train_extroctor.higher_trans

        # BCNN

        # image self attention
        self.ISA_1_c = self.pre_train_extroctor.ISA_1_c
        self.ISA_1_b = self.pre_train_extroctor.ISA_1_b
        self.ISA_1_p = self.pre_train_extroctor.ISA_1_p
        self.ISA_1_ca = self.pre_train_extroctor.ISA_1_ca
        self.ISA_1_sa = self.pre_train_extroctor.ISA_1_sa
        self.ISA_1_pool = self.pre_train_extroctor.ISA_1_pool
        self.dropout_1 = nn.Dropout(0.2)

        self.ISA_2_c = self.pre_train_extroctor.ISA_2_c
        self.ISA_2_b = self.pre_train_extroctor.ISA_2_b
        self.ISA_2_p = self.pre_train_extroctor.ISA_2_p
        self.ISA_2_ca = self.pre_train_extroctor.ISA_2_ca
        self.ISA_2_sa = self.pre_train_extroctor.ISA_2_sa
        self.dropout_2 = nn.Dropout(0.2)


        del self.pre_train_extroctor

        self.fc = nn.Linear(512, 512)

    def forward(self, img):

        N = img.size()[0]

        # backbone
        lower_feature, higher_feature = self.backbone(img)

        # lower and higher transformation
        lf = self.lower_trans(lower_feature)
        hf = self.higher_trans(higher_feature)
        # print("lower_feature.shape:{}".format(lower_feature.shape))
        # print("higher_feature.shape:{}".format(higher_feature.shape))
        # lower_feature.shape: torch.Size([8, 64, 32, 32])
        # higher_feature.shape: torch.Size([8, 64, 32, 32])
        # print("=========================")

        # BCNN
        lower_feature = lf.view(N, 64, 32*32)
        higher_feature = hf.view(N, 64, 32*32)
        X = torch.bmm(lower_feature, torch.transpose(higher_feature, 1, 2)) / (32 ** 2)

        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = torch.unsqueeze(X, dim=1)

        # print("fusion.shape:{}".format(X.shape))
        # fusion.shape: torch.Size([8, 1, 64, 64])
        # print("=========================")

        # image self attention
        X = self.ISA_1_c(X)
        X = self.ISA_1_b(X)
        X = self.ISA_1_p(X)
        X = self.ISA_1_ca(X) * X
        X = self.ISA_1_sa(X) * X
        X = self.ISA_1_pool(X)
        X = self.dropout_1(X)

        X = self.ISA_2_c(X)
        X = self.ISA_2_b(X)
        X = self.ISA_2_p(X)
        X = self.ISA_2_ca(X) * X
        X = self.ISA_2_sa(X) * X
        X = self.dropout_2(X)

        X = X.view(N, -1)

        X = self.fc(X)

        return l2norm(X, -1), lf, hf

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count
#
# model = ExtractFeature()
# from torch.autograd import Variable
# input = Variable(torch.zeros(10, 3, 256, 256))
# feature = model(input)
# print(feature.shape)
# print(params_count(model))
#
# exit()

# ----------------------------------------------------------------------------
# -------------------------   Text Feature Encoding         ------------------

class textCNN(nn.Module):
    def __init__(self, vocab, opt, lstm_dropout=0.25, out_dropout=-1):
        super(textCNN, self).__init__()
        Vocab = len(vocab)+1  ## 已知词的数量
        Dim = 300  ##每个词向量长度
        Ci = 1  ##输入的channel数
        Knum = 100 ## 每种卷积核的数量
        Ks = [3,4,5]  ## 卷积核list，形如[2,3,4]
        Cla = 512

        self.embed = nn.Embedding(Vocab, Dim)  ## 词向量，这里直接随机
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层

        self.dropout = nn.Dropout(lstm_dropout)
        self.fc = nn.Linear(len(Ks)*Knum,Cla) ##全连接层

    def forward(self, x):
        x = self.embed(x)  # (8, 18, 300)

        x = x.unsqueeze(1)  # (8, 1, 18, 300)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)

        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)

        x = torch.cat(x, 1)  # (N,Knum*len(Ks))

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc(x)  # (N, C)

        return l2norm(logit, -1)

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    # w1 = l2norm(im, dim=-1)
    # w2 = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12
