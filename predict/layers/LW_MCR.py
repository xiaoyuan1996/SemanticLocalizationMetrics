# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/05/05

import copy

from .MCR_utils import *


class unsupervised_Visual_Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(unsupervised_Visual_Model, self).__init__()

        self.extract_feature = ExtractFeature()

        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        feature, _, _ = self.extract_feature(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class unsupervised_Text_Model(nn.Module):
    def __init__(self, feature_dim=128, vocab_words=[]):
        super(unsupervised_Text_Model, self).__init__()


        self.text_feature = textCNN(
            vocab= vocab_words,
            opt = None
        )
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        feature = self.text_feature(x)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature()

#        self.pre_train_extroctor = unsupervised_Visual_Model()
#        state = torch.load('./model/unsupervised_pretrain_model/visual_pre.pth')
#        self.pre_train_extroctor.load_state_dict(state)
#        del state
#        self.extract_feature = self.pre_train_extroctor.extract_feature
#        del self.pre_train_extroctor



        self.text_feature = textCNN(
            vocab= vocab_words,
            opt = opt
        )

 #       self.pre_train_extroctor = unsupervised_Text_Model(vocab_words= vocab_words,)
 #       state = torch.load('./model/unsupervised_pretrain_model/text_pre.pth')
 #       self.pre_train_extroctor.load_state_dict(state)
 #       del state
 #       self.text_feature = self.pre_train_extroctor.text_feature
 #       del self.pre_train_extroctor

        self.Eiters = 0

        self.model_name = 'LW_MCR'


    def forward(self, img, text, text_lens=None):

        # extract features
        visual_feature, lower_feature, higher_feature = self.extract_feature(img)

        # text features
        text_feature = self.text_feature(text)

        # print("visual_feature.shape:{}".format(visual_feature.shape))
        # print("text_feature.shape:{}".format(text_feature.shape))
        # visual_feature.shape: torch.Size([8, 512])
        # text_feature.shape: torch.Size([8, 512])
        # print("=========================")
        # exit(0)

        sims = cosine_sim(visual_feature, text_feature)

        return sims, [visual_feature, text_feature, lower_feature, higher_feature]

def factory(opt, vocab_words, cuda=True, data_parallel=True,device_ids=[0]):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model, device_ids=device_ids).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model


