# -----------------------------------------------------------
# "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information"
# Yuan, Zhiqiang and Zhang, Wenkai and Changyuan Tian and Xuee, Rong and Zhengyuan Zhang and Wang, Hongqi and Fu, Kun and Sun, Xian
# Writen by YuanZhiqiang, 2021.  Our code is depended on AMFMN
# ------------------------------------------------------------
import copy

import torch.nn.init

from .AMFMN_utils import *


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im, dim=-1)
    s = l2norm(s, dim=-1)
    w12 = im.mm(s.t())
    return w12

class BaseModel(nn.Module):
    def __init__(self, opt={}, vocab_words=[]):
        super(BaseModel, self).__init__()

        # img feature
        self.extract_feature = ExtractFeature(opt = opt)
        self.drop_g_v = nn.Dropout(0.3)

        # vsa feature
        self.mvsa =VSA_Module(opt = opt)

        # text feature
        self.text_feature = Skipthoughts_Embedding_Module(
            vocab= vocab_words,
            opt = opt
        )

        # weight
        self.gw = opt['global_local_weight']['global']
        self.lw = opt['global_local_weight']['local']

        self.Eiters = 0

        self.model_name = 'AMFMN'

    def forward(self, img, input_local_rep, input_local_adj, text, text_lens=None):

        # extract features
        lower_feature, higher_feature, solo_feature = self.extract_feature(img)

        # mvsa featrues
        global_feature = self.mvsa(lower_feature, higher_feature, solo_feature)

        # text features
        text_feature = self.text_feature(text)

        sims = cosine_sim(global_feature, text_feature)
        return sims

def factory(opt, vocab_words, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    model = BaseModel(opt, vocab_words)

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model



