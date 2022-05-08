# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/05/05

import nltk
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

# Encoder for LW_MCR
class EncoderLWMCR:
    def image_encoder(self, model, image_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        # data preprocessing
        image = Image.open(image_path).convert('RGB')
        image = transform(image)  # torch.Size([3, 256, 256])
        image = torch.unsqueeze(image, dim=0).cuda()

        visual_feature, lower_feature, higher_feature = model.extract_feature(image)
        global_feature = l2norm(visual_feature, dim=-1)

        # to cpu vector
        vector = global_feature.cpu().detach().numpy()[0]

        return vector

    def text_encoder(self, model, vocab, text):

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(text.lower())
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]

        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)
        caption = torch.unsqueeze(caption, dim=0).cuda()

        # model processing
        text_feature = model.text_feature(caption)
        text_feature = l2norm(text_feature, dim=-1)

        # to cpu vector
        vector = text_feature.cpu().detach().numpy()[0]

        return vector

# Encoder for AMFMN
class EncoderAMFMN:
    def image_encoder(self, model, image_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        # check image size
        # image_path = trans_bigimage_to_small(image_path)

        # data preprocessing
        image = Image.open(image_path).convert('RGB')
        image = transform(image)  # torch.Size([3, 256, 256])
        image = torch.unsqueeze(image, dim=0).cuda()

        # model processing
        lower_feature, higher_feature, solo_feature = model.extract_feature(image)
        global_feature = model.mvsa(lower_feature, higher_feature, solo_feature)
        global_feature = l2norm(global_feature, dim=-1)

        # to cpu vector
        vector = global_feature.cpu().detach().numpy()[0]

        return vector

    def text_encoder(self, model, vocab, text):

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(text.lower())
        punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        tokens = [k for k in tokens if k not in punctuations]
        tokens_UNK = [k if k in vocab.word2idx.keys() else '<unk>' for k in tokens]

        caption = []
        caption.extend([vocab(token) for token in tokens_UNK])
        caption = torch.LongTensor(caption)
        caption = torch.unsqueeze(caption, dim=0).cuda()
        caption = caption.expand((2, caption.shape[-1]))

        # model processing
        text_feature = model.text_feature(caption)
        text_feature = l2norm(text_feature, dim=-1)

        # to cpu vector
        vector = text_feature.cpu().detach().numpy()[0]

        return vector

class Encoder:
    def __init__(self, model):
        if model.model_name == 'AMFMN':
            self.encoder = EncoderAMFMN()
        elif model.model_name == 'LW_MCR':
            self.encoder = EncoderLWMCR()
        else:
            raise NotImplementedError

    def cosine_sim(self, image_vector, text_vector):
        """
        计算两个向量间的余弦相似度
        :param image_vector: 图片编码向量
        :param text_vector: 文本编码向量
        :return: 相似度
        """
        if hasattr(self.encoder, 'calc_similarity'):
            return self.encoder.calc_similarity(image_vector, text_vector)
        else:
            image_vector = image_vector / np.linalg.norm(image_vector)
            text_vector = text_vector / np.linalg.norm(text_vector)

            similarity = np.mean(np.multiply(image_vector, text_vector))
            return similarity

    def image_encoder(self, model, image_path):
        """
        提供的图像编码函数
        :param model: 模型文件
        :param image_path: 图像路径
        :return: 编码向量
        """

        return self.encoder.image_encoder(model, image_path)

    def text_encoder(self, model, vocab, text):
        """
        提供的文本编码函数
        :param model: 模型文件
        :param vocab: 文本字典
        :param text: 编码文本
        :return: 编码向量
        """

        return self.encoder.text_encoder(model, vocab, text)


if __name__ == "__main__":
    from model_init import model_init

    prefix = "./"
    yaml_path = "option/RSITMD/RSITMD_AMFMN.yaml"
    test_jpg = "./test_data/sparseresidential_3814.tif"
    test_caption = "many airplane parked in the airport"

    # init model
    model, vocab = model_init(
        prefix_path = "./",
        yaml_path = yaml_path
    )

    # encoder
    encoder = Encoder(model)
    visual_vector = encoder.image_encoder(model, test_jpg)
    text_vector = encoder.text_encoder(model, vocab, test_caption)

    print("visual_vector:", np.shape(visual_vector))
    print("text_vector:", np.shape(text_vector))

    if len(visual_vector) == len(text_vector):
        print("Encoder test successful!")

    sims = encoder.cosine_sim(visual_vector, text_vector)
    print("Calc sim successful!")






