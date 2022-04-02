
#encoding:utf-8
# -----------------------------------------------------------
# "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval"
# Yuan, Zhiqiang and Zhang, Wenkai and Fu, Kun and Li, Xuan and Deng, Chubo and Wang, Hongqi and Sun, Xian
# IEEE Transactions on Geoscience and Remote Sensing 2021
# Writen by YuanZhiqiang, 2021.  Our code is depended on MTFN
# ------------------------------------------------------------

import os
import argparse
import yaml
import torch
from vocabs import deserialize_vocab

def parser_options(prefix_path, yaml_path):
    # load model options
    with open(os.path.join(prefix_path,yaml_path), 'r') as handle:
        options = yaml.safe_load(handle)

    return options

def model_init(prefix_path, yaml_path):
    options = parser_options(prefix_path, yaml_path)

    # choose model
    if options['model']['name'] == "AMFMN":
        from layers import AMFMN as models
    else:
        raise NotImplementedError

    # make vocab
    vocab = deserialize_vocab(os.path.join(prefix_path,options['dataset']['vocab_path']))
    vocab_word = sorted(vocab.word2idx.items(), key=lambda x: x[1], reverse=False)
    vocab_word = [tup[0] for tup in vocab_word]

    model = models.factory(options['model'],
                           vocab_word,
                           cuda=True,
                           data_parallel=False)

    checkpoint = torch.load(options['optim']['resume'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()

    return model, vocab

if __name__ == "__main__":
    prefix = "./"
    yaml_path = "option/RSITMD_mca/RSITMD_AMFMN.yaml"

    model, vocab = model_init(
        prefix_path = "./",
        yaml_path = yaml_path
    )

    try:
        model.eval()
        print(vocab)
        print("Successfully load the model.")
    except Exception as e:
        print("Failed to load the model.")
