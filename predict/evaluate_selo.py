# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/05/05

import json
import logging
import os
import sys

sys.path.append("..")
from evaluations.SLM import SLM

if __name__ == "__main__":
    import argparse

    # settings
    parser = argparse.ArgumentParser(description="SLM")
    parser.add_argument("--yaml_path", type=str, default="option/RSITMD/RSITMD_AMFMN.yaml", help="config yaml path")
    parser.add_argument("--cache_path", type=str, default="cache/RSITMD_AMFMN", help="cache path")
    parser.add_argument("--src_data_path", type=str, default="../test_data/imgs", help="testset images path")
    parser.add_argument("--src_anno_path", type=str, default="../test_data/annotations/anno.json", help="testset annotations path")
    opt = parser.parse_args()

    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # start eval
    slm_metric = SLM()

    # load from annotations
    with open(opt.src_anno_path,'r',encoding='utf8')as fp:
            json_data = json.load(fp)

    for idx, item in enumerate(json_data):
        # load sample
        img = item['jpg_name']
        text = item['caption']
        probmap_path = os.path.join(opt.cache_path, "probmap_{}.jpg".format(idx))
        points = item['points']

        # logging
        logger.info("Processing {}/{}: {}".format(idx, len(json_data), img))
        logger.info("Corresponding text: {}".format(text))

        # evaluate #
        metrics = slm_metric.evaluate(probmap_path, region_list=points)
        slm_metric.append_metric(metrics)

    slm_metric.get_the_mean_metric()