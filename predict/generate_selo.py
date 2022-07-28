# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/05/03

import json
import os
import sys
import time

import cv2
import numpy as np
import utils
from model_encoder import Encoder
from model_init import model_init

sys.path.append("..")
from evaluations.SLM import SLM

def split_image(img_path, steps, cache_path):
    t1 = time.time()

    subimage_files_dir = os.path.join(cache_path, os.path.basename(img_path).split(".")[0])

    # 裁切图像文件夹
    subimages_dir = subimage_files_dir +'_subimages'
    if os.path.exists(subimages_dir):
        utils.delete_dire(subimages_dir)
    else:
        os.makedirs(subimages_dir)

    # Read Image
    source_img = cv2.imread(img_path)
    img_weight = np.shape(source_img)[0]
    img_height = np.shape(source_img)[1]
    logger.info("img size:{}x{}".format(img_weight, img_height))

    for step in steps:
        logger.info("Start split images with step {}".format(step))
        for gap in [step, 0.5 * step]:
            gap = int(gap)

            # Cut img
            for h in range(0 + (step - gap), img_height, step):
                h_start, h_end = h, h + step
                # bound?
                if h_end >= img_height:
                    h_start, h_end = img_height - step, img_height

                for w in range(0 + (step - gap), img_weight, step):
                    w_start, w_end = w, w + step
                    # bound?
                    if w_end >= img_weight:
                        w_start, w_end = img_weight - step, img_weight

                    cut_img_name = str(w_start) + "_" + str(w_end) + "_" + str(h_start) + "_" + str(h_end) + ".jpg"
                    cut_img = source_img[w_start:w_end, h_start:h_end]
                    cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)

                    cv2.imwrite(os.path.join(subimages_dir, cut_img_name), cut_img)


    logger.info("Image {} has been split successfully.".format(img_path))

    return time.time() - t1

def generate_heatmap(img_path, text, output_file_h, output_file_a, output_file_p, cache_path):

    subimages_dir = os.path.join(cache_path, os.path.basename(img_path).split(".")[0]) +'_subimages'

    logger.info("Start calculate similarities ...")
    cal_start = time.time()

    # init encoder
    encoder = Encoder(model)

    # text vector
    text_vector = encoder.text_encoder(model, vocab, text)

    # read subimages
    subimages = os.listdir(subimages_dir)
    sim_results = []
    for subimage in subimages:
        image_vector = encoder.image_encoder(model, os.path.join(subimages_dir, subimage))
        sim_results.append(encoder.cosine_sim(text_vector, image_vector))
    cal_end = time.time()
    logger.info("Calculate similarities in {}s".format(cal_end-cal_start))
    t2 = cal_end-cal_start

    logger.info("Start generate heatmap ...")
    generate_start = time.time()

    # read Image
    source_img = cv2.imread(img_path)
    img_row = np.shape(source_img)[0]
    img_col = np.shape(source_img)[1]

    # mkdir map
    heat_map = np.zeros([img_row, img_col], dtype=float)
    heat_num = np.zeros([img_row, img_col], dtype=float)
    for idx,file in enumerate(subimages):
        r_start, r_end, c_start, c_end = file.replace(".jpg","").split("_")

        heat_map[int(r_start):int(r_end), int(c_start):int(c_end)] += sim_results[idx]
        heat_num[int(r_start):int(r_end), int(c_start):int(c_end)] += 1


    for i in range(np.shape(heat_map)[0]):
        for j in range(np.shape(heat_map)[1]):
            heat_map[i,j] = heat_map[i,j] / heat_num[i,j]
    t3 = time.time() - generate_start

    logger.info("Generate finished, start optim ...")
    optim_start = time.time()
    # filter
    adaptive = np.asarray(heat_map)
    adaptive = adaptive-np.min(adaptive)
    probmap = adaptive/np.max(adaptive)
    # must convert to type unit8
    probmap = np.uint8(255 * probmap)
    probmap = cv2.medianBlur(probmap,251)
    heatmap = cv2.applyColorMap(probmap, cv2.COLORMAP_JET)
    img_add = cv2.addWeighted(source_img, 0.7, heatmap, 0.3, 0)
    generate_end = time.time()
    logger.info("Generate heatmap in {}s".format(generate_end-generate_start))

    logger.info("Saving heatmap in {} ...".format(output_file_h))
    logger.info("Saving heatmap in {} ...".format(output_file_a))
    logger.info("Saving heatmap in {} ...".format(output_file_p))
    cv2.imwrite( output_file_p ,probmap)
    cv2.imwrite( output_file_h ,heatmap)
    cv2.imwrite( output_file_a ,img_add)
    logger.info("Saved ok.")

    # clear temp
    utils.delete_dire(subimages_dir)
    os.rmdir(subimages_dir)

    t4 = generate_end - optim_start
    return t2, t3, t4

if __name__ == "__main__":

    import argparse

    # settings
    parser = argparse.ArgumentParser(description="SLM")
    parser.add_argument("--yaml_path", type=str, default="option/RSITMD/RSITMD_AMFMN.yaml", help="config yaml path")
    parser.add_argument("--cache_path", type=str, default="cache/RSITMD_AMFMN", help="cache path")
    parser.add_argument("--src_data_path", type=str, default="../test_data/imgs", help="testset images path")
    parser.add_argument("--src_anno_path", type=str, default="../test_data/annotations/anno.json", help="testset annotations path")
    parser.add_argument("--step", type=str, default="256_512_768", help="step")
    opt = parser.parse_args()

    # mkdir
    if not os.path.exists(opt.cache_path):
        os.mkdir(opt.cache_path)

    # params
    steps = [int(step) for step in opt.step.split("_")]

    # logging
    logger = utils.get_logger(os.path.join(opt.cache_path, 'log.txt'))

    # init model
    model, vocab = model_init(
        prefix_path = "./",
        yaml_path = opt.yaml_path
    )

    # start eval
    slm_metric = SLM()

    # load from annotations
    with open(opt.src_anno_path,'r',encoding='utf8')as fp:
            json_data = json.load(fp)

    t1_all, t2_all, t3_all, t4_all = 0, 0, 0, 0
    total_time = time.time()

    for idx, item in enumerate(json_data):
        # load sample
        img = item['jpg_name']
        text = item['caption']
        points = item['points']

        # path
        img_path = os.path.join(opt.src_data_path, img)
        heatmap_path = os.path.join(opt.cache_path, "heatmap_{}.jpg".format(idx))
        probmap_path = os.path.join(opt.cache_path, "probmap_{}.jpg".format(idx))
        addmap_path = os.path.join(opt.cache_path, "addmap_{}.jpg".format(idx))

        # logging
        logger.info("Processing {}/{}: {}".format(idx, len(json_data), img))
        logger.info("Corresponding text: {}".format(text))

        # processing
        t1 = split_image(img_path, steps, opt.cache_path)
        t2, t3, t4 = generate_heatmap(img_path, text, heatmap_path, addmap_path, probmap_path, opt.cache_path)

        t1_all += t1
        t2_all += t2
        t3_all += t3
        t4_all += t4

        # evaluate #
        metrics = slm_metric.evaluate(probmap_path, region_list=points)
        slm_metric.append_metric(metrics)

    slm_metric.get_the_mean_metric()

    all_time = time.time() - total_time
    logger.info("Time-Cut: {:.4f}s".format(t1_all))
    logger.info("Time-Sim: {:.4f}".format(t2_all))
    logger.info("Time-Gnt: {:.4f}".format(t3_all))
    logger.info("Time-Flt: {:.4f}".format(t4_all))
    logger.info("Time-Total: {:.4f}".format(all_time))

