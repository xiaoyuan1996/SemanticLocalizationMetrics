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

# 将图片按照512*512剪裁，并返回剪裁时间和剪切次数
# img_path：图片位置，比如0.jpg
# subimages_dir：剪裁的图片放的位置
# cut_count：剪裁的次数
def split_image(img_path, subimages_dir):
    t1 = time.time()
    # Read Image
    source_img = cv2.imread(img_path)
    img_height = np.shape(source_img)[0]
    img_weight = np.shape(source_img)[1]
    step = 512
    # 剪裁图像
    # Cut img
    for h in range(0, img_height, step):
        h_start, h_end = h, h + step
        if h_end >= img_height:
            h_start, h_end = img_height - step, img_height
        for w in range(0, img_weight, step):
            w_start, w_end = w, w + step
            # bound?
            if w_end >= img_weight:
                w_start, w_end = img_weight - step, img_weight
            cut_img_name = str(h_start) + "_" + str(h_end) + "_" + str(w_start) + "_" + str(w_end) + ".jpg"
            cut_img = source_img[h_start:h_end, w_start:w_end]
            cut_img = cv2.resize(cut_img, (256, 256), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(subimages_dir, cut_img_name), cut_img)
    split_time = time.time() - t1

    return split_time

# 获取裁剪图像的特征向量
# subimages_dir：剪裁好的图片所在位置，比如：0_subimages
# img_vectors：每一张剪裁好图片的特征向量的列表集合
# n_clusters：k-means的聚类个数
# text：文本
def get_img_vectors(subimages_dir, text, boximages_dir, encoder, img_path, sim_results1):

    t2 = time.time()

    # read subimages
    subimages = os.listdir(subimages_dir)

    # 获取文本的特征向量
    text_vector = encoder.text_encoder(model, vocab, text)

    random_times = 4

    for i in range(random_times):
        tmp = text.split()
        np.random.shuffle(tmp)
        text_bak = " ".join([i for i in tmp])
        text_vector += encoder.text_encoder(model, vocab, text_bak)
    text_vector /= (random_times + 1)

    # 获取每一个剪裁好图片的特征,并计算与文本的相似度
    for subimage in subimages:
        # 一维的特征向量
        image_vector = encoder.image_encoder(model, os.path.join(subimages_dir, subimage))
        sim = encoder.cosine_sim(image_vector,text_vector)
        sim_results1.append(sim)


    # 将相似度排序，返回索引列表
    sim_indexs = np.argsort(sim_results1)
    source_img = cv2.imread(img_path)
    img_height = np.shape(source_img)[0]
    img_weight = np.shape(source_img)[1]


    select1 = int(len(sim_indexs)*0.15)+1
    select_index1 = sim_indexs[-select1:]

    for i in select_index1:
        img_name = subimages[i]
        h_start, h_end, w_start, w_end = img_name.replace(".jpg", "").split("_")
        cut_img_size = 512
        p1 = [int(h_start), int(w_start)]
        p2 = [int(h_start), int(w_end)]
        p3 = [int(h_end), int(w_start)]
        p4 = [int(h_end), int(w_end)]
        p5 = [int(h_start) + cut_img_size * 1 / 4, int(w_start) + cut_img_size * 1 / 4]
        p6 = [int(h_start) + cut_img_size * 1 / 4, int(w_start) + cut_img_size * 3 / 4]
        p7 = [int(h_start) + cut_img_size * 3 / 4, int(w_start) + cut_img_size * 1 / 4]
        p8 = [int(h_start) + cut_img_size * 3 / 4, int(w_start) + cut_img_size * 3 / 4]
        p9 = [int(h_start) + cut_img_size * 1 / 2, int(w_start) + cut_img_size * 1 / 2]
        points = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
        for point in points:
            h = point[0]
            w = point[1]
            sizes = [64,128]
            for size in sizes:
                h_start1 = int(h - size if h - size > 0 else 0)
                h_end1 = int(h_start1 + size * 2 if h_start1 + size * 2 < img_height else img_height)
                w_start1 = int(w - size if w - size > 0 else 0)
                w_end1 = int(w_start1 + size * 2 if w_start1 + size * 2 < img_weight else img_weight)
                box_img_name = str(h_start1) + "_" + str(h_end1) + "_" + str(w_start1) + "_" + str(w_end1) + ".jpg"
                box_img = source_img[h_start1:h_end1, w_start1:w_end1]
                box_img = cv2.resize(box_img, (256, 256), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(os.path.join(boximages_dir, box_img_name), box_img)
    stage2_time = time.time()-t2


    return text_vector,select_index1,stage2_time

def generate_heatmap(img_path, output_file_h, output_file_a, output_file_p, text_vector, boximages_dir, subimages_dir,sim_results1,select_index1,
                     encoder):

    t3 = time.time()

    # read subimages
    boximages = os.listdir(boximages_dir)
    subimages = os.listdir(subimages_dir)

    sim_results = []
    for boximage in boximages:
        image_vector = encoder.image_encoder(model, os.path.join(boximages_dir, boximage))
        sim_results.append(encoder.cosine_sim(text_vector, image_vector))


    # read Image
    source_img = cv2.imread(img_path)
    img_row = np.shape(source_img)[0]
    img_col = np.shape(source_img)[1]

    # mkdir map
    heat_map = np.zeros([img_row, img_col], dtype=float)
    heat_num = np.zeros([img_row, img_col], dtype=float)

    for idx in select_index1:
        h_start, h_end, w_start, w_end = subimages[idx].replace(".jpg", "").split("_")
        heat_map[int(h_start):int(h_end), int(w_start):int(w_end)] += sim_results1[idx]
        heat_num[int(h_start):int(h_end), int(w_start):int(w_end)] += 1


    for idx,file in enumerate(boximages):
        h_start, h_end, w_start, w_end = file.replace(".jpg","").split("_")
        heat_map[int(h_start):int(h_end), int(w_start):int(w_end)] += sim_results[idx]
        heat_num[int(h_start):int(h_end), int(w_start):int(w_end)] += 1

    for i in range(np.shape(heat_map)[0]):
        for j in range(np.shape(heat_map)[1]):
            if(heat_num[i,j]!=0.0):
                heat_map[i,j] = heat_map[i,j] / heat_num[i,j]

    # filter
    adaptive = np.asarray(heat_map)
    adaptive = adaptive-np.min(adaptive)
    probmap = adaptive/np.max(adaptive)
    # must convert to type unit8
    probmap = np.uint8(255 * probmap)
    # 中值滤波
    # probmap = cv2.medianBlur(probmap,251)
    # 高斯滤波
    probmap = cv2.GaussianBlur(probmap, (255, 255), 0, 0)
    heatmap = cv2.applyColorMap(probmap, cv2.COLORMAP_JET)
    img_add = cv2.addWeighted(source_img, 0.7, heatmap, 0.3, 0)
    t3 = time.time()-t3

    cv2.imwrite( output_file_p ,probmap)
    cv2.imwrite( output_file_h ,heatmap)
    cv2.imwrite( output_file_a ,img_add)

    # clear temp
    utils.delete_dire(boximages_dir)
    os.rmdir(boximages_dir)

    utils.delete_dire(subimages_dir)
    os.rmdir(subimages_dir)

    return t3


if __name__ == "__main__":

    import argparse

    # settings
    parser = argparse.ArgumentParser(description="SLM")
    parser.add_argument("--yaml_path", type=str, default="option/RSITMD/RSITMD_AMFMN.yaml", help="config yaml path")
    parser.add_argument("--cache_path", type=str, default="cache/RSITMD_AMFMN", help="cache path")
    parser.add_argument("--src_data_path", type=str, default="../test_data/imgs", help="testset images path")
    parser.add_argument("--src_anno_path", type=str, default="../test_data/annotations/anno.json",
                        help="testset annotations path")
    opt = parser.parse_args()

    # mkdir
    if not os.path.exists(opt.cache_path):
        os.mkdir(opt.cache_path)

    # logging
    logger = utils.get_logger(os.path.join(opt.cache_path, 'log.txt'))

    # init model
    model, vocab = model_init(
        prefix_path="./",
        yaml_path=opt.yaml_path
    )

    # init encoder
    encoder = Encoder(model)

    # start eval
    slm_metric = SLM()

    # load from annotations
    with open(opt.src_anno_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)

    total_time = time.time()
    second_time_list = []

    for idx, item in enumerate(json_data):
        # load sample
        img = item['jpg_name']
        text = item['caption']
        points = item['points']
        count = 0

        # path
        img_path = os.path.join(opt.src_data_path, img)
        subimage_files_dir = os.path.join(opt.cache_path, os.path.basename(img_path).split(".")[0])
        heatmap_path = os.path.join(opt.cache_path, "heatmap_{}.jpg".format(idx))
        probmap_path = os.path.join(opt.cache_path, "probmap_{}.jpg".format(idx))
        addmap_path = os.path.join(opt.cache_path, "addmap_{}.jpg".format(idx))

        # 裁切图像文件夹
        subimages_dir = subimage_files_dir + '_subimages'
        # anchor_box的文件夹
        boximages_dir = subimage_files_dir + '_boximages'
        if os.path.exists(subimages_dir):
            utils.delete_dire(subimages_dir)
        else:
            os.makedirs(subimages_dir)

        if (os.path.exists(boximages_dir)):
            utils.delete_dire(boximages_dir)
        else:
            os.makedirs(boximages_dir)

        times = time.time()

        # processing
        split_image(img_path, subimages_dir)

        sim_results1 = []
        select_index1 = []
        stage2_time = 0
        text_vector, select_index1, stage2_time = get_img_vectors(subimages_dir, text, boximages_dir, encoder, img_path,
                                                                  sim_results1)

        generate_heatmap(img_path, heatmap_path, addmap_path, probmap_path, text_vector,
                         boximages_dir, subimages_dir, sim_results1, select_index1, encoder)

        times = time.time() - times
        second_time_list.append(times)
        logger.info("Processing {}/{}: {}，的总时间为：{}".format(idx, len(json_data), img, times))
        logger.info("Corresponding text: {}".format(text))

        # evaluate #
        metrics = slm_metric.evaluate(probmap_path, region_list=points)
        slm_metric.append_metric(metrics)

    slm_metric.get_the_mean_metric()

    all_time = time.time() - total_time
    logger.info("Time-Total: {:.4f}".format(all_time))
    logger.info("second_time_list={}".format(second_time_list))
