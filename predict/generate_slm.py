import os, time
import numpy as np
import cv2
import logging
from tqdm import tqdm

from model_init import model_init
from model_encoder import image_encoder, text_encoder, cosine_sim
import utils

def split_image(img_path, steps, cache_path):
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

def generate_heatmap(img_path, text, output_file_h, output_file_a, output_file_p, cache_path):
    subimages_dir = os.path.join(cache_path, os.path.basename(img_path).split(".")[0]) +'_subimages'

    logger.info("Start calculate similarities ...")
    cal_start = time.time()

    # text vector
    text_vector = text_encoder(model, vocab, text)

    # read subimages
    subimages = os.listdir(subimages_dir)
    sim_results = []
    for subimage in subimages:
        image_vector = image_encoder(model, os.path.join(subimages_dir, subimage))
        sim_results.append(cosine_sim(text_vector, image_vector))
    cal_end = time.time()
    logger.info("Calculate similarities in {}s".format(cal_end-cal_start))


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

    logger.info("Generate finished, start optim ...")
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

if __name__ == "__main__":

    import argparse

    # settings
    parser = argparse.ArgumentParser(description="SLM")
    parser.add_argument("--yaml_path", type=str, default="option/RSITMD_mca/RSITMD_AMFMN.yaml", help="config yaml path")
    parser.add_argument("--cache_path", type=str, default="cache", help="cache path")
    parser.add_argument("--src_data_path", type=str, default="../test_data/imgs", help="testset images path")
    parser.add_argument("--src_anno_path", type=str, default="../test_data/annotations", help="testset annotations path")
    opt = parser.parse_args()

    # params
    steps = [512, 768]

    # logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # init model
    model, vocab = model_init(
        prefix_path = "./",
        yaml_path = opt.yaml_path
    )

    # start eval
    text = "there is a green pond next to the gray road."

    for idx, img in enumerate(os.listdir(opt.src_data_path)):
        # path
        img_path = os.path.join(opt.src_data_path, img)
        heatmap_path = os.path.join(opt.cache_path, "heatmap_{}".format(img.replace(".tif", ".jpg")))
        probmap_path = os.path.join(opt.cache_path, "probmap_{}".format(img.replace(".tif", ".jpg")))
        addmap_path = os.path.join(opt.cache_path, "addmap_{}".format(img.replace(".tif", ".jpg")))

        # logging
        logger.info("Processing {}/{}: {}".format(idx+1, len(os.listdir(opt.src_data_path)), img))

        # processing
        split_image(img_path, steps, opt.cache_path)
        generate_heatmap(img_path, text, heatmap_path, addmap_path, probmap_path, opt.cache_path)

        # evaluate
