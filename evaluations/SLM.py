# **
# * Copyright @2022 AI, ZHIHU Inc. (zhihu.com)
#
# @author yuanzhiqiang <yuanzhiqiang@zhihu.com>
#         2022/3/28
import math

import cv2
import numpy as np
import time
import logging
from functools import reduce
from scipy.ndimage import maximum_filter
import skimage

class SLM(object):
    def __init__(self):
        # logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

        # parameters
        self.rca_expand_factor = 0.8
        self.rca_filter_times = 10


    def read_gray_to_prob(self, probmap_path):
        """
        Read the prob maps, and trans to probility
        :param probmap_path: probmap routh
        :return: probability
        """
        gray_image = cv2.imread(probmap_path, cv2.IMREAD_GRAYSCALE)
        prob = gray_image / 255.0
        return prob

    def generate_mask_by_points(self, prob, points_list):
        """
        Generate mask by regions
        :param prob: probability
        :param points_list: regions
        :return: mask
        """
        H, W = prob.shape

        mask = np.zeros((H, W))

        points_list = [np.array(i, np.int32) for i in points_list]

        # fill
        cv2.fillPoly(mask, points_list, 1)

        return mask

    def logging_acc(self, metrics_dict, prob_path):
        """
        logging the metrics
        :param metrics_dict: dict of metrics
        :param prob_path: path
        :return: 0
        """
        self.logger.info("")
        self.logger.info("Eval {}".format(prob_path))
        self.logger.info("++++++++++++++++++++++++++++++++++++")
        self.logger.info("+++++++ Calc the SLM METRICS +++++++")
        for metric, value in metrics_dict.items():
            self.logger.info("++++     {}:{:.4f}   ++++".format(metric, value))
        self.logger.info("++++++++++++++++++++++++++++++++++++")

    def calc_sum_prob(self, prob):
        """
        calc sum of the prob
        :param prob: probability
        :return: sum of all probability
        """
        return np.sum(prob)

    def get_peak_centers_in_prob(self, prob):
        """
        find peak centers in probability
        :param prob: probability
        :return: points
        """
        pass

    def rsa(self, prob, mask):
        """
        calc the matric of rsa: makes salient area fill the annotation mask
        :param prob: probability
        :param mask: mask
        :return: rca
        """
        prob_region = np.multiply(prob, mask)
        rca = np.sum(prob_region) / np.sum(mask)
        return rca

    def _get_region_center_radius(self, region_point):
        """
        get the region center and radius
        :param region_point: regions
        :return: mid_x, mid_y, radius
        """
        mid_x = int(reduce(lambda x, y: x+y, np.array(region_point)[:, 0]) / len(region_point))
        mid_y = int(reduce(lambda x, y: x+y, np.array(region_point)[:, 1]) / len(region_point))
        radius = int(np.mean([np.linalg.norm(np.array(point) - np.array([mid_x, mid_y])) for point in region_point]) * self.rca_expand_factor)
        return mid_x, mid_y, radius

    def _get_prob_center_in_gray(self, prob):
        """
        get the top point with the highest probability from the probability map
        :param prob: probability
        :return: centers
        """

        # recover the prob
        gray_img = np.asarray(prob * 255.0, dtype=np.uint8)

        # soften
        for i in range(self.rca_filter_times):
            gray_img = cv2.medianBlur(gray_img, 251)

        # get probability binary map
        mx = maximum_filter(gray_img, size=1000)
        gray_img = np.where(mx == gray_img, gray_img, 0)
        gray_img = np.asarray(gray_img > 0, np.uint8) * 255

        # get probability area information
        labels = skimage.measure.label(gray_img, connectivity=2)
        all_region_infos = skimage.measure.regionprops(labels)
        centers = [[int(i) for i in prop.centroid][::-1] for prop in all_region_infos]

        # construch v-center list and sort
        v_center = [[c[0], c[1], prob[c[1]][c[0]]] for c in centers]
        v_center.sort(key= lambda x: x[2], reverse=True)
        centers = list(map(lambda x: x[:2], v_center))

        return centers

    def _get_offset_between_real_and_synthetic(self, real_center_radius, prob_centers):
        """
        calculate true center offset from result center
        :param real_center_radius: real_center_radius
        :param prob_centers: prob_centers
        :return: offsets
        """
        offsets = []
        for center_radius in real_center_radius:
            x, y, r = center_radius
            # calc the l2 dis
            dises = list(map(lambda p: np.linalg.norm(np.array([x, y] - np.array(p))), prob_centers))

            # filter the dis in cicle
            dises = [dis for dis in dises if dis <= r]

            # if no prob center set it to radius
            offsets.append(np.mean(dises) if len(dises) != 0 else r)

        return offsets

    def _trans_rca_offset_to_scalable_rca(self, offsets, centers_and_radius):
        """
        convert distance offset to rcs value
        :param offsets: offsets
        :return: centers_and_radius
        """

        # granular transformation
        granular_offet = np.mean([off/v[2] for off, v in zip(offsets, centers_and_radius)])

        return granular_offet

    def rca(self, region_lists, prob):
        """
        calc the matric of rcs: makes attention center close to annotation center
        :param region_lists: regions
        :param prob: probability
        :return: rca
        """

        # get the annotation center and radius
        centers_and_radius = [self._get_region_center_radius(i) for i in region_lists]

        # get the point with the highest probability from the probability map
        prob_centers = self._get_prob_center_in_gray(prob)

        # calculate true center offset from result center
        offsets = self._get_offset_between_real_and_synthetic(centers_and_radius, prob_centers)

        # convert distance offset to rcs value
        rca = self._trans_rca_offset_to_scalable_rca(offsets, centers_and_radius)

        return rca

    def rus(self, prob, mask):
        """
        calc the matric of rus: makes useless area / salient area less
        :param prob: probability
        :param mask: mask
        :return: rus
        """
        prob_region = np.multiply(prob, mask)
        rus = (np.sum(prob) - np.sum(prob_region)) / np.sum(prob_region)
        return rus

    def ram(self, rsa, rus, rca):
        """
        calculate the mean indicator
        :param rsa: rsa
        :param rus: rus
        :param rca: rca
        :return: mi
        """
        return (rsa + (1 - math.exp(- rus))+ (1 -rca)) / 3

    def evaluate(self, prob_path, region_list):
        """
        evaluate the slm task
        :param prob_path: probability map path
        :param region_list: region points
        :return: slm metrics
        """
        # read prob
        prob = slm.read_gray_to_prob(prob_path)

        # generate mask
        mask = slm.generate_mask_by_points(prob, region_list)

        # rsa
        rsa = self.rsa(prob, mask)

        # rus
        rus = self.rus(prob, mask)

        # rca
        rca = self.rca(region_list, prob)

        # mi
        mi = self.ram(rsa, rus, rca)

        # sort metrics
        metrics = {
            "↑ Rsa [0 ~ 1]": rsa,
            "↓ Rus [0 ~ -]": rus,
            "↓ Rca [0 ~ 1]": rca,
            "↓ mi  [0 ~ 1]": mi,
        }
        self.logging_acc(metrics, prob_path)

        return metrics

if __name__ == "__main__":

    points = [
        [[1882, 203], [1743, 1729], [2701, 1817], [2840, 291]],
    ]

    prob_path = "../predict/cache/probmap_1.jpg"

    # eval
    slm = SLM()
    metrics = slm.evaluate(prob_path, region_list=points)
    metrics = slm.evaluate("../predict/cache/probmap_2.jpg", region_list=points)