# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/03/08

import logging
from functools import reduce

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from skimage import measure


class SLM(object):
    def __init__(self):
        # logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

        # parameters
        self.rsu_beta = 0.707
        self.rsu_eps = 1e-7

        self.ras_expand_factor = 1.5
        self.ras_filter_times = 5
        self.ras_scala_beta = 3

        self.rda_eta = 0.5

        self.rmi_wsu = 0.4
        self.rmi_was = 0.35
        self.rmi_wda = 0.25

        # visual settings
        self.visual_ras = False
        self.src_addmap_path = None

        # sum indicator
        self.all_metrics = self._format_output_dict()

    def _format_output_dict(self, *params):
        """
        format output dict
        :param params: keys
        :return: format dict
        """
        len_params = len(params)
        if len_params == 0: init_param = [[] for i in range(4)]
        elif len_params == 4: init_param = params
        else: raise NotImplementedError

        return {
            "↑ Rsu [0 ~ 1]": init_param[0],
            "↑ Rda [0 ~ 1]": init_param[1],
            "↓ Ras [0 ~ 1]": init_param[2],
            "↑ Rmi [0 ~ 1]": init_param[3]
        }

    def logging_acc(self, metrics_dict, prob_path = None, ave = False):
        """
        logging the metrics
        :param metrics_dict: dict of metrics
        :param prob_path: path
        :return: 0
        """

        if not ave:
            self.logger.info("Eval {}".format(prob_path))
        else:
            self.logger.info("+++++++++++++++Average++++++++++++++")

        self.logger.info("+++++++ Calc the SLM METRICS +++++++")
        for metric, value in metrics_dict.items():
            self.logger.info("++++     {}:{:.4f}   ++++".format(metric, value))
        self.logger.info("++++++++++++++++++++++++++++++++++++\n")

    def set_visual_options(self, visual_ras, src_addmap_path):
        """
        set visual options
        :param visual_ras: flag
        :param src_addmap_path: set src addmap path
        """
        self.visual_ras = visual_ras
        self.src_addmap_path = src_addmap_path
        return True

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

    def _get_region_center_radius(self, region_point):
        """
        get the region center and radius
        :param region_point: regions
        :return: mid_x, mid_y, radius
        """
        mid_x = int(reduce(lambda x, y: x+y, np.array(region_point)[:, 0]) / len(region_point))
        mid_y = int(reduce(lambda x, y: x+y, np.array(region_point)[:, 1]) / len(region_point))
        radius = int(np.mean([np.linalg.norm(np.array(point) - np.array([mid_x, mid_y])) for point in region_point]) * self.ras_expand_factor)
        return mid_x, mid_y, radius

    def _get_prob_center_in_gray(self, prob):
        """
        get the top point with the highest probability from the probability map
        :param prob: probability
        :return: centers
        """

        # recover the prob
        gray_img = np.asarray(prob * 255.0, dtype=np.uint8)

        # construct continuous area
        continuous_area = np.asarray(gray_img > 150, np.uint8) * 255
        continuous_area = np.uint8(measure.label(continuous_area, connectivity=2))

        # soften
        for i in range(self.ras_filter_times):
            gray_img = cv2.boxFilter(gray_img, ddepth=-1, ksize=(50, 50))

        # get probability binary map
        mx = maximum_filter(gray_img, size=1000)
        gray_img = np.where(mx == gray_img, gray_img, 0)
        gray_img = np.asarray(gray_img > 0, np.uint8) * 255

        # get probability area information
        labels = measure.label(gray_img, connectivity=2)
        all_region_infos = measure.regionprops(labels)
        centers = [[int(i) for i in prop.centroid][::-1] for prop in all_region_infos]

        # construct v-center list and sort
        v_center = [[c[0], c[1], prob[c[1]][c[0]]] for c in centers]
        v_center.sort(key= lambda x: x[2], reverse=True)
        centers = list(map(lambda x: x[:2], v_center))

        # filter centers
        centers = [i for i in centers if prob[i[1]][i[0]] >= 0.5]

        return centers, continuous_area

    def _get_offset_between_real_and_synthetic(self, real_center_radius, prob_centers, bina_img):
        """
        calculate true center offset from result center
        :param real_center_radius: real_center_radius
        :param prob_centers: prob_centers
        :return: offsets
        """

        # check prob_centers is not None
        if len(prob_centers) == 0 : return [real_center_radius[0][2]]

        offsets = []
        for center_radius in real_center_radius:
            x, y, r = center_radius

            # calc the l2 dis
            dises = list(map(lambda p: np.linalg.norm(np.array([x, y] - np.array(p))), prob_centers))

            # filter the dis in cicle
            dises = list(filter(lambda d: d <= r, dises))

            # if no prob center set it to radius
            offsets.append(np.mean(dises) if len(dises) != 0 else r)

        return offsets

    def _trans_ras_offset_to_scalable_ras(self, offsets, centers_and_radius):
        """
        convert distance offset to ras value
        :param offsets: offsets
        :return: centers_and_radius
        """

        # granular transformation
        granular_offet = np.mean([off/v[2] for off, v in zip(offsets, centers_and_radius)])

        # scala transformation
        granular_offet = (np.exp(self.ras_scala_beta * granular_offet) - 1) / (np.exp(self.ras_scala_beta) - 1)

        return granular_offet

    def ras(self, region_lists, prob, visual=True, src_img=None):
        """
        calc the matric of ras: makes attention center close to annotation center
        :param region_lists: regions
        :param prob: probability
        :return: ras
        """

        # get the annotation center and radius
        centers_and_radius = [self._get_region_center_radius(i) for i in region_lists]

        # get the point with the highest probability from the probability map
        prob_centers, bina_img = self._get_prob_center_in_gray(prob)

        # calculate true center offset from result center
        offsets = self._get_offset_between_real_and_synthetic(centers_and_radius, prob_centers, bina_img)

        # convert distance offset to rcs value
        ras = self._trans_ras_offset_to_scalable_ras(offsets, centers_and_radius)

        # visual
        if visual and (src_img != None):
            src_img = cv2.imread(src_img)

            # logging something
            print("centers_and_radius: ", centers_and_radius)
            print("prob_centers: ", prob_centers)
            print("offsets: ", offsets)

            # backup area
            for c_r in centers_and_radius:
                cv2.circle(src_img, (c_r[0], c_r[1]), c_r[2], 2, 3)

            # candidate points
            for idx, point in enumerate(prob_centers):
                cv2.circle(src_img, tuple(point), 6*(idx+1), 1, 4)
                cv2.putText(src_img, str(idx+1), tuple(point), cv2.FONT_HERSHEY_COMPLEX, 6, (0, 0, 0), 25)

            cv2.imwrite("./img_circle.jpg", src_img)

            print(prob_centers)

        return ras

    def rsu(self, prob, mask):
        """
        calc the salient area proportion
        :param prob: probability
        :param mask: mask
        :return: rsu
        """

        all_mask_value = np.sum(np.multiply(prob, mask))
        all_value = np.sum(prob)
        H, W = np.shape(mask)
        all_mask = np.sum(mask)

        left_frac = all_mask_value / (all_value - all_mask_value + self.rsu_eps)

        right_frac = (H * W - all_mask) / all_mask

        rsu = -np.exp(-1 * self.rsu_beta * left_frac * right_frac) + 1

        return rsu

    def rda(self, region_lists, prob):
        """
        calc the matric of rda: makes attention center focus on one point
        :param region_lists: regions
        :param prob: probability
        :return: rda
        """

        # get the annotation center and radius
        centers_and_radius = [self._get_region_center_radius(i) for i in region_lists]

        # get the point with the highest probability from the probability map
        prob_centers, bina_img = self._get_prob_center_in_gray(prob)

        # set value
        rda = []
        for c_r in centers_and_radius:
            x, y, r = c_r

            # calc the backup points
            backup_points = list(filter(lambda p: np.linalg.norm(np.array([x, y] - np.array(p))) <= r, prob_centers))

            # margin condition
            len_backup_points = len(backup_points)
            if len_backup_points <= 1 :
                rda.append(float(len_backup_points))
                continue

            # if len_backup_points >= 2, calc the attention discrete
            centers_attention = np.average(backup_points, axis=0)
            dises = list(map(lambda p: np.linalg.norm(np.array(centers_attention - np.array(p))), backup_points))
            meas_dis = np.mean(dises) / r

            rda_single = 0.5 * (1 - meas_dis) + np.exp(- self.rda_eta * (len_backup_points + 2))

            rda.append(rda_single)

        return np.mean(rda)


    def rmi(self, rsu, rda, ras):
        """
        calculate the mean indicator
        :param rsu: rsu
        :param rda: rda
        :param ras: ras
        :return: rmi
        """
        return self.rmi_wsu * rsu + self.rmi_was * (1 - ras) + self.rmi_wda * rda

    def evaluate(self, prob_path, region_list):
        """
        evaluate the slm task
        :param prob_path: probability map path
        :param region_list: region points
        :return: slm metrics
        """
        # read prob
        prob = self.read_gray_to_prob(prob_path)

        # generate mask
        mask = self.generate_mask_by_points(prob, region_list)

        # rsu
        rsu = self.rsu(prob, mask)

        # ras
        ras = self.ras(region_list, prob, visual=self.visual_ras, src_img=self.src_addmap_path)

        # rda
        rda = self.rda(region_list, prob)

        # mi
        rmi = self.rmi(rsu, rda, ras)

        # sort metrics
        metrics = self._format_output_dict(rsu, rda, ras, rmi)
        self.logging_acc(metrics, prob_path)

        return metrics

    def append_metric(self, metric):
        """
        append metric to calc ave indicator
        :param metric: sort metrics
        """
        for k in metric.keys():
            self.all_metrics[k].append(metric[k])

    def get_the_mean_metric(self):
        """
        get the mean metric
        """
        mean_metric = {}
        for k in self.all_metrics:
            mean_metric[k] = np.mean(self.all_metrics[k])

        self.logging_acc(mean_metric, ave=True)
        return mean_metric


if __name__ == "__main__":

    points = [
            [
                [
                    1163.84851074219,
                    948.244812011719
                ],
                [
                    1360.29187011719,
                    883.699096679688
                ],
                [
                    1497.80224609375,
                    993.146118164063
                ],
                [
                    1649.34436035156,
                    1299.03637695313
                ],
                [
                    1660.56970214844,
                    2067.9716796875
                ],
                [
                    1691.43933105469,
                    2132.51733398438
                ],
                [
                    1635.31262207031,
                    2292.478515625
                ],
                [
                    1090.88391113281,
                    2289.67211914063
                ],
                [
                    1147.01062011719,
                    1481.44812011719
                ],
                [
                    1278.908203125,
                    1374.80749511719
                ],
                [
                    1166.65490722656,
                    1209.23376464844
                ],
                [
                    1169.46118164063,
                    937.019470214844
                ],
                [
                    1169.46118164063,
                    937.019470214844
                ],
                [
                    1169.46118164063,
                    937.019470214844
                ]
            ]
        ]

    prob_path = "../predict/cache/probmap_9.jpg"
    add_path = "../predict/cache/addmap_9.jpg"

    # eval
    slm = SLM()
    # slm.set_visual_options(visual_ras=True, src_addmap_path=add_path)
    metrics = slm.evaluate(prob_path, region_list=points)
