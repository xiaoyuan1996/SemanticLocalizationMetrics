# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/04/03

import json
import os

import matplotlib.image as imgplt
import matplotlib.pyplot as plt


def analyze_samples(json_path):
    # load json
    with open(json_path,'r',encoding='utf8')as fp:
            json_data = json.load(fp)

    # analyze
    print("=========================")
    print("Lens of items: {}\n".format(len(json_data)))
    print("===== Map Relation ======")
    for idx, item in enumerate(json_data):
        print("Idx:{}, Filename:{}, Caption:{}".format(idx, json_data[idx]['jpg_name'], json_data[idx]['caption'].replace("\n", "")))

    return json_data

def visual_data(png_path, json_data, show_idx):

    # load show data
    show_json_data = json_data[show_idx]
    finalname = os.path.join(png_path, show_json_data['jpg_name'])
    plotpoints = show_json_data['points']
    captiondata = show_json_data['caption']

    print("\n=========================")
    print("===== Visual Data=========")
    print("Idx: {}".format(show_idx))
    print("Filename: {}".format(finalname))
    print("Caption: {}".format(captiondata.replace("\n", "")))
    print("Annotations: {}".format(plotpoints))

    # visual data
    pic = imgplt.imread(finalname)
    plt.imshow(pic)
    plt.title(show_json_data['jpg_name'])

    xdata = []
    ydata = []
    for k in range(len(plotpoints)):
        xdata.clear()
        ydata.clear()
        for j in range(len(plotpoints[k])):
            item = plotpoints[k][j]
            x = item[0]
            y = item[1]
            plt.scatter(x, y, s=25, c='r')
            xdata.append(x)
            ydata.append(y)
        plt.plot(xdata, ydata, c='b')
        plt.plot([xdata[0], xdata[j]], [ydata[0], ydata[j]], c='b')
    plt.show()


if __name__ == "__main__":
    png_path = "./imgs/"
    json_path = "./annotations/anno.json"
    show_idx = 44

    # analyze samples
    json_data = analyze_samples(json_path)

    # plot one sample
    visual_data(png_path, json_data, show_idx)

