# Learning to Evaluate Performance of Multi-modal Semantic Localization 
(undergoing review)
##### Author: Zhiqiang Yuan, Chongyang Li, Zhuoying Pan, et. al 

<a href="https://github.com/xiaoyuan1996/retrievalSystem"><img src="https://travis-ci.org/Cadene/block.bootstrap.pytorch.svg?branch=master"/></a>
![Supported Python versions](https://img.shields.io/badge/python-3.7-blue.svg)
![Supported OS](https://img.shields.io/badge/Supported%20OS-Linux-yellow.svg)
![npm License](https://img.shields.io/npm/l/mithril.svg)
<a href="https://pypi.org/project/mitype/"><img src="https://img.shields.io/pypi/v/mitype.svg"></a>

### -------------------------------------------------------------------------------------
### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update

### -------------------------------------------------------------------------------------

#### Contexts

* [Introduction](#introduction)
* [Dataset And Metrics](#dataset-and-metrics)
  * [Testdata](#testdata)
  * [Metrics](#metrics)
* [Baselines](#baselines)
  * [Comparison of SeLo Performance on Different Trainsets](#comparison-of-selo-performance-on-different-trainsets)
  * [Comparison of SeLo Performance on Different Scales](#comparison-of-selo-performance-on-different-scales)
  * [Comparison of SeLo Performance on Different Retrieval Models](#comparison-of-selo-performance-on-different-retrieval-models)
  * [Analysis of Time Consumption](#analysis-of-time-consumption)
* [Implementation](#implementation)
  * [Environment](#environment)
  * [Run The Demo](#run-the-demo)
  * [Customize Model](#customize-model)
* [Epilogue](#epilogue)
* [Citation](#citation)
### -------------------------------------------------------------------------------------



## INTRODUCTION
An official evaluation metric for semantic localization.

<img src="https://github.com/xiaoyuan1996/SemanticLocalizationMetrics/blob/master/figure/compare.jpg" width="700"  alt="compare"/>

**Fig.1.** (a) Results of airplane detection. (b) Results of semantic localization with query of ``white planes parked in the open space of the white airport''. Compared with tasks such as detection, SeLo achieves semantic-level retrieval with only caption-level annotation during training, which can adapt to higher-level retrieval tasks.



<img src="https://github.com/xiaoyuan1996/SemanticLocalizationMetrics/blob/master/figure/demo.gif" width="700"  alt="shwon"/>

**Fig.2.** Visualization of SeLo with query of "the red rails where the grey train is located run through the residential area".


The semantic localization (SeLo) task refers to using cross-modal information such as text to quickly localize RS images at the semantic level [\[link\]](https://ieeexplore.ieee.org/document/9437331).
This task implements semantic-level detection, which only uses caption-level supervision information.
In our opinion, it is a meaningful and interesting work, which realizes the unification of sub-tasks such as detection and segmentation.

![visual image](./figure/SeLo.jpg)

**Fig.3.** Framework of Semantic Localization. After multi-scale segmentation of large RS images, we perform cross-modal similarity calculation on query and multiple slices. The calculated regional probabilities are then utilized to pixel-level averaging, which generates the SeLo map after further noise suppression.

We contribute test sets, evaluation metrics and baselines for semantic localization, and provide a detailed demo to use this evaluation framework.
Any questions can open a Github [issue](https://github.com/xiaoyuan1996/SemanticLocalizationMetrics/issues).
Start and enjoy!



### -------------------------------------------------------------------------------------

## DATASET AND METRICS

#### TESTDATA

We contribute a semantic localization testset to provide systematic evaluation for SeLo task. 
The images in SLT come from Google Earth, and Fig. 4 exhibits several samples from the testset. 
Every sample includes a large image in RS scene with the size of 3k × 2k to 10k × 10k, a query sentence, and one or more corresponding semantic bounding boxes.

<img src="https://github.com/xiaoyuan1996/SemanticLocalizationMetrics/blob/master/figure/sample.jpg" width="700"  alt="sample"/>


**Fig.4.** Four samples of Semantic Localization Testset. (a) Query: “ships without cargo floating on the black sea are docked in the port”. (b) Query: “a white
airplane ready to take off on a grayblack runway”. (c) Query: “some cars are parked in a parking lot surrounded by green woods”. (d) Query: “the green
football field is surrounded by a red track”.

**TABLE I** Quantitative Statistics of Semantic Localization Testset.

|   Parameter   | Value |      Parameter       | Value  |
| ------------- | ------| ---------------------| -------|
|  Word Number  | 160   |  Caption Ave Length  |  11.2  |
| Sample Number | 59    | Ave Resolution Ratio (m) | 0.3245 |
| Channel Number| 3     |  Ave Region Number   |  1.75  |
| Image Number  | 22    | Ave Attention Ratio  |  0.068 |




#### METRICS

We systematically model and study semantic localization in detail, and propose multiple discriminative evaluation metrics to quantify this task based on significant area proportion, attention shift distance, and discrete attention distance.

<img src="https://github.com/xiaoyuan1996/SemanticLocalizationMetrics/blob/master/figure/indicator.jpg" width="900"  alt="shwon"/>

**Fig.5.** Three proposed evaluation metrics for semantic localization. (a) Rsu aims to calculate the attention ratio of the ground-truth area to the useless
area. (b) Ras attempts to quantify the shift distance of the attention from the GT center. (c) Rda evaluates the discreteness of the generated attention from
probability divergence distance and candidate attention number.

**TABLE II** Explanation of the indicators.

| Indicator | Range | Meaning  |
| --------- | ------| ---------|
| Rsu  | ↑ [ 0 ~ 1 ] | Calc the salient area proportion |
| Ras  | ↓ [ 0 ~ 1 ] | Makes attention center close to annotation center |  
| Rda  | ↑ [ 0 ~ 1 ] | Makes attention center focus on one point |
| Rmi  | ↑ [ 0 ~ 1 ] | Calculate the mean indicator of SeLo task |



<img src="https://github.com/xiaoyuan1996/SemanticLocalizationMetrics/blob/master/figure/indicator_verify.jpg" width="700"  alt="shwon"/>

**Fig.6.** Qualitative analysis of SeLo indicator validity. (a) Query: “eight large white oil storage tanks built on grey concrete floor”. (b) Query: “a white plane
parked in a tawny clearing inside the airport”. (c) Query: “lots of white and black planes parked inside the grey and white airport”.

## BASELINES

All experiments all carried out at Intel(R) Xeon(R) Gold 6226R CPU @2.90GHz and a single NVIDIA RTX 3090 GPU.

#### Comparison of SeLo Performance on Different Trainsets

|   Trainset   | ↑ Rsu |      ↑ Rda       | ↓ Ras | ↑ Rmi  |
| ------------- | ------| -------------- | -------| -------|
|  Sydney  | 0.5844   |  0.5670  |  0.5026 | 0.5496  |
|  UCM | 0.5821    | 0.4715 | 0.5277 | 0.5160 |
| RSITMD| **0.6920**     |  **0.6667**   |  **0.3323**  | **0.6772** |
| RSICD  | 0.6661    | 0.5773  |  0.3875 | 0.6251


#### Comparison of SeLo Performance on Different Scales

|     | Scale-128 | Scale-256 | Scale-512 | Scale-768  | ↑ Rsu |      ↑ Rda       | ↓ Ras | ↑ Rmi  | Time (m)
| --- | ----------| ----------| ----------| -----------| ----- |      -----       | ----- | -----  | -----  |
|  s1 |    √      |  √        |           |            | 0.6389 |     0.6488       | 0.2878 | 0.6670  | 33.81 |
|  s2 |           |  √        |  √        |            | 0.6839 |     0.6030       | 0.3326 | 0.6579 | 14.25 |
|  s3 |           |           |  √        | √          | 0.6897 |     0.6371       | 0.3933 | 0.6475  | **11.23** |
|  s4 |    √      |  √        |  √        |            | 0.6682 |     **0.7072**       | **0.2694** | **0.6998**  | 34.60 |
|  s5 |           |  √        |  √        | √          | **0.6920** |     0.6667       | 0.3323 | 0.6772  | 16.92 |
|  s6 |    √      |  √        |  √        | √          | 0.6809 |     0.6884       | 0.3025 | 0.6886  | 36.28 |


#### Comparison of SeLo Performance on Different Retrieval Models

|   Trainset   | ↑ Rsu |      ↑ Rda       | ↓ Ras | ↑ Rmi  | Time (m) |
| ------------- | ------| -------------- | -------| -------| -------- |
|  VSE++  | 0.6364   |  0.5829  |  0.4166 | 0.6045  | 15.61
|  LW-MCR | 0.6698    | 0.6021 | 0.4335 | 0.6167 | **15.47**
| SCAN| 0.6421     |  0.6132   |  0.3871  | 0.6247 | 16.32
| CAMP  | 0.6819    | 0.6314  |  0.3912 | 0.6437 | 18.24
| AMFMN  | **0.6920**    | **0.6667**  |  **0.3323** | **0.6772** | 16.92

#### Analysis of Time Consumption

|   Scale (128, 256)   | Cut |      Sim       | Gnt | Flt  | Total |
| ------------- | ------| -------------- | -------| -------| ------|
| Times(m) | 2.85| 20.60 | 7.40| 0.73| 33.81|
| Rate(%) | 8.42| 60.94 | 21.88| 2.16| -|
					
|   Scale (512, 768)   | Cut |      Sim       | Gnt | Flt  | Total |
| ------------- | ------| -------------- | -------| -------| ------|
| Times(m) | 0.46| 1.17 | 6.96| 0.67| 11.23|
| Rate(%) | 4.06| 10.42 | 61.98| 5.97| -|

|   Scale (256, 512, 768)   | Cut |      Sim       | Gnt | Flt  | Total |
| ------------- | ------| -------------- | -------| -------| ------|
| Times(m) | 0.93| 5.72 | 7.38| 0.74| 16.92|
| Rate(%) | 5.52| 33.82 | 43.60| 4.37| -|								

## IMPLEMENTATION

#### ENVIRONMENT

1.Pull our project and install the requirements, make sure the code path only include English: 
    
```
   $ apt-get install python3
   $ git clone git@github.com:xiaoyuan1996/SemanticLocalizationMetrics.git
   $ cd SemanticLocalizationMetrics
   $ pip install -r requirements.txt
```

2.Prepare checkpoints and test iamges:


* Download pretrain checkpoints **SLM_checkpoints.zip** to **./predict/checkpoints/** from [BaiduYun (passwd: NIST)](https://pan.baidu.com/s/1DyRbY7s3jx-ZCWbcC_XHlw) or [GoogleDriver](https://drive.google.com/drive/folders/1LISJHiLVxPCiry1i7xJtOvuQ77nbEZD1?usp=sharing), make sure:
   
  + ./predict/checkpoints/
    + xxx.pth

* Download test images **SLM_testimgs.zip** to **./test_data/imgs/** from [BaiduYun (passwd: NIST)](https://pan.baidu.com/s/1DyRbY7s3jx-ZCWbcC_XHlw) or [GoogleDriver](https://drive.google.com/drive/folders/1LISJHiLVxPCiry1i7xJtOvuQ77nbEZD1?usp=sharing), make sure:

  + ./test_data/imgs/
    + xxx.tif

3.Download SkipThought Files to **/data** from [seq2vec (Password:NIST)](https://pan.baidu.com/s/1jz61ZYs8NZflhU_Mm4PbaQ) (or other path, but you should change the **option['model']['seq2vec']['dir_st']**)

4.Check the environments

```
    $ cd predict
    $ python model_encoder.py
    
    visual_vector: (512,)
    text_vector: (512,)
    Encoder test successful!
    Calc sim successful!
```


#### RUN THE DEMO
Run the follow command, and generated SeLo maps will be saved in **cache/**. 
```
   $ cd predict
   $ nohup python generate_selo.py --cache_path cache/RSITMD_AMFMN
   $ tail -f cache/RSITMD_AMFMN/log.txt

    2022-05-05 22:01:58,180 - __main__ - INFO - Processing 31/59: 20.jpg
    2022-05-05 22:01:58,180 - __main__ - INFO - Corresponding text: lots of white and black planes parked inside the grey and white airport.
    
    2022-05-05 22:01:59,518 - __main__ - INFO - img size:10000x10001
    2022-05-05 22:01:59,518 - __main__ - INFO - Start split images with step 256
    2022-05-05 22:02:02,657 - __main__ - INFO - Start split images with step 512
    2022-05-05 22:02:04,077 - __main__ - INFO - Start split images with step 768
    2022-05-05 22:02:04,818 - __main__ - INFO - Image ../test_data/imgs/20.jpg has been split successfully.
    2022-05-05 22:02:04,819 - __main__ - INFO - Start calculate similarities ...
    2022-05-05 22:02:48,182 - __main__ - INFO - Calculate similarities in 43.36335849761963s
    2022-05-05 22:02:48,182 - __main__ - INFO - Start generate heatmap ...
    2022-05-05 22:03:40,673 - __main__ - INFO - Generate finished, start optim ...
    2022-05-05 22:03:45,500 - __main__ - INFO - Generate heatmap in 57.31790471076965s
    2022-05-05 22:03:45,500 - __main__ - INFO - Saving heatmap in cache/heatmap_31.jpg ...
    2022-05-05 22:03:45,501 - __main__ - INFO - Saving heatmap in cache/addmap_31.jpg ...
    2022-05-05 22:03:45,501 - __main__ - INFO - Saving heatmap in cache/probmap_31.jpg ...
    2022-05-05 22:03:48,540 - __main__ - INFO - Saved ok.
    2022-05-05 22:03:59,562 - root - INFO - Eval cache/probmap_31.jpg
    2022-05-05 22:03:59,562 - root - INFO - +++++++ Calc the SLM METRICS +++++++
    2022-05-05 22:03:59,562 - root - INFO - ++++     ↑ Rsu [0 ~ 1]:0.9281   ++++
    2022-05-05 22:03:59,562 - root - INFO - ++++     ↑ Rda [0 ~ 1]:0.4689   ++++
    2022-05-05 22:03:59,562 - root - INFO - ++++     ↓ Ras [0 ~ 1]:0.0633   ++++
    2022-05-05 22:03:59,562 - root - INFO - ++++     ↑ Rmi [0 ~ 1]:0.8163   ++++
    2022-05-05 22:03:59,562 - root - INFO - ++++++++++++++++++++++++++++++++++++
    ...  

   $ ls cache/RSITMD_AMFMN
```



#### CUSTOMIZE MODEL

1. Put the pretrain ckpt file to **checkpoints**. 
2. Add your own model to **layers** and corresponding config yaml to **options/**.
3. Change **model_init.model_init** to your own models.
4. Add the class of  **EncoderSpecModel** to **model_encoder.py**.
5. Run:
```
python generate_selo.py --yaml_path option/xxx.yaml --cache_path cache/xxx
```

## EPILOGUE
So far, our attitude towards the semantic localization task is positive and optimistic, which realizes the detection at the semantic level with only the annotation at the caption level.
We sincerely hope that this project will facilitate the development of semantic localization tasks.
We welcome researchers to look into this direction, which is a possibility to achieve refined object semantic detection.

![visual image](./figure/selo_with_subtask.jpg)

**Fig.7.** Combine SeLo with other tasks. The top of the figure shows the detection results after add the SeLo map with query of “two parallel green
playgrounds”. The bottom of the figure shows the road extraction results after add the SeLo map with query of “the red rails where the grey train is located
run through the residential area”. (a) Source images. (b) Results of specific tasks. (c) Results of specific SeLo maps. (d) Fusion results of specific tasks and
SeLo map.

## CITATION
```
Z. Yuan et al., "Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3078451.

Z. Yuan et al., "A Lightweight Multi-scale Crossmodal Text-Image Retrieval Method In Remote Sensing," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2021.3124252.

Z. Yuan et al., "Remote Sensing Cross-Modal Text-Image Retrieval Based on Global and Local Information," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2022.3163706.

Z. Yuan et al., "Learning to Evaluate Peformance of Multi-modal Semantic Localization," undergoing review.
```
