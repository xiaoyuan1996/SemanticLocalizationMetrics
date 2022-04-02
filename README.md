# Learning to Evaluate Peformance of Multi-modal Semantic Localization
##### Author: Zhiqiang Yuan, Chongyang Li, et. al 

<a href="https://github.com/xiaoyuan1996/retrievalSystem"><img src="https://travis-ci.org/Cadene/block.bootstrap.pytorch.svg?branch=master"/></a>
![Supported Python versions](https://img.shields.io/badge/python-3.7-blue.svg)
![Supported OS](https://img.shields.io/badge/Supported%20OS-Linux-yellow.svg)
![npm License](https://img.shields.io/npm/l/mithril.svg)
<a href="https://pypi.org/project/mitype/"><img src="https://img.shields.io/pypi/v/mitype.svg"></a>

### -------------------------------------------------------------------------------------
### Welcome :+1:_<big>`Fork and Star`</big>_:+1:, then we'll let you know when we update

### -------------------------------------------------------------------------------------

## INTRODUCTION
An official evaluation metric for semantic localization.

The semantic localization task refers to using cross-modal information such as text to quickly localize RS images at the semantic level [\[link\]](https://ieeexplore.ieee.org/document/9437331).
This task implements semantic-level detection, which only uses caption-level supervision information.
In our opinion, it is a meaningful and interesting work, which realizes the unification of sub-tasks such as detection and segmentation.

We contribute test sets, evaluation metrics and baselines for semantic localization, and provide a detailed demo to use this evaluation framework.
Any questions can open a Github [issue](https://github.com/xiaoyuan1996/SemanticLocalizationMetrics/issues).
Start and enjoy!

### TESTDATA

### METRICS

### BASELINES

## IMPLEMENTATION

### ENVIRONMENT

1. Install the requirements
    
```
   $ apt-get install python3
   $ git clone git@github.com:xiaoyuan1996/SemanticLocalizationMetrics.git
   $ cd SemanticLocalizationMetrics
   $ pip install -r requirements.txt
```

2. Prepare checkpoints and test iamges


* Download pretrain checkpoints to **./predict/checkpoints/** from [BaiduYun]() or [GoogleDriver](), make sure:
   
  + ./predict/checkpoints/
    + xxx.pth

* Download test images to **./test_data/imgs/** from [BaiduYun]() or [GoogleDriver](), make sure:

  + ./test_data/imgs/
    + xxx.tif

3. Check the environments

```
    $ cd predict
    $ python model_encoder.py
    
    ECHO: visual_vector: (512,)
    ECHO: text_vector: (512,)
    ECHO: Encoder test successful!
```


### RUN THE DEMO
1. Run the follow command 
```
   $ cd predict
   $ python generate_slm.py
   
    2022-03-30 11:53:50,300 - __main__ - INFO - Processing 1/24: 1.tif
    2022-03-30 11:53:50,971 - __main__ - INFO - img size:6105x6105
    2022-03-30 11:53:50,971 - __main__ - INFO - Start split images with step 512
    2022-03-30 11:53:55,026 - __main__ - INFO - Start split images with step 768
    2022-03-30 11:53:56,870 - __main__ - INFO - Image ../test_data/imgs/1.tif has been split successfully.
    2022-03-30 11:53:56,870 - __main__ - INFO - Start calculate similarities ...
    2022-03-30 11:54:40,450 - __main__ - INFO - Calculate similarities in 43.57924294471741s
    2022-03-30 11:54:40,450 - __main__ - INFO - Start generate heatmap ...
    2022-03-30 11:55:02,675 - __main__ - INFO - Generate finished, start optim ...
    2022-03-30 11:55:07,296 - __main__ - INFO - Generate heatmap in 26.84576725959778s
    2022-03-30 11:55:07,297 - __main__ - INFO - Saving heatmap in cache/heatmap_1.jpg ...
    2022-03-30 11:55:07,297 - __main__ - INFO - Saving heatmap in cache/addmap_1.jpg ...
    2022-03-30 11:55:08,542 - __main__ - INFO - Saved ok.
    ...  
   $ ls cache
```

2. Generated heatmaps and addmaps will be saved in **cache/**.

### CUSTOMIZE MODEL

#### OVERRIDE FUNCTIONS

1. Put the pretrain ckpt file to **checkpoints**. 
2. Add your own model to **layers** and corresponding config yaml to **options/**.
3. Change **model_init.model_init** to your own models.
4. Override **model_encoder.image_encoder** and **model_encoder.text_encoder**.
5. Run:
```
python generate_slm.py --yaml_path option/xxx.yaml
```



#### CHECK
#### RUN

## EPILOGUE
So far, our attitude towards the semantic localization task is positive and optimistic, which realizes the detection at the semantic level with only the annotation at the caption level.
We sincerely hope that this project will facilitate the development of semantic localization tasks.
We welcome researchers to look into this direction, which is a possibility to achieve refined object semantic detection.
## CITATION