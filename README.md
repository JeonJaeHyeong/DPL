# Semantic Diversity-aware Prototype-based Learning for Unbiased Scene Graph Generation (ECCV 2024)

This is the official code for ECCV 2024 paper [Semantic Diversity-aware Prototype-based Learning for Unbiased Scene Graph Generation](https://arxiv.org/abs/2407.15396).

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions, the recommended configuration is cuda-11.1 & pytorch-1.8.2.  

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.


## Pretrained Models

1. For VG dataset, the pretrained object detector we used is provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), you can download it from [this link](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EQIy64T-EK9Er9y8kVCDaukB79gJwfSsEIbey9g0Xag6lg?e=wkKHJs). 
2. For GQA dataset, we trained a new object detector which can be downloaded from [this link](https://drive.google.com/drive/folders/1OS4-XOQmDZtL9Tssy1jWG-LTupBgIFEX?usp=drive_link). However, for better results, we suggest pretraining a new model on GQA, as we did not pretrain it multiple times to select the best pre-trained model.


## Perform training on Scene Graph Generation

### Set the dataset path

First, please refer to the ```pysgg/config/paths_catalog.py``` and set the ```DATA_DIR``` to be your dataset path, and organize all the files like this:
```bash
datasets
  |--vg   
    |--glove
      |--.... (glove files, will autoly download)
    |--VG_100K
      |--.... (images)
    |--VG-SGG-with-attri.h5 
    |--VG-SGG-dicts-with-attri.json
    |--image_data.json    
  |--gqa
    |--images
      |--.... (images)
    |--GQA_200_ID_Info.json
    |--GQA_200_Train.json
    |--GQA_200_Test.json

  |--detector_model
    |--pretrained_faster_rcnn
      |-vg_faster_det.pth
    |--GQA
      |--gqa_det.pth

```

### Choose a dataset

You can choose the training/testing dataset by setting the following parameter:
``` bash
GLOBAL_SETTING.DATASET_CHOICE 'VG'  # ['VG', 'GQA_200']
```

### Choose a task

To comprehensively evaluate the performance, we follow three conventional tasks: 1) **Predicate Classification (PredCls)** predicts the relationships of all the pairwise objects by employing the given ground-truth bounding boxes and classes; 2) **Scene Graph Classification (SGCls)** predicts the objects classes and their pairwise relationships by employing the given ground-truth object bounding boxes; and 3) **Scene Graph Detection (SGDet)** detects all the objects in an image, and predicts their bounding boxes, classes, and pairwise relationships.

For **Predicate Classification (PredCls)**, you need to set:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True
```
For **Scene Graph Classification (SGCls)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX True MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```
For **Scene Graph Detection (SGDet)**:
``` bash
MODEL.ROI_RELATION_HEAD.USE_GT_BOX False MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False
```

### Examples of the Training Command


Motifs + DPL for PredCls Task.
```bash
export gpu_num=1
export EXP=checkpoints
export CUDA_VISIBLE_DEVICES="0"
OUTPATH=$EXP/VG/motif/predcls/DPL
mkdir -p $OUTPATH

python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    OUTPUT_DIR $OUTPATH  \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifsLikePredictor_DPL \
    GLOBAL_SETTING.BASIC_ENCODER Motifs \
    SOLVER.IMS_PER_BATCH $[3*$gpu_num] \
    TEST.IMS_PER_BATCH $[$gpu_num] \
    SOLVER.MAX_ITER 60000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    MODEL.ROI_RELATION_HEAD.DPL.N_DIM 128   \
    MODEL.ROI_RELATION_HEAD.DPL.ALPHA 10    \
    MODEL.ROI_RELATION_HEAD.DPL.AVG_NUM_SAMPLE 20      \
    MODEL.ROI_RELATION_HEAD.DPL.RADIUS 1.0     \
    GLOBAL_SETTING.DATASET_CHOICE "VG" \
```


VCTree + DPL for PredCls Task.
```bash
export gpu_num=1
export EXP=checkpoints
export CUDA_VISIBLE_DEVICES="0"
OUTPATH=$EXP/VG/motif/predcls/DPL
mkdir -p $OUTPATH

python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
    tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    OUTPUT_DIR $OUTPATH  \
    MODEL.ROI_RELATION_HEAD.PREDICTOR VCTreePredictor_DPL \
    GLOBAL_SETTING.BASIC_ENCODER Motifs \
    SOLVER.IMS_PER_BATCH $[3*$gpu_num] \
    TEST.IMS_PER_BATCH $[$gpu_num] \
    SOLVER.MAX_ITER 60000 \
    SOLVER.VAL_PERIOD 2000 \
    SOLVER.CHECKPOINT_PERIOD 2000 \
    MODEL.ROI_RELATION_HEAD.DPL.N_DIM 128   \
    MODEL.ROI_RELATION_HEAD.DPL.ALPHA 10    \
    MODEL.ROI_RELATION_HEAD.DPL.AVG_NUM_SAMPLE 20      \
    MODEL.ROI_RELATION_HEAD.DPL.RADIUS 1.0     \
    GLOBAL_SETTING.DATASET_CHOICE "VG" \
```

You can simply run the scripts located in the script folder.

### Other options

Our model is set to sample the same number of samples for each predicate by default. However, it is also possible to allocate a different number of samples for each predicate by setting the following option to True. However, this option is only available for the VG dataset and cannot be used with the GQA dataset, as the pred_counts.pkl file is not available.
``` bash

MODEL.ROI_RELATION_HEAD.DPL.FREQ_BASED_DIFF_N True
```

This will allow more samples to be allocated to head classes and fewer samples to tail classes, while keeping the average number of samples the same.

## Evaluation

We provide the trained model from [this link](https://drive.google.com/drive/folders/1OS4-XOQmDZtL9Tssy1jWG-LTupBgIFEX?usp=drive_link). You can evaluate it by running the following command.

```bash
export gpu_num=1

checkpoint_dir="checkpoint path"
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10001 --nproc_per_node=1 \
    tools/relation_test_net.py \
    --config-file "$checkpoint_dir/config.yml" \
    TEST.IMS_PER_BATCH $[$gpu_num] \
    MODEL.ROI_RELATION_HEAD.EVALUATE_REL_PROPOSAL False \
    TEST.ALLOW_LOAD_FROM_CACHE True \
```

## Citations

If you find this project helps your research, please kindly consider citing our papers in your publications.

```
@article{jeon2024semantic,
  title={Semantic Diversity-aware Prototype-based Learning for Unbiased Scene Graph Generation},
  author={Jeon, Jaehyeong and Kim, Kibum and Yoon, Kanghoon and Park, Chanyoung},
  journal={arXiv preprint arXiv:2407.15396},
  year={2024}
}
```


## Acknowledgment
This repository is developed on top of the following code bases:

1. Scene graph benchmarking framework develped by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
2. A Toolkit for Scene Graph Benchmark in Pytorch by [Rongjie Li](https://github.com/SHTUPLUS/PySGG)
3. Stacked Hybrid-Attention and Group Collaborative Learning for Unbiased Scene Graph Generation in Pytorch by [Xingning Dong](https://github.com/dongxingning/SHA-GCL-for-SGG)
4. Vision Relation Transformer for Unbiased Scene Graph Generation by [Gopika Sudhakaran](https://github.com/visinf/veto)

-->