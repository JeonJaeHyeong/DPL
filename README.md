# Semantic Diversity-aware Prototype-based Learning for Unbiased Scene Graph Generation (ECCV 2024)

This is the official code for ECCV 2024 paper [Semantic Diversity-aware Prototype-based Learning for Unbiased Scene Graph Generation](https://arxiv.org/abs/2407.15396).

Please wait a bit longer for the repository to be completed.

## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.


## Pretrained Models

For VG dataset, the pretrained object detector we used is provided by [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch), you can download it from [this link](https://1drv.ms/u/s!AmRLLNf6bzcir8xemVHbqPBrvjjtQg?e=hAhYCw). For GQA dataset, we trained a new object detector which can be downloaded from [this link](https://drive.google.com/file/d/1RHiIZRFyclii9X3FGd-bS9zIl94jsTTx/view?usp=drive_link). 

Put the checkpoint into the folder:
```
mkdir -p checkpoints/detection/pretrained_faster_rcnn/
mv /path/gqa_det.pth checkpoints/detection/pretrained_faster_rcnn/
mv /path/gqa_det.pth checkpoints/detection/pretrained_faster_rcnn/
```

Please wait a moment until the repository is complete.

<!--
### Scene Graph Generation Model
You can follow the following instructions to train your own, which takes 4 GPUs for train each SGG model. The results should be very close to the reported results given in paper.

We provide the one-click script for training our BGNN model( in `scripts/rel_train_BGNN_[vg/oiv6/oiv4].sh`)
or you can copy the following command to train
```
gpu_num=4 && python -m torch.distributed.launch --master_port 10028 --nproc_per_node=$gpu_num \
       tools/relation_train_net.py \
       --config-file "configs/e2e_relBGNN_vg.yaml" \
        DEBUG False \
        EXPERIMENT_NAME "BGNN-3-3" \
        SOLVER.IMS_PER_BATCH $[3*$gpu_num] \
        TEST.IMS_PER_BATCH $[$gpu_num] \
        SOLVER.VAL_PERIOD 3000 \
        SOLVER.CHECKPOINT_PERIOD 3000 

```
We also provide the trained model pth of [BGNN(vg)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/Ee4PdxluTphEicUDckJIfmEBisAyUgkjeuerN_rjrG1CIw?e=pgr8a5),[BGNN(oiv6)](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EdKOrWAOf4hMiDWbR3CgYrMB9w7ZwWul-Wc6IUSbs51Idw?e=oEEHIQ)



## Test
Similarly, we also provide the `rel_test.sh` for directly produce the results from the checkpoint provide by us.
By replacing the parameter of `MODEL.WEIGHT` to the trained model weight and selected dataset name in `DATASETS.TEST`, you can directly eval the model on validation or test set.


## Citations

If you find this project helps your research, please kindly consider citing our papers in your publications.

```
@InProceedings{Li_2021_CVPR,
    author    = {Li, Rongjie and Zhang, Songyang and Wan, Bo and He, Xuming},
    title     = {Bipartite Graph Network With Adaptive Message Passing for Unbiased Scene Graph Generation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {11109-11119}
}
```


## Acknowledgment
This repository is developed on top of the scene graph benchmarking framwork develped by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)


-->