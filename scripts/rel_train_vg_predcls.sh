#!/bin/bash
export OMP_NUM_THREADS=1
export gpu_num=2
export EXP=checkpoints
export CUDA_VISIBLE_DEVICES="7,8"

OUTPATH=$EXP/VG/motif/predcls/put_exp_name
mkdir -p $OUTPATH
cp pysgg/modeling/roi_heads/relation_head/roi_relation_predictors.py $OUTPATH/roi_relation_predictors.py

#python tools/relation_train_net.py \
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