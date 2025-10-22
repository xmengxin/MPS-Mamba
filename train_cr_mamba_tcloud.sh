#!/usr/bin/env bash

CONFIG=$1

#python -m torch.distributed.launch --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/Train-nb/train_cascade62_HourGlass2dw_Rsinsa_DGC_x2_16_32.yml --auto_resume --launcher pytorch
#conda activate RGT
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt options/T-Cloud/T-cloud-MHSVM_pos_tiny.yml
