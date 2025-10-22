#!/bin/bash
#SBATCH -J hit-hourglass-1
#SBATCH -p dzagnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1

source /work/home/dengye/soft/anaconda3/bin/activate vim_mamba
python basicsr/train.py -opt options/T-Cloud/T-cloud-MHSVM_pos_tiny.yml
