#!/bin/sh
#SBATCH --job-name vaeimg2
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=48GB
#SBATCH --time=200:00:00


module load cuda/10.2
module load cudnn/7.6.5-cuda10.2

python3 train_coco.py --epoch=300 --dataset='coco' --mode='train' --batch_size=19
