#!/bin/bash
#SBATCH -N 1 # n nodes
#SBATCH --ntasks-per-node=32 # cpu cores
##SBATCH --time 2-00:00:00
#SBATCH --job-name=train_sdf
##SBATCH --error logs/%J.err_
##SBATCH --output logs/%J.out_
#SBATCH --mem=230GB
#SBATCH --partition=gpu
##SBATCH --gres=gpu:2

module load tools/anaconda3/2021.05
eval "$(conda shell.bash hook)"
conda activate historic


now=$(date +"%Y%m%d_%H%M%S")
jobname="train-$1-$now"
echo "job name is $jobname"

config_file=$2
mkdir -p log
mkdir -p logs/${jobname}
cp ${config_file} logs/${jobname}

python train.py --cfg_path ${config_file} --save_path /home/ro38seb/3d_recon/neural_recon_w/ \
  --num_gpus $3 --num_nodes $4 \
  --num_epochs 20 --batch_size 2048 --test_batch_size 512 --num_workers 30 \
  --exp_name ${jobname} 2>&1|tee log/${jobname}.log \
