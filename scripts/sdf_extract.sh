#!/bin/bash
#SBATCH -N 1 # n nodes
#SBATCH --ntasks-per-node=16 # cpu cores
#SBATCH --time 03:00:00
#SBATCH --job-name=sdf_extract
#SBATCH --error logs/%J.err_
#SBATCH --output logs/%J.out_
#SBATCH --mem=128GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

set -x
set -u

now=$(date +"%Y%m%d_%H%M%S")
jobname="sdf-$1-$now"
echo "job name is $jobname"

module load tools/anaconda3/2021.05
eval "$(conda shell.bash hook)"
conda activate historic

#now=$(date +"%Y%m%d_%H%M%S")
#jobname="sdf-$1-$now"
#echo "job name is $jobname"

config_file=$2
ckpt_path=$3
eval_level=$4

python -m torch.distributed.launch --nproc_per_node=2  tools/extract_mesh.py \
--cfg_path ${config_file} \
--mesh_size 1024 --chunk 102144 \
--ckpt_path $ckpt_path \
--mesh_radius 1 --mesh_origin "0, 0, 0" --chunk_rgb 1024 --vertex_color --eval_level ${eval_level} 
