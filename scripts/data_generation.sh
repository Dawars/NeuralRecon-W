#!/bin/bash
#SBATCH -N 1 # n nodes
#SBATCH --ntasks-per-node=16 # cpu cores
#SBATCH --time 2-00:00:00
#SBATCH --job-name=create_cache
##SBATCH --error logs/%J.err_
##SBATCH --output logs/%J.out_
#SBATCH --mem=200GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load tools/anaconda3/2021.05
eval "$(conda shell.bash hook)"
conda activate historic

set -x
set -u
set -e

now=$(date +"%Y%m%d_%H%M%S")
echo "working directory is $(pwd)"
jobname="data-generation-$1-$now"
dataset_name="phototourism"
cache_dir="cache_splits"
root_dir=$1
min_observation=-1

if [ ! -f $root_dir/*.tsv ]; then
    python tools/prepare_data/prepare_data_split.py \
    --root_dir $root_dir \
    --num_test 10 \
    --min_observation $min_observation --roi_threshold 0 --static_threshold 0
fi
python tools/prepare_data/prepare_data_cache.py \
--root_dir $root_dir \
--dataset_name $dataset_name --cache_dir $cache_dir \
--img_downscale $2 \
--semantic_map_path semantic_maps \
--split_to_chunks 64 \
2>&1|tee log/${jobname}.log
