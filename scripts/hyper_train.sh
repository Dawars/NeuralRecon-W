
now=$(date +"%Y%m%d_%H%M%S")
jobname="train-$1-$now"
echo "job name is $jobname"

config_file=$2

python train_hyper.py  --cfg_path ${config_file} --save_path /home/ro38seb/3d_recon/neural_recon_w/ \
  --num_epochs 20 --batch_size 2048 --test_batch_size 512 --num_workers 30 \
  --exp_name ${jobname}