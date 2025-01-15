#!/usr/bin/bash

#SBATCH -J ect2
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 6-0
#SBATCH -o logs/slurm-%A.out


if [ ! -f /local_datasets/edm2-imagenet-64x64.zip ]; then
  echo "edm2-imagenet-64x64.zip 파일이 없으므로 현재 디렉토리에서 복사합니다."
  cp edm2-imagenet-64x64.zip /local_datasets/
else
  echo "edm2-imagenet-64x64.zip 파일이 이미 존재합니다."
fi

bash run_imgnet_ecm.sh 2 6008 --desc bs128.100k.ect2.IN.xs.vanilla --outdir=ect2 --preset=edm2-img64-xs --transfer=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-xs-0134217-0.110.pkl

exit 0
