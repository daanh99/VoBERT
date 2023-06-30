#!/bin/bash

#SBATCH --job-name="Logbert"
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=32GB
#SBATCH --account=education-eemcs-msc-cs

module load 2022r2
module load gpu
module load python/3.8.12
module load py-pip
module load cuda/11.7

previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

srun source env/bin/activate && ./init.sh && python main.py > logbert.log

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
