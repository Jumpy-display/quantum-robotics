#!/bin/bash
#SBATCH -c 32                   # 32 CPUs
#SBATCH --mem=32gb              # 32 GB RAM
#SBATCH --gres=gpu:rtxa5000:1   # 1 GPU (A6000)
#SBATCH --time=2-00:00:00       # 2 days
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --output=sbatch_logs/train_ppo_%j.out
#SBATCH --error=sbatch_logs/train_ppo_%j.err

export HOME=/fs/nexus-projects/open_vectormap/dqacs/
export TMPDIR=$HOME/tmp

cd $HOME
source $HOME/venvs/metaworld2/bin/activate
cd metaworld-assembly-v3
python PPO_metaworld.py