#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -n1
#SBATCH -c8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=tmp/repo_dmcgb_%j.log

export MUJOCO_GL=egl

SEED=0
DOMAIN="walker"
TASK="walk"
RUN_NAME="repo_${DOMAIN}_${TASK}_seed_${SEED}"
srun python experiments/train_repo.py --algo=repo \
                                      --wandb_group=repo_${DOMAIN}_${TASK}_baseline \
                                      --wandb_run_name=$RUN_NAME \
                                      --domain_name=$DOMAIN \
                                      --task_name=$TASK \
                                      --env_id=${DOMAIN}_${TASK} \
                                      --num_steps=1_000_000 \
                                      --seed=$SEED
