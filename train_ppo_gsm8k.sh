#!/bin/bash

# PPO Training Script for GSM8K Dataset
# 
# This script trains a Qwen2.5-0.5B model using PPO (Proximal Policy Optimization) 
# on the GSM8K mathematical reasoning dataset.
#
# Prerequisites:
# - CUDA >= 12.4 and cuDNN >= 9.8.0
# - verl submodule initialized and installed
# - GSM8K dataset available at $HOME/data/gsm8k/
#
# Usage:
#   ./train_ppo_gsm8k.sh
#
# Environment Variables (can be customized):
#   PROJECT_NAME: WandB project name (default: gsm8k-training)
#   RUN_NAME: WandB run name (default: qwen-2.5-0.5b-low-mem)
#   WANDB_ENTITY: WandB entity name (default: thinhlpg)
#   CUDA_VISIBLE_DEVICES: GPU to use (default: 0)

# Set environment variables
export HYDRA_FULL_ERROR=1

# Fix for CUDA initialization issues
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Prevent Ray from modifying CUDA_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# Set specific GPU if you have multiple
export CUDA_VISIBLE_DEVICES=0

# Set your project and run names
export PROJECT_NAME="gsm8k-training"  # Change this to your project name
export RUN_NAME="qwen-2.5-0.5b-low-mem"  # Change this to your specific run name
export WANDB_ENTITY="thinhlpg"
 
# Run the training with minimal memory settings
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=$HOME/data/gsm8k/train.parquet \
 data.val_files=$HOME/data/gsm8k/test.parquet \
 data.train_batch_size=2 \
 data.max_prompt_length=256 \
 data.max_response_length=128 \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=1 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.2 \
 actor_rollout_ref.rollout.max_num_seqs=64 \
 actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 critic.optim.lr=1e-5 \
 critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
 critic.ppo_micro_batch_size_per_gpu=1 \
 critic.forward_micro_batch_size_per_gpu=1 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','wandb'] \
 trainer.project_name=$PROJECT_NAME \
 trainer.experiment_name=$RUN_NAME \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=1 \
 trainer.nnodes=1 \
 trainer.save_freq=1 \
 trainer.test_freq=1 \
 trainer.total_epochs=10 2>&1 | tee verl_demo_low_mem.log