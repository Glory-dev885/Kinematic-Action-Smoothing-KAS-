# IQL_KAS

A plug-and-play inference-stage action smoothing framework for offline reinforcement learning in continuous control.

This repository implements **Implicit Q-Learning (IQL)** with **Kinematic Action Smoothing (KAS)** on D4RL locomotion benchmarks.
The proposed method performs **action smoothing only at inference time**, without modifying the original offline RL training pipeline.

---

## Features

* Offline RL baseline: IQL
* Inference-stage smoothing: Kalman-based KAS
* Supports D4RL locomotion benchmarks
* Metrics:

  * Return
  * D4RL Score
  * Action Jerk
* Batch evaluation across environments and seeds
* CSV result export
* Flax model checkpoint saving and loading

---

## Project Structure

.
├── configs
├── wrappers
├── train_offline.py
├── main.py
├── learner.py
├── evaluation.py
├── run_experiments.sh
├── requirements.txt
├── dataset_utils.py
├── policy.py
├── value_net.py
├── actor.py
├── critic.py
├── common.py
└── README.md

---

## Installation

pip install -r requirements.txt

If d4rl installation fails:

pip install git+https://github.com/Farama-Foundation/D4RL

---

## Training

python train_offline.py 
--env_name halfcheetah-expert-v2 
--seed 0 
--save_dir ./tmp 
--models_dir ./models

Multi-seed training:

for seed in 0 1 2 3 4
do
python train_offline.py 
--env_name halfcheetah-expert-v2 
--seed $seed 
--save_dir ./tmp 
--models_dir ./models
done

---

## Evaluation

Single environment:

python main.py 
--env_name halfcheetah-expert-v2 
--models_dir ./models 
--output_dir ./exp_csv 
--model_prefix IQL 
--seed 0 
--eval_episodes 10

Aggregate over seeds:

python main.py 
--env_name halfcheetah-expert-v2 
--models_dir ./models 
--output_dir ./exp_csv 
--model_prefix IQL 
--aggregate_seeds 
--seed_start 0 
--seed_end 4 
--eval_episodes 10

Aggregate all environments:

python main.py 
--models_dir ./models 
--output_dir ./exp_csv 
--model_prefix IQL 
--aggregate_all_envs 
--seed_start 0 
--seed_end 4 
--eval_episodes 10

Or run:

bash run_experiments.sh

---