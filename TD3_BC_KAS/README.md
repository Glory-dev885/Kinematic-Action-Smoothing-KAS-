# TD3_BC_KAS

A plug-and-play inference-stage action smoothing framework for offline reinforcement learning in continuous control.

This repository implements **TD3+BC with Kinematic Action Smoothing (KAS)** on D4RL locomotion benchmarks.  
The proposed method performs **action smoothing only at inference time**, without modifying the original offline RL training pipeline.

---

## Features

- Offline RL baseline: TD3+BC
- Inference-stage smoothing: Kalman-based KAS
- Supports D4RL locomotion tasks
- Metrics:
  - D4RL Score
  - Action Jerk
- Batch evaluation across environments and seeds
- CSV result export

---

## Project Structure

.
├── main.py
├── TD3_BC_KAS.py
├── utils.py
├── requirements.txt
├── run_experiments.sh
└── README.md

---

## Installation

pip install -r requirements.txt

If d4rl installation fails:

pip install git+https://github.com/Farama-Foundation/D4RL

---

## Training

Single run:

python main.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --normalize \
  --max_timesteps 1000000 \
  --eval_freq 5000 \
  --batch_size 256
  --save_model

Multi-env + multi-seed:

for env in hopper-medium-v2 walker2d-medium-v2 halfcheetah-medium-v2
do
  for seed in 0 1 2 3 4
  do
    python main.py \
      --env $env \
      --seed $seed \
      --normalize \
      --max_timesteps 1000000 \
      --eval_freq 5000
      --save_model
  done
done

---

## Evaluation

Evaluate one model (baseline + KAS):

python main.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --normalize \
  --models_dir "./models" \
  --load_model TD3_BC_hopper-medium-v2_0 \
  --eval_only \
  --ab_test \
  --blend_beta 0.90 \
  --gate_R_mult 20 \
  --gate_frac 0.05 \
  --kalman_Q 1.0 \
  --kalman_R 0.05

---

Batch evaluation + CSV:

python main.py \
  --normalize \
  --models_dir "./models" \
  --aggregate_all_envs \
  --seed_start 0 \
  --seed_end 4 \
  --eval_only \
  --ab_test \
  --blend_beta 0.90 \
  --gate_R_mult 20 \
  --gate_frac 0.05 \
  --kalman_Q 1.0 \
  --kalman_R 0.05 \
  --out_csv "./TD3_BC_KAS.csv"

or:

bash run_experiments.sh