import os
from typing import Tuple

import flax
import gym
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner import Learner

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('models_dir', './models/', 'Directory to save flax model checkpoints.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10, 'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False
)

flags.DEFINE_float('kalman_q', 1.0, 'Kalman process noise Q.')
flags.DEFINE_float('kalman_r', 5e-2, 'Kalman measurement noise R.')
flags.DEFINE_float('gate_frac', 0.20, 'Innovation gate threshold as a fraction of max_action.')
flags.DEFINE_float('gate_r_mult', 20.0, 'R_t = R * gate_r_mult when innovation is large.')
flags.DEFINE_float('blend_beta', 0.90, 'Output blend: a=(1-beta)*a_kf + beta*a_raw.')
flags.DEFINE_float('kalman_p_init', 1.0, 'Initial covariance of Kalman filter.')
flags.DEFINE_float('dt', 0.01, 'Control timestep used by inference-time Kalman smoothing.')

flags.DEFINE_boolean('ab_test', True, 'Evaluate both baseline and kalman during eval.')
flags.DEFINE_boolean('eval_with_kalman', False, 'If ab_test=False, whether to evaluate with Kalman.')


def normalize(dataset):
    trajs = split_into_trajectories(
        dataset.observations,
        dataset.actions,
        dataset.rewards,
        dataset.masks,
        dataset.dones_float,
        dataset.next_observations
    )

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew
        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str, seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)

    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    dataset = D4RLDataset(env)

    if 'antmaze' in FLAGS.env_name:
        dataset.rewards -= 1.0
    elif ('halfcheetah' in FLAGS.env_name or
          'walker2d' in FLAGS.env_name or
          'hopper' in FLAGS.env_name):
        normalize(dataset)

    return env, dataset


def maybe_log_eval(summary_writer, eval_stats, step, prefix):
    for k, v in eval_stats.items():
        summary_writer.add_scalar(f'{prefix}/{k}', v, step)


def print_compare_table(step, base_stats, kalman_stats):
    has_score = ('d4rl_score' in base_stats) and ('d4rl_score' in kalman_stats)

    print("\n" + "=" * 78)
    print(f"[Evaluation @ step {step}]")
    print("=" * 78)
    if has_score:
        print(f"{'Method':<12} {'Return':>12} {'Length':>12} {'D4RL Score':>14} {'Avg Jerk':>14}")
        print("-" * 78)
        print(f"{'Baseline':<12} {base_stats['return']:>12.3f} {base_stats['length']:>12.3f} {base_stats['d4rl_score']:>14.3f} {base_stats['jerk']:>14.6f}")
        print(f"{'Kalman':<12}   {kalman_stats['return']:>12.3f} {kalman_stats['length']:>12.3f} {kalman_stats['d4rl_score']:>14.3f} {kalman_stats['jerk']:>14.6f}")
        print("-" * 78)
        print(f"{'Delta(K-B)':<12} {kalman_stats['return'] - base_stats['return']:>12.3f} "
              f"{kalman_stats['length'] - base_stats['length']:>12.3f} "
              f"{kalman_stats['d4rl_score'] - base_stats['d4rl_score']:>14.3f} "
              f"{kalman_stats['jerk'] - base_stats['jerk']:>14.6f}")
    else:
        print(f"{'Method':<12} {'Return':>12} {'Length':>12} {'Avg Jerk':>14}")
        print("-" * 60)
        print(f"{'Baseline':<12} {base_stats['return']:>12.3f} {base_stats['length']:>12.3f} {base_stats['jerk']:>14.6f}")
        print(f"{'Kalman':<12}   {kalman_stats['return']:>12.3f} {kalman_stats['length']:>12.3f} {kalman_stats['jerk']:>14.6f}")
        print("-" * 60)
        print(f"{'Delta(K-B)':<12} {kalman_stats['return'] - base_stats['return']:>12.3f} "
              f"{kalman_stats['length'] - base_stats['length']:>12.3f} "
              f"{kalman_stats['jerk'] - base_stats['jerk']:>14.6f}")
    print("=" * 78)


def save_compare_table(save_dir, seed, step, base_stats, kalman_stats):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f'compare_table_seed{seed}_step{step}.txt')

    has_score = ('d4rl_score' in base_stats) and ('d4rl_score' in kalman_stats)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f"[Evaluation @ step {step}]\n")
        f.write("=" * 78 + "\n")
        if has_score:
            f.write(f"{'Method':<12} {'Return':>12} {'Length':>12} {'D4RL Score':>14} {'Avg Jerk':>14}\n")
            f.write("-" * 78 + "\n")
            f.write(f"{'Baseline':<12} {base_stats['return']:>12.3f} {base_stats['length']:>12.3f} {base_stats['d4rl_score']:>14.3f} {base_stats['jerk']:>14.6f}\n")
            f.write(f"{'Kalman':<12}   {kalman_stats['return']:>12.3f} {kalman_stats['length']:>12.3f} {kalman_stats['d4rl_score']:>14.3f} {kalman_stats['jerk']:>14.6f}\n")
            f.write("-" * 78 + "\n")
            f.write(f"{'Delta(K-B)':<12} {kalman_stats['return'] - base_stats['return']:>12.3f} "
                    f"{kalman_stats['length'] - base_stats['length']:>12.3f} "
                    f"{kalman_stats['d4rl_score'] - base_stats['d4rl_score']:>14.3f} "
                    f"{kalman_stats['jerk'] - base_stats['jerk']:>14.6f}\n")
        else:
            f.write(f"{'Method':<12} {'Return':>12} {'Length':>12} {'Avg Jerk':>14}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Baseline':<12} {base_stats['return']:>12.3f} {base_stats['length']:>12.3f} {base_stats['jerk']:>14.6f}\n")
            f.write(f"{'Kalman':<12}   {kalman_stats['return']:>12.3f} {kalman_stats['length']:>12.3f} {kalman_stats['jerk']:>14.6f}\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Delta(K-B)':<12} {kalman_stats['return'] - base_stats['return']:>12.3f} "
                    f"{kalman_stats['length'] - base_stats['length']:>12.3f} "
                    f"{kalman_stats['jerk'] - base_stats['jerk']:>14.6f}\n")
        f.write("=" * 78 + "\n")

    print(f"[table] saved to: {out_path}")

def save_flax_model(model, path):
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(model.params))

def save_iql_models(agent, save_dir, env_name, seed, prefix="IQL"):
    os.makedirs(save_dir, exist_ok=True)

    actor_path = os.path.join(save_dir, f"{prefix}_{env_name}_{seed}_actor.flax")
    critic_path = os.path.join(save_dir, f"{prefix}_{env_name}_{seed}_critic.flax")
    value_path = os.path.join(save_dir, f"{prefix}_{env_name}_{seed}_value.flax")

    save_flax_model(agent.actor, actor_path)
    save_flax_model(agent.critic, critic_path)
    save_flax_model(agent.value, value_path)

    print(f"[model] saved actor to: {actor_path}")
    print(f"[model] saved critic to: {critic_path}")
    print(f"[model] saved value to: {value_path}")

def main(_):
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)),
        write_to_disk=True
    )
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)
    agent = Learner(
        FLAGS.seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        max_steps=FLAGS.max_steps,
        kalman_q=FLAGS.kalman_q,
        kalman_r=FLAGS.kalman_r,
        gate_frac=FLAGS.gate_frac,
        gate_r_mult=FLAGS.gate_r_mult,
        blend_beta=FLAGS.blend_beta,
        kalman_p_init=FLAGS.kalman_p_init,
        **kwargs
    )

    eval_returns = []
    eval_returns_baseline = []
    eval_returns_kalman = []

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm
    ):
        batch = dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            for k, v in update_info.items():
                if hasattr(v, "ndim") and v.ndim == 0:
                    summary_writer.add_scalar(f'training/{k}', v, i)
                else:
                    summary_writer.add_histogram(f'training/{k}', v, i)
            summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            if FLAGS.ab_test:
                eval_stats_base = evaluate(
                    agent,
                    env,
                    FLAGS.eval_episodes,
                    apply_kalman=False,
                    dt=FLAGS.dt
                )

                eval_stats_kalman = evaluate(
                    agent,
                    env,
                    FLAGS.eval_episodes,
                    apply_kalman=True,
                    dt=FLAGS.dt
                )

                print_compare_table(i, eval_stats_base, eval_stats_kalman)
                save_compare_table(FLAGS.save_dir, FLAGS.seed, i, eval_stats_base, eval_stats_kalman)

                maybe_log_eval(summary_writer, eval_stats_base, i, 'evaluation_baseline')
                maybe_log_eval(summary_writer, eval_stats_kalman, i, 'evaluation_kalman')
                summary_writer.flush()

                eval_returns_baseline.append((i, eval_stats_base['return']))
                eval_returns_kalman.append((i, eval_stats_kalman['return']))

                np.savetxt(
                    os.path.join(FLAGS.save_dir, f'{FLAGS.seed}_baseline.txt'),
                    eval_returns_baseline,
                    fmt=['%d', '%.6f']
                )
                np.savetxt(
                    os.path.join(FLAGS.save_dir, f'{FLAGS.seed}_kalman.txt'),
                    eval_returns_kalman,
                    fmt=['%d', '%.6f']
                )

            else:
                eval_stats = evaluate(
                    agent,
                    env,
                    FLAGS.eval_episodes,
                    apply_kalman=FLAGS.eval_with_kalman,
                    dt=FLAGS.dt
                )

                print("\n" + "=" * 60)
                print(f"[Evaluation @ step {i}]")
                print("=" * 60)
                for k, v in eval_stats.items():
                    print(f"{k:<16}: {v:.6f}")
                print("=" * 60)

                prefix = 'evaluation_kalman' if FLAGS.eval_with_kalman else 'evaluation_baseline'
                maybe_log_eval(summary_writer, eval_stats, i, prefix)
                summary_writer.flush()

                eval_returns.append((i, eval_stats['return']))
                np.savetxt(
                    os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                    eval_returns,
                    fmt=['%d', '%.6f']
                )

            save_iql_models(
                agent,
                FLAGS.models_dir,
                FLAGS.env_name,
                FLAGS.seed,
                prefix="IQL"
            )

    print("\nTraining finished.")


if __name__ == '__main__':
    app.run(main)