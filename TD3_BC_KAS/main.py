import os
import re
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import gym
import d4rl

import utils
import TD3_BC_KAS


# ============================================================
# Utils
# ============================================================
def get_env_dt(env, fallback_dt: float = 0.01) -> float:
    try:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "sim") and hasattr(unwrapped, "frame_skip"):
            sim = unwrapped.sim
            if hasattr(sim, "model") and hasattr(sim.model, "opt") and hasattr(sim.model.opt, "timestep"):
                return float(sim.model.opt.timestep) * int(unwrapped.frame_skip)
    except Exception:
        pass

    for attr in ["dt", "control_dt", "timestep"]:
        if hasattr(env, attr):
            try:
                return float(getattr(env, attr))
            except Exception:
                pass

    return float(fallback_dt)


def make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env.seed(seed)
    env.action_space.seed(seed)
    return env


def infer_model_name(env_name: str, seed: int) -> str:
    return f"TD3_BC_{env_name}_{seed}"


def parse_model_filename(actor_filename: str) -> Optional[Tuple[str, int]]:
    name = os.path.basename(actor_filename)
    m = re.match(r"^TD3_BC_(.+)_(\d+)_actor$", name)
    if m is None:
        return None

    env_name = m.group(1)
    seed = int(m.group(2))
    return env_name, seed


def discover_models(
    models_dir: str,
    seed_start: int,
    seed_end: int,
) -> List[Dict]:
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"models_dir does not exist: {models_dir}")

    discovered = []
    for fname in os.listdir(models_dir):
        if not fname.endswith("_actor"):
            continue

        parsed = parse_model_filename(fname)
        if parsed is None:
            continue

        env_name, seed = parsed
        if not (seed_start <= seed <= seed_end):
            continue

        model_base = infer_model_name(env_name, seed)
        actor_path = os.path.join(models_dir, f"{model_base}_actor")
        critic_path = os.path.join(models_dir, f"{model_base}_critic")
        if not (os.path.exists(actor_path) and os.path.exists(critic_path)):
            continue

        discovered.append(
            {
                "env": env_name,
                "seed": seed,
                "model_base": model_base,
                "model_path": os.path.join(models_dir, model_base),
            }
        )

    discovered.sort(key=lambda x: (x["env"], x["seed"]))
    return discovered


def init_policy_for_env(args, env_name: str, seed: int):
    env = make_env(env_name, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    env_dt = get_env_dt(env, fallback_dt=0.01)
    dt = float(args.dt) if args.dt > 0 else float(env_dt)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    gate_threshold = float(args.gate_frac) * max_action

    policy = TD3_BC_KAS.TD3_BC_KAS(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        alpha=args.alpha,
        kalman_q=args.kalman_Q,
        kalman_r=args.kalman_R,
        gate_threshold=gate_threshold,
        gate_r_mult=args.gate_R_mult,
        blend_beta=args.blend_beta,
        kalman_p_init=1.0,
    )

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0.0, 1.0

    return env, policy, mean, std, dt, env_dt


def eval_policy_with_metrics(
    policy,
    env_name: str,
    seed: int,
    mean,
    std,
    dt: float,
    apply_kalman: bool,
    eval_episodes: int = 10,
    seed_offset: int = 100,
):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + seed_offset)
    eval_env.action_space.seed(seed + seed_offset)

    total_reward = 0.0
    jerks = []

    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        policy.reset_filters()

        actions = []
        while not done:
            s = (np.array(state).reshape(1, -1) - mean) / std
            action = policy.select_action(s, dt=dt, apply_kalman=apply_kalman)
            actions.append(np.array(action, dtype=np.float64).copy())

            state, reward, done, _ = eval_env.step(action)
            total_reward += reward

        if len(actions) >= 3:
            jerks.append(TD3_BC_KAS.compute_jerk(actions))

    avg_reward = total_reward / eval_episodes
    avg_jerk = float(np.mean(jerks)) if len(jerks) > 0 else 0.0
    d4rl_score = float(eval_env.get_normalized_score(avg_reward) * 100.0)

    eval_env.close()

    print("---------------------------------------")
    print(f"Environment:   {env_name}")
    print(f"Seed:          {seed}")
    print(f"Eval episodes: {eval_episodes}")
    print(f"apply_kalman:  {apply_kalman}")
    print(f"D4RL Score:    {d4rl_score:.6f}")
    print(f"Avg Jerk:      {avg_jerk:.6f}")
    print("---------------------------------------")

    return {"d4rl_score": d4rl_score, "jerk": avg_jerk}


def write_env_summary_csv(rows: List[Dict], out_csv: str):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "env",
        "n_seeds",
        "baseline_score_mean",
        "baseline_score_std",
        "baseline_jerk_mean",
        "baseline_jerk_std",
        "kalman_score_mean",
        "kalman_score_std",
        "kalman_jerk_mean",
        "kalman_jerk_std",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[csv] Results saved to: {out_csv}")


def summarize_by_env(per_seed_results: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in per_seed_results:
        grouped[row["env"]].append(row)

    summary_rows = []
    for env_name in sorted(grouped.keys()):
        rows = grouped[env_name]

        baseline_scores = np.array([r["baseline_score"] for r in rows], dtype=np.float64)
        baseline_jerks = np.array([r["baseline_jerk"] for r in rows], dtype=np.float64)
        kalman_scores = np.array([r["kalman_score"] for r in rows], dtype=np.float64)
        kalman_jerks = np.array([r["kalman_jerk"] for r in rows], dtype=np.float64)

        summary_rows.append(
            {
                "env": env_name,
                "n_seeds": len(rows),
                "baseline_score_mean": float(np.mean(baseline_scores)),
                "baseline_score_std": float(np.std(baseline_scores)),
                "baseline_jerk_mean": float(np.mean(baseline_jerks)),
                "baseline_jerk_std": float(np.std(baseline_jerks)),
                "kalman_score_mean": float(np.mean(kalman_scores)),
                "kalman_score_std": float(np.std(kalman_scores)),
                "kalman_jerk_mean": float(np.mean(kalman_jerks)),
                "kalman_jerk_std": float(np.std(kalman_jerks)),
            }
        )

    return summary_rows


# ============================================================
# Args
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--normalize", action="store_true", help="normalize states using dataset stats")
    parser.add_argument("--eval_episodes", default=10, type=int)

    parser.add_argument("--eval_only", action="store_true",
                        help="only evaluate a loaded model, no training")
    parser.add_argument("--apply_kalman", action="store_true",
                        help="apply kalman smoothing during evaluation")
    parser.add_argument("--no_kalman", action="store_true",
                        help="disable kalman smoothing during evaluation (overrides --apply_kalman)")
    parser.add_argument("--ab_test", action="store_true",
                        help="run both baseline (no kalman) and kalman evaluation, print both")

    parser.add_argument("--aggregate_all_envs", action="store_true",
                        help="scan models_dir and evaluate all discovered env-seed models into one csv")
    parser.add_argument("--seed_start", default=0, type=int,
                        help="minimum seed to include in aggregate_all_envs mode")
    parser.add_argument("--seed_end", default=4, type=int,
                        help="maximum seed to include in aggregate_all_envs mode")
    parser.add_argument("--out_csv", default="",
                        help="output csv path for aggregate_all_envs mode")

    parser.add_argument("--models_dir", default=r"/mnt/d/OfflineRL/TD3_BC_KAS/models",
                        help="directory that contains saved model files")
    parser.add_argument("--load_model", default="",
                        help="model name WITHOUT suffix, e.g. TD3_BC_hopper-medium-v2_0")
    parser.add_argument("--save_model", action="store_true",
                    help="save model and optimizer parameters")

    # training
    parser.add_argument("--max_timesteps", default=1_000_000, type=int)
    parser.add_argument("--eval_freq", default=5_000, type=int)
    parser.add_argument("--batch_size", default=256, type=int)

    # TD3+BC
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--alpha", default=2.5, type=float)

    # KAS
    parser.add_argument("--kalman_Q", default=1.0, type=float)
    parser.add_argument("--kalman_R", default=5e-2, type=float)
    parser.add_argument("--gate_frac", default=0.20, type=float,
                        help="gate threshold as a fraction of max_action")
    parser.add_argument("--gate_R_mult", default=20.0, type=float,
                        help="R_t = R * gate_R_mult when innovation is large")
    parser.add_argument("--blend_beta", default=0.90, type=float,
                        help="a_out = (1-beta)*a_kf + beta*a_raw")

    # dt
    parser.add_argument("--dt", default=-1.0, type=float,
                        help="override dt (seconds). If <0, auto-read from env")

    return parser.parse_args()


# ============================================================
# Aggregate evaluation
# ============================================================
def run_aggregate_all_envs(args):
    discovered = discover_models(
        models_dir=args.models_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )

    if len(discovered) == 0:
        raise RuntimeError(
            f"No matching models found in {args.models_dir} for seeds [{args.seed_start}, {args.seed_end}]. "
            f"Expected files like TD3_BC_<env>_<seed>_actor and TD3_BC_<env>_<seed>_critic."
        )

    print(f"[aggregate] Found {len(discovered)} model(s):")
    for item in discovered:
        print(f"  env={item['env']}, seed={item['seed']}, model={item['model_base']}")

    per_seed_results: List[Dict] = []

    for item in discovered:
        env_name = item["env"]
        seed = item["seed"]
        model_path = item["model_path"]

        print("\n======================================================")
        print(f"[aggregate] env={env_name}, seed={seed}")
        print(f"[aggregate] model_path={model_path}")
        print("======================================================")

        env, policy, mean, std, dt, env_dt = init_policy_for_env(args, env_name, seed)
        try:
            print(f"[dt] Using dt = {dt:.6f} (env_dt={env_dt:.6f}, override={args.dt})")
            policy.load(model_path)

            print("[baseline] evaluating without kalman...")
            res_baseline = eval_policy_with_metrics(
                policy=policy,
                env_name=env_name,
                seed=seed,
                mean=mean,
                std=std,
                dt=dt,
                apply_kalman=False,
                eval_episodes=args.eval_episodes,
            )

            print("[kalman] evaluating with kalman...")
            res_kalman = eval_policy_with_metrics(
                policy=policy,
                env_name=env_name,
                seed=seed,
                mean=mean,
                std=std,
                dt=dt,
                apply_kalman=True,
                eval_episodes=args.eval_episodes,
            )

            per_seed_results.append(
                {
                    "env": env_name,
                    "seed": seed,
                    "baseline_score": res_baseline["d4rl_score"],
                    "baseline_jerk": res_baseline["jerk"],
                    "kalman_score": res_kalman["d4rl_score"],
                    "kalman_jerk": res_kalman["jerk"],
                }
            )

        finally:
            try:
                env.close()
            except Exception:
                pass

    summary_rows = summarize_by_env(per_seed_results)

    print("\n================ Per-Env Summary ================")
    for row in summary_rows:
        print(
            f"{row['env']}: "
            f"baseline_score={row['baseline_score_mean']:.3f}±{row['baseline_score_std']:.3f}, "
            f"baseline_jerk={row['baseline_jerk_mean']:.6f}±{row['baseline_jerk_std']:.6f}, "
            f"kalman_score={row['kalman_score_mean']:.3f}±{row['kalman_score_std']:.3f}, "
            f"kalman_jerk={row['kalman_jerk_mean']:.6f}±{row['kalman_jerk_std']:.6f}"
        )
    print("=================================================")

    if args.out_csv:
        write_env_summary_csv(summary_rows, args.out_csv)


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    file_name = f"TD3_BC_{args.env}_{args.seed}"

    print("---------------------------------------")
    print(f"Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # ---------------- aggregate all envs ----------------
    if args.aggregate_all_envs:
        run_aggregate_all_envs(args)
        return

    # ---------------- single env path ----------------
    env = make_env(args.env, args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_dt = get_env_dt(env, fallback_dt=0.01)
    dt = float(args.dt) if args.dt > 0 else float(env_dt)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    gate_threshold = float(args.gate_frac) * max_action

    policy = TD3_BC_KAS.TD3_BC_KAS(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        discount=args.discount,
        tau=args.tau,
        alpha=args.alpha,
        kalman_q=args.kalman_Q,
        kalman_r=args.kalman_R,
        gate_threshold=gate_threshold,
        gate_r_mult=args.gate_R_mult,
        blend_beta=args.blend_beta,
        kalman_p_init=1.0,
    )

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
    replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))

    if args.normalize:
        mean, std = replay_buffer.normalize_states()
    else:
        mean, std = 0.0, 1.0

    print(f"[dt] Using dt = {dt:.6f} (env_dt={env_dt:.6f}, override={args.dt})")
    print(f"[models_dir] {args.models_dir}")

    if args.load_model:
        model_path = os.path.join(args.models_dir, args.load_model)
        print(f"[load] loading model from: {model_path}_actor / {model_path}_critic")
        policy.load(model_path)

    apply_kalman = bool(args.apply_kalman)
    if args.no_kalman:
        apply_kalman = False

    # ---------------- eval only ----------------
    if args.eval_only:
        if not args.load_model:
            raise ValueError("eval_only requires --load_model to be set (model name without suffix).")

        if args.ab_test:
            print("\n[Baseline]")
            res0 = eval_policy_with_metrics(
                policy, args.env, args.seed, mean, std, dt=dt,
                apply_kalman=False, eval_episodes=args.eval_episodes
            )

            print("\n[Kalman]")
            res1 = eval_policy_with_metrics(
                policy, args.env, args.seed, mean, std, dt=dt,
                apply_kalman=True, eval_episodes=args.eval_episodes
            )

            print("\n[Summary]")
            print(f"  Score: baseline={res0['d4rl_score']:.6f}, kalman={res1['d4rl_score']:.6f}")
            print(f"  Jerk:  baseline={res0['jerk']:.6f}, kalman={res1['jerk']:.6f}")
        else:
            eval_policy_with_metrics(
                policy, args.env, args.seed, mean, std, dt=dt,
                apply_kalman=apply_kalman, eval_episodes=args.eval_episodes
            )
        return

    # ---------------- training ----------------
    evaluations = []
    for t in range(int(args.max_timesteps)):
        policy.train(replay_buffer, batch_size=args.batch_size)

        if (t + 1) % int(args.eval_freq) == 0:
            print(f"Time steps: {t + 1}")

            res = eval_policy_with_metrics(
                policy, args.env, args.seed, mean, std, dt=dt,
                apply_kalman=apply_kalman, eval_episodes=args.eval_episodes
            )
            evaluations.append(res)

            np.save(f"./results/{file_name}", evaluations)

            if args.save_model:
                policy.save(f"./models/{file_name}")


if __name__ == "__main__":
    main()