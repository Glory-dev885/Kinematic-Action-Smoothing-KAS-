import argparse
import glob
import os

import flax
import gym
import d4rl
import numpy as np
import pandas as pd

import wrappers
from evaluation import evaluate
from learner import Learner


def safe_pct_change(new_value, old_value):
    """
    Percentage change:
        (new - old) / |old| * 100
    Return NaN when old_value is too close to zero.
    """
    old_value = float(old_value)
    new_value = float(new_value)

    if abs(old_value) < 1e-12:
        return np.nan
    return (new_value - old_value) / abs(old_value) * 100.0


def get_env_dt(env, fallback_dt=0.01):
    """
    Try to read dt from MuJoCo env.
    Fall back to env.dt or fallback_dt.
    """
    candidates = [env]
    cur = env
    for _ in range(10):
        if hasattr(cur, "env"):
            cur = cur.env
            candidates.append(cur)
        else:
            break

    for e in candidates:
        try:
            return float(e.sim.model.opt.timestep)
        except Exception:
            pass

    for e in candidates:
        try:
            if hasattr(e, "dt"):
                return float(e.dt)
        except Exception:
            pass

    return float(fallback_dt)


def make_env(env_name: str, seed: int):
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def build_agent(env, seed, args):
    kwargs = {
        "actor_lr": 3e-4,
        "value_lr": 3e-4,
        "critic_lr": 3e-4,
        "hidden_dims": (256, 256),
        "discount": 0.99,
        "tau": 0.005,
        "expectile": 0.8,
        "temperature": 0.1,
        "dropout_rate": None,
        "max_steps": 1000000,
        "opt_decay_schedule": "cosine",
        "kalman_q": args.kalman_q,
        "kalman_r": args.kalman_r,
        "gate_frac": args.gate_frac,
        "gate_r_mult": args.gate_r_mult,
        "blend_beta": args.blend_beta,
        "kalman_p_init": args.kalman_p_init,
    }

    agent = Learner(
        seed,
        env.observation_space.sample()[np.newaxis],
        env.action_space.sample()[np.newaxis],
        **kwargs,
    )
    return agent


def load_flax_params(model, path):
    with open(path, "rb") as f:
        params = flax.serialization.from_bytes(model.params, f.read())
    return model.replace(params=params)


def load_iql_models(agent, models_dir, env_name, seed, prefix="IQL"):
    actor_path = os.path.join(models_dir, f"{prefix}_{env_name}_{seed}_actor.flax")
    critic_path = os.path.join(models_dir, f"{prefix}_{env_name}_{seed}_critic.flax")
    value_path = os.path.join(models_dir, f"{prefix}_{env_name}_{seed}_value.flax")

    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"actor model not found: {actor_path}")
    if not os.path.exists(critic_path):
        raise FileNotFoundError(f"critic model not found: {critic_path}")
    if not os.path.exists(value_path):
        raise FileNotFoundError(f"value model not found: {value_path}")

    agent.actor = load_flax_params(agent.actor, actor_path)
    agent.critic = load_flax_params(agent.critic, critic_path)
    agent.value = load_flax_params(agent.value, value_path)
    agent.target_critic = agent.critic

    return agent


def print_single_compare_table(env_name, seed, dt, result_row):
    print("\n" + "=" * 92)
    print(f"[IQL + KAS Comparison] env={env_name}, seed={seed}, dt={dt:.6f}")
    print("=" * 92)

    print(f"{'Method':<12} {'Return':>14} {'Avg Jerk':>16}")
    print("-" * 92)
    print(
        f"{'Baseline':<12} "
        f"{result_row['return_base']:>14.3f} "
        f"{result_row['jerk_base']:>16.6f}"
    )
    print(
        f"{'KAS':<12} "
        f"{result_row['return_kalman']:>14.3f} "
        f"{result_row['jerk_kalman']:>16.6f}"
    )
    print("-" * 92)
    print(
        f"{'Delta':<12} "
        f"{result_row['return_delta']:>14.3f} "
        f"{result_row['jerk_delta']:>16.6f}"
    )
    print(
        f"{'Change (%)':<12} "
        f"{result_row['return_pct_change']:>13.2f}% "
        f"{result_row['jerk_pct_change']:>15.2f}%"
    )
    print("=" * 92)


def parse_model_name(filename, prefix):
    """
    parse:
        IQL_halfcheetah-expert-v2_0_actor.flax
    -> env='halfcheetah-expert-v2', seed=0
    """
    base = os.path.basename(filename)
    if not base.endswith("_actor.flax"):
        return None

    stem = base[:-len("_actor.flax")]
    parts = stem.split("_")

    prefix_parts = prefix.split("_")
    if parts[:len(prefix_parts)] != prefix_parts:
        return None

    if len(parts) < len(prefix_parts) + 2:
        return None

    try:
        seed = int(parts[-1])
    except Exception:
        return None

    env_name = "_".join(parts[len(prefix_parts):-1])
    if env_name == "":
        return None

    return env_name, seed


def detect_env_seed_models(models_dir, prefix, seed_start, seed_end):
    env2seeds = {}
    actor_files = glob.glob(os.path.join(models_dir, "*_actor.flax"))

    for af in actor_files:
        parsed = parse_model_name(af, prefix)
        if parsed is None:
            continue

        env_name, seed = parsed
        if seed < seed_start or seed > seed_end:
            continue

        critic_path = os.path.join(models_dir, f"{prefix}_{env_name}_{seed}_critic.flax")
        value_path = os.path.join(models_dir, f"{prefix}_{env_name}_{seed}_value.flax")
        if not os.path.exists(critic_path) or not os.path.exists(value_path):
            continue

        env2seeds.setdefault(env_name, set()).add(seed)

    return {k: sorted(list(v)) for k, v in env2seeds.items() if len(v) > 0}


def build_result_row(env_name, seed, dt, base_stats, kalman_stats):
    row = {
        "env": env_name,
        "seed": seed,
        "dt": float(dt),

        "return_base": float(base_stats["return"]),
        "return_kalman": float(kalman_stats["return"]),
        "return_delta": float(kalman_stats["return"] - base_stats["return"]),
        "return_pct_change": float(safe_pct_change(kalman_stats["return"], base_stats["return"])),

        "jerk_base": float(base_stats["jerk"]),
        "jerk_kalman": float(kalman_stats["jerk"]),
        "jerk_delta": float(kalman_stats["jerk"] - base_stats["jerk"]),
        "jerk_pct_change": float(safe_pct_change(kalman_stats["jerk"], base_stats["jerk"])),
    }

    return row


def save_single_csv(row, output_dir, env_name, seed):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame([row])
    out_path = os.path.join(output_dir, f"{env_name}_seed{seed}_comparison.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[csv] saved single-seed result to: {out_path}")


def aggregate_rows(rows, env_name):
    """
    Aggregation logic:
    1. base / kas / delta keep mean and std
    2. percentage change does NOT compute std
    3. percentage change is computed from aggregated means
    """
    df = pd.DataFrame(rows)

    summary = {
        "env": env_name,
        "num_seeds": int(len(df)),
        "seed_list": ",".join(str(int(s)) for s in df["seed"].tolist()),
    }

    stats_with_std = [
        "dt",
        "return_base",
        "return_kalman",
        "return_delta",
        "jerk_base",
        "jerk_kalman",
        "jerk_delta",
    ]

    for col in stats_with_std:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=0))

    summary["return_pct_change"] = float(
        safe_pct_change(summary["return_kalman_mean"], summary["return_base_mean"])
    )
    summary["jerk_pct_change"] = float(
        safe_pct_change(summary["jerk_kalman_mean"], summary["jerk_base_mean"])
    )

    return pd.DataFrame([summary])


def print_aggregate_table(env_name, agg_df):
    row = agg_df.iloc[0].to_dict()

    print("\n" + "#" * 128)
    print(f"# Aggregate Results over Seeds: {env_name}")
    print("#" * 128)

    print(
        f"{'Metric':<18} "
        f"{'Base (mean±std)':<28} "
        f"{'KAS (mean±std)':<28} "
        f"{'Change (%)':<16}"
    )
    print("-" * 128)

    return_base_str = f"{row['return_base_mean']:.3f} ± {row['return_base_std']:.3f}"
    return_kalman_str = f"{row['return_kalman_mean']:.3f} ± {row['return_kalman_std']:.3f}"
    return_pct_str = f"{row['return_pct_change']:.2f}%"

    print(
        f"{'Return':<18} "
        f"{return_base_str:<28} "
        f"{return_kalman_str:<28} "
        f"{return_pct_str:<16}"
    )

    jerk_base_str = f"{row['jerk_base_mean']:.6f} ± {row['jerk_base_std']:.6f}"
    jerk_kalman_str = f"{row['jerk_kalman_mean']:.6f} ± {row['jerk_kalman_std']:.6f}"
    jerk_pct_str = f"{row['jerk_pct_change']:.2f}%"

    print(
        f"{'Avg Jerk':<18} "
        f"{jerk_base_str:<28} "
        f"{jerk_kalman_str:<28} "
        f"{jerk_pct_str:<16}"
    )

    print("#" * 128)


def print_combined_aggregate_table(combined_df):
    print("\n" + "#" * 180)
    print("# Combined Aggregate Results over All Environments")
    print("#" * 180)

    print(
        f"{'Environment':<27} "
        f"{'Return (Base)':<18} "
        f"{'Return (KAS)':<18} "
        f"{'Return Change (%)':<18} "
        f"{'Jerk (Base)':<20} "
        f"{'Jerk (KAS)':<20} "
        f"{'Jerk Change (%)':<18}"
    )
    print("-" * 180)

    for _, row in combined_df.iterrows():
        env_name = row["env"]
        return_base = f"{row['return_base_mean']:.3f}±{row['return_base_std']:.3f}"
        return_kalman = f"{row['return_kalman_mean']:.3f}±{row['return_kalman_std']:.3f}"
        return_pct = f"{row['return_pct_change']:.2f}%"

        jerk_base = f"{row['jerk_base_mean']:.6f}±{row['jerk_base_std']:.6f}"
        jerk_kalman = f"{row['jerk_kalman_mean']:.6f}±{row['jerk_kalman_std']:.6f}"
        jerk_pct = f"{row['jerk_pct_change']:.2f}%"

        print(
            f"{env_name:<27} "
            f"{return_base:<18} "
            f"{return_kalman:<18} "
            f"{return_pct:<18} "
            f"{jerk_base:<20} "
            f"{jerk_kalman:<20} "
            f"{jerk_pct:<18}"
        )

    print("#" * 180)


def save_aggregate_csv(rows, output_dir, env_name):
    os.makedirs(output_dir, exist_ok=True)

    detail_df = pd.DataFrame(rows)
    detail_path = os.path.join(output_dir, f"{env_name}_all_seeds_detail.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    agg_df = aggregate_rows(rows, env_name)
    agg_path = os.path.join(output_dir, f"{env_name}_aggregate.csv")
    agg_df.to_csv(agg_path, index=False, encoding="utf-8-sig")

    print(f"[csv] saved multi-seed detail to: {detail_path}")
    print(f"[csv] saved multi-seed aggregate to: {agg_path}")

    return agg_df


def run_one_seed(args, env_name, seed):
    env = make_env(env_name, seed)
    env_dt = get_env_dt(env, fallback_dt=0.01)
    dt = float(args.dt) if args.dt > 0 else float(env_dt)

    agent = build_agent(env, seed, args)
    agent = load_iql_models(
        agent=agent,
        models_dir=args.models_dir,
        env_name=env_name,
        seed=seed,
        prefix=args.model_prefix,
    )

    base_stats = evaluate(
        agent,
        env,
        args.eval_episodes,
        apply_kalman=False,
        dt=dt,
    )

    kalman_stats = evaluate(
        agent,
        env,
        args.eval_episodes,
        apply_kalman=True,
        dt=dt,
    )

    row = build_result_row(env_name, seed, dt, base_stats, kalman_stats)
    print_single_compare_table(env_name, seed, dt, row)
    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="halfcheetah-expert-v2")
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument("--model_prefix", type=str, default="IQL")
    parser.add_argument("--output_dir", type=str, default="./experiments")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval_episodes", type=int, default=10)

    parser.add_argument("--aggregate_seeds", action="store_true")
    parser.add_argument("--aggregate_all_envs", action="store_true")
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--seed_end", type=int, default=4)

    parser.add_argument(
        "--dt",
        type=float,
        default=-1.0,
        help="If >0, override env dt. Otherwise auto-read from env.",
    )

    parser.add_argument("--kalman_q", type=float, default=1.0)
    parser.add_argument("--kalman_r", type=float, default=5e-2)
    parser.add_argument("--gate_frac", type=float, default=5e-2)
    parser.add_argument("--gate_r_mult", type=float, default=20.0)
    parser.add_argument("--blend_beta", type=float, default=0.90)
    parser.add_argument("--kalman_p_init", type=float, default=1.0)

    args = parser.parse_args()

    if args.aggregate_all_envs:
        env2seeds = detect_env_seed_models(
            models_dir=args.models_dir,
            prefix=args.model_prefix,
            seed_start=args.seed_start,
            seed_end=args.seed_end,
        )
        if len(env2seeds) == 0:
            raise RuntimeError(
                f"No matching flax models found in {args.models_dir}. "
                f"Expected pattern: {args.model_prefix}_{{env}}_{{seed}}_actor.flax"
            )

        all_agg_dfs = []
        for env_name in sorted(env2seeds.keys()):
            rows = []
            for seed in env2seeds[env_name]:
                rows.append(run_one_seed(args, env_name, seed))
            agg_df = aggregate_rows(rows, env_name)
            all_agg_dfs.append(agg_df)

        if all_agg_dfs:
            os.makedirs(args.output_dir, exist_ok=True)
            combined_df = pd.concat(all_agg_dfs, ignore_index=True)
            combined_path = os.path.join(args.output_dir, "comparison.csv")
            combined_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
            print(f"[csv] saved all envs aggregate to: {combined_path}")
            print_combined_aggregate_table(combined_df)
        return

    if args.aggregate_seeds:
        rows = []
        for seed in range(args.seed_start, args.seed_end + 1):
            rows.append(run_one_seed(args, args.env_name, seed))
        agg_df = save_aggregate_csv(rows, args.output_dir, args.env_name)
        print_aggregate_table(args.env_name, agg_df)
        return

    row = run_one_seed(args, args.env_name, args.seed)
    save_single_csv(row, args.output_dir, args.env_name, args.seed)


if __name__ == "__main__":
    main()