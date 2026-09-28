"""
train.py

1. Trains a tabular RL agent on the IoV Digital Twin synchronization MDP.
2. Evaluates the trained agent against two baselines (Periodic, Threshold).
3. Saves metrics (CSV), charts (SVG), and a browser dashboard to ./results/.

Run:
    python3 -m src.train
"""

import argparse
import json
import numpy as np
import os
import pandas as pd

from .environment import IoVDigitalTwinEnv, TRANSMIT, DELAY, DROP, ACTION_NAMES
from .q_learning_agent import DoubleQLearningAgent, QLearningAgent
from .baselines import PeriodicPolicy, ThresholdPolicy

RESULTS_DIR = "results"
N_TRAIN_EPISODES = 1500
N_EVAL_EPISODES = 50
EPISODE_LENGTH = 200
SEED = 42
DEFAULT_DATA_PATH = "data/synthetic_data.csv"


def _points(values, x0, y0, width, height):
    values = list(values)
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    span = hi - lo or 1.0
    last = max(len(values) - 1, 1)
    coords = []
    for i, value in enumerate(values):
        x = x0 + width * i / last
        y = y0 + height - height * (value - lo) / span
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def write_training_svg(rewards, avg_errors, bandwidth, path):
    series = [
        ("Training Reward", rewards, "#0f7b6c"),
        ("Avg DT Sync Error", avg_errors, "#b44b35"),
        ("Bandwidth Used", bandwidth, "#3f5f8f"),
    ]
    panels = []
    for i, (title, values, color) in enumerate(series):
        x0 = 55 + i * 330
        panels.append(f'<text x="{x0}" y="30" font-size="16" font-weight="700">{title}</text>')
        panels.append(f'<rect x="{x0}" y="45" width="285" height="190" fill="#fff" stroke="#d7dde3"/>')
        panels.append(f'<polyline points="{_points(values, x0 + 10, 55, 265, 165)}" fill="none" stroke="{color}" stroke-width="2"/>')
        panels.append(f'<text x="{x0}" y="260" font-size="12" fill="#5c6773">episodes: {len(values)}</text>')
        panels.append(f'<text x="{x0 + 190}" y="260" font-size="12" fill="#5c6773">last: {values[-1]:.2f}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="285" viewBox="0 0 1040 285">{"".join(panels)}</svg>'
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def write_comparison_svg(summary, path):
    policies = list(summary["Policy"])
    metrics = [
        ("Avg DT Error", "#b44b35"),
        ("Avg Bandwidth Used", "#0f7b6c"),
    ]
    panels = []
    for panel_idx, (metric, color) in enumerate(metrics):
        x0 = 65 + panel_idx * 500
        values = list(summary[metric])
        max_value = max(values) or 1.0
        panels.append(f'<text x="{x0}" y="30" font-size="16" font-weight="700">{metric}</text>')
        panels.append(f'<rect x="{x0}" y="45" width="400" height="240" fill="#fff" stroke="#d7dde3"/>')
        for i, (policy, value) in enumerate(zip(policies, values)):
            bar_h = 185 * value / max_value
            x = x0 + 35 + i * 120
            y = 260 - bar_h
            panels.append(f'<rect x="{x}" y="{y:.1f}" width="55" height="{bar_h:.1f}" fill="{color}"/>')
            panels.append(f'<text x="{x}" y="{y - 6:.1f}" font-size="11">{value:.2f}</text>')
            panels.append(f'<text x="{x - 18}" y="278" font-size="10" fill="#5c6773">{policy[:18]}</text>')
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="315" viewBox="0 0 1040 315">{"".join(panels)}</svg>'
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def build_env_kwargs(data_path):
    if data_path and os.path.exists(data_path):
        return {"trace_path": data_path}
    return {}


def infer_state_ranges(data_path):
    if not data_path or not os.path.exists(data_path):
        return ((0, 35), (3, 30), (1, 40), (0, 25))

    df = pd.read_csv(data_path)
    mapping = {
        "speed": "Speed_mps",
        "bandwidth": "Bandwidth_Mbps",
        "delay": "Latency_ms",
        "dt_error": "DT_Error_m",
    }
    ranges = []
    for column in mapping.values():
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        lo = float(values.min())
        hi = float(values.max())
        pad = max((hi - lo) * 0.15, 1.0)
        ranges.append((max(0.0, lo - pad), hi + pad))
    return tuple(ranges)


def make_agent(algorithm, state_ranges):
    agent_cls = DoubleQLearningAgent if algorithm == "double-q" else QLearningAgent
    return agent_cls(
        bins=(7, 7, 7, 9),
        state_ranges=state_ranges,
        alpha=0.12,
        gamma=0.96,
        epsilon_decay=0.992,
        seed=SEED,
    )


def train_agent(algorithm, env_kwargs, state_ranges, n_train_episodes):
    env = IoVDigitalTwinEnv(episode_length=EPISODE_LENGTH, seed=SEED, **env_kwargs)
    agent = make_agent(algorithm, state_ranges)

    episode_rewards = []
    episode_avg_errors = []
    episode_bandwidth = []

    for ep in range(n_train_episodes):
        state = env.reset()
        total_reward = 0.0
        errors = []
        bandwidth_used = 0.0

        for _ in range(EPISODE_LENGTH):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            errors.append(info["dt_error"])
            bandwidth_used += info["bandwidth_used"]

            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_avg_errors.append(np.mean(errors))
        episode_bandwidth.append(bandwidth_used)

        if (ep + 1) % 100 == 0:
            print(
                f"Episode {ep+1}/{n_train_episodes} | "
                f"Reward: {total_reward:.1f} | "
                f"AvgErr: {np.mean(errors):.2f} | "
                f"BW used: {bandwidth_used:.1f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    return agent, episode_rewards, episode_avg_errors, episode_bandwidth


def run_policy(policy_fn, env_kwargs, reset_fn=None, n_episodes=N_EVAL_EPISODES, seed=SEED + 1):
    """policy_fn(state) -> action. Returns per-episode metrics."""
    env = IoVDigitalTwinEnv(episode_length=EPISODE_LENGTH, seed=seed, **env_kwargs)
    all_avg_error, all_bandwidth, all_reward = [], [], []
    action_counts = {TRANSMIT: 0, DELAY: 0, DROP: 0}

    for _ in range(n_episodes):
        if reset_fn:
            reset_fn()
        state = env.reset()
        errors, bw_used, total_reward = [], 0.0, 0.0

        for _ in range(EPISODE_LENGTH):
            action = policy_fn(state)
            action_counts[action] += 1
            next_state, reward, done, info = env.step(action)
            state = next_state

            errors.append(info["dt_error"])
            bw_used += info["bandwidth_used"]
            total_reward += reward
            if done:
                break

        all_avg_error.append(np.mean(errors))
        all_bandwidth.append(bw_used)
        all_reward.append(total_reward)

    total_actions = sum(action_counts.values())
    action_dist = {ACTION_NAMES[k]: v / total_actions for k, v in action_counts.items()}

    return {
        "avg_dt_error": float(np.mean(all_avg_error)),
        "avg_bandwidth_used": float(np.mean(all_bandwidth)),
        "avg_reward": float(np.mean(all_reward)),
        "action_distribution": action_dist,
    }


def write_dashboard(summary, action_dist_df, metadata):
    table_html = summary.to_html(index=False, classes="metrics", float_format=lambda x: f"{x:.3f}")
    action_html = action_dist_df.reset_index(names="Policy").to_html(index=False, classes="metrics", float_format=lambda x: f"{x:.3f}")
    best_error = summary.sort_values("Avg DT Error").iloc[0]["Policy"]
    best_reward = summary.sort_values("Avg Episode Reward", ascending=False).iloc[0]["Policy"]

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IoV Digital Twin RL Dashboard</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17212b;
      --muted: #5c6773;
      --line: #d7dde3;
      --panel: #f7f9fb;
      --accent: #0f7b6c;
      --accent-2: #b44b35;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 56px) 18px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0; font-size: clamp(28px, 4vw, 46px); letter-spacing: 0; }}
    main {{ padding: 24px clamp(18px, 4vw, 56px) 42px; }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
      margin: 18px 0 24px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--panel);
    }}
    .label {{ color: var(--muted); font-size: 13px; margin-bottom: 6px; }}
    .value {{ font-size: 20px; font-weight: 700; }}
    section {{ margin-top: 28px; }}
    h2 {{ font-size: 21px; margin: 0 0 12px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    img {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }}
    table.metrics {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      font-size: 14px;
    }}
    .metrics th, .metrics td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    .metrics th {{ background: var(--panel); }}
    .metrics tr:last-child td {{ border-bottom: 0; }}
  </style>
</head>
<body>
  <header>
    <h1>IoV Digital Twin RL Dashboard</h1>
  </header>
  <main>
    <div class="meta">
      <div class="stat"><div class="label">Data source</div><div class="value">{metadata["data_source"]}</div></div>
      <div class="stat"><div class="label">Algorithm</div><div class="value">{metadata["algorithm"]}</div></div>
      <div class="stat"><div class="label">Lowest DT error</div><div class="value">{best_error}</div></div>
      <div class="stat"><div class="label">Best reward</div><div class="value">{best_reward}</div></div>
    </div>

    <section>
      <h2>Policy Comparison</h2>
      {table_html}
    </section>

    <section class="grid">
      <div>
        <h2>Training Curves</h2>
        <img src="training_curves.svg" alt="Training reward, DT error, and bandwidth curves">
      </div>
      <div>
        <h2>Evaluation Chart</h2>
        <img src="policy_comparison.svg" alt="Average DT error and bandwidth comparison">
      </div>
    </section>

    <section>
      <h2>Action Distribution</h2>
      {action_html}
    </section>
  </main>
</body>
</html>
"""
    with open(f"{RESULTS_DIR}/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Train IoV DT synchronization policies.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="CSV trace to train/evaluate on.")
    parser.add_argument("--algorithm", choices=["double-q", "q-learning"], default="double-q")
    parser.add_argument("--episodes", type=int, default=N_TRAIN_EPISODES)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    env_kwargs = build_env_kwargs(args.data)
    state_ranges = infer_state_ranges(args.data if env_kwargs else None)
    data_source = args.data if env_kwargs else "stochastic simulator"

    print("=" * 60)
    print(f"Training {args.algorithm} agent...")
    print(f"Data source: {data_source}")
    print("=" * 60)
    agent, rewards, avg_errors, bandwidth = train_agent(
        args.algorithm,
        env_kwargs,
        state_ranges,
        args.episodes,
    )

    write_training_svg(rewards, avg_errors, bandwidth, f"{RESULTS_DIR}/training_curves.svg")

    # -------------------- Evaluation --------------------
    print("\n" + "=" * 60)
    print("Evaluating RL agent vs baselines...")
    print("=" * 60)

    rl_result = run_policy(lambda s: agent.select_action(s, greedy=True), env_kwargs)

    periodic = PeriodicPolicy(period=5)
    periodic_result = run_policy(periodic.select_action, env_kwargs, reset_fn=periodic.reset)

    threshold = ThresholdPolicy(error_threshold=3.0)
    threshold_result = run_policy(threshold.select_action, env_kwargs, reset_fn=threshold.reset)

    summary = pd.DataFrame(
        {
            "Policy": [f"RL ({args.algorithm})", "Periodic (every 5 steps)", "Threshold (err>=3.0)"],
            "Avg DT Error": [
                rl_result["avg_dt_error"],
                periodic_result["avg_dt_error"],
                threshold_result["avg_dt_error"],
            ],
            "Avg Bandwidth Used": [
                rl_result["avg_bandwidth_used"],
                periodic_result["avg_bandwidth_used"],
                threshold_result["avg_bandwidth_used"],
            ],
            "Avg Episode Reward": [
                rl_result["avg_reward"],
                periodic_result["avg_reward"],
                threshold_result["avg_reward"],
            ],
        }
    )
    print("\n", summary.to_string(index=False))
    summary.to_csv(f"{RESULTS_DIR}/policy_comparison.csv", index=False)

    # Bandwidth savings relative to periodic baseline (closest to "sending everything")
    bw_reduction_vs_periodic = (
        (periodic_result["avg_bandwidth_used"] - rl_result["avg_bandwidth_used"])
        / periodic_result["avg_bandwidth_used"]
        * 100
    )
    err_change_vs_periodic = (
        (rl_result["avg_dt_error"] - periodic_result["avg_dt_error"])
        / periodic_result["avg_dt_error"]
        * 100
    )
    print(f"\nRL bandwidth reduction vs periodic baseline: {bw_reduction_vs_periodic:.1f}%")
    print(f"RL DT error change vs periodic baseline: {err_change_vs_periodic:+.1f}%")

    write_comparison_svg(summary, f"{RESULTS_DIR}/policy_comparison.svg")

    # Action distribution for RL agent
    action_dist_df = pd.DataFrame(
        [rl_result["action_distribution"], periodic_result["action_distribution"], threshold_result["action_distribution"]],
        index=[f"RL ({args.algorithm})", "Periodic", "Threshold"],
    )
    action_dist_df.to_csv(f"{RESULTS_DIR}/action_distribution.csv")
    print("\nAction distribution (fraction of steps):\n", action_dist_df)

    # Save Q-table for reuse
    np.save(f"{RESULTS_DIR}/q_table.npy", agent.q_table)
    metadata = {
        "algorithm": args.algorithm,
        "data_source": data_source,
        "episodes": args.episodes,
        "state_ranges": state_ranges,
    }
    with open(f"{RESULTS_DIR}/run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    write_dashboard(summary, action_dist_df, metadata)

    print(f"\nAll results saved to ./{RESULTS_DIR}/")
    print(f"Open ./{RESULTS_DIR}/dashboard.html for the UI.")


if __name__ == "__main__":
    main()
