import csv
import glob
import json
import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import hydra
import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from archive.baselines.IPPO.ippo_rnn_overcooked_v2 import (
    ActorCriticRNN as IPPOActorCriticRNN,
    ScannedRNN as IPPOScannedRNN,
)
from baselines.IPPO.ippo_rnn_overcooked_v2_v3 import (
    ActorCriticRNN as IPPOActorCriticRNNV3,
    ScannedRNN as IPPOScannedRNNV3,
)
from archive.baselines.TARL.mappo_rnn_overcooked_v2_v2 import (
    ActorRNN as MAPPOActorRNN,
    ScannedRNN as MAPPOScannedRNN,
)
from baselines.MAPPO.mappo_rnn_overcooked_v2_v3 import (
    ActorRNN as MAPPOActorRNNV3,
    ScannedRNN as MAPPOScannedRNNV3,
)
from baselines.TARL.talora_rnn_overcooked_v2 import (
    ActorRNN as TALORAActorRNN,
    ScannedRNN as TALORAScannedRNN,
)
from jaxmarl.viz.overcooked_v2_visualizer import OvercookedV2Visualizer
from jaxmarl.wrappers.baselines import load_params


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in name)


def _to_float(x) -> float:
    return float(np.asarray(x))


def _to_int(x) -> int:
    return int(np.asarray(x))


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _stack_state_sequence(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _resolve_metric_stats(values: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _resolve_checkpoint_path(model_cfg: Dict, default_root: str, env_name: str) -> str:
    algo = model_cfg["algo"]
    checkpoint = model_cfg.get("checkpoint", "")
    tag = model_cfg.get("checkpoint_tag", "best")

    def _choose_latest(matches: List[str]) -> str:
        if len(matches) == 0:
            return ""
        matches = sorted(matches, key=lambda p: os.path.getmtime(p))
        return matches[-1]

    if checkpoint:
        abs_path = to_absolute_path(checkpoint)
        if os.path.isdir(abs_path):
            matches = glob.glob(os.path.join(abs_path, f"*_{tag}.safetensors"))
            chosen = _choose_latest(matches)
        else:
            chosen = abs_path if os.path.exists(abs_path) else ""
        if chosen:
            return chosen
        raise FileNotFoundError(
            f"Cannot resolve checkpoint from '{checkpoint}' with tag='{tag}'."
        )

    search_root = to_absolute_path(os.path.join(default_root, algo, env_name))
    matches = glob.glob(
        os.path.join(search_root, "**", f"*_{tag}.safetensors"), recursive=True
    )
    chosen = _choose_latest(matches)
    if chosen:
        return chosen
    raise FileNotFoundError(
        f"No checkpoint found for algo={algo}, env={env_name}, tag={tag} under {search_root}."
    )


class PartnerPolicy:
    def __init__(self, model_cfg: Dict, action_dim: int, deterministic: bool = True):
        self.name = model_cfg["name"]
        self.algo = model_cfg["algo"]
        self.policy_config = model_cfg.get("policy_config", {})
        self.hidden_dim = int(self.policy_config.get("GRU_HIDDEN_DIM", 128))
        self.deterministic = deterministic
        self.model_id = model_cfg.get(
            "model_id", f"{self.algo}:{model_cfg['resolved_checkpoint']}"
        )

        self.params = load_params(model_cfg["resolved_checkpoint"])
        self.action_dim = action_dim

        if self.algo == "ippo_rnn_overcooked_v2":
            self._init_ippo()
        elif self.algo == "ippo_rnn_overcooked_v2_v3":
            self._init_ippo_v3()
        elif self.algo == "mappo_rnn_overcooked_v2_v2":
            self._init_mappo_v2()
        elif self.algo == "mappo_rnn_overcooked_v2_v3":
            self._init_mappo_v3()
        elif self.algo in ("mappo_rnn_overcooked_v2_ab_tta", "talora_rnn_overcooked_v2"):
            self._init_talora()
        else:
            supported = (
                "ippo_rnn_overcooked_v2, ippo_rnn_overcooked_v2_v3, "
                "mappo_rnn_overcooked_v2_v2, mappo_rnn_overcooked_v2_v3, "
                "mappo_rnn_overcooked_v2_ab_tta, talora_rnn_overcooked_v2"
            )
            raise ValueError(f"Unsupported algo in eval: {self.algo}. Supported: {supported}")

    def _init_ippo(self):
        self.network = IPPOActorCriticRNN(self.action_dim, config=self.policy_config)
        if self.deterministic:
            def _infer(params, hidden, obs_t, done_t):
                new_hidden, pi, _ = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.mode()
        else:
            def _infer(params, hidden, obs_t, done_t, rng):
                new_hidden, pi, _ = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.sample(seed=rng)
        self._infer = jax.jit(_infer)
        self._carry_init = IPPOScannedRNN.initialize_carry

    def _init_ippo_v3(self):
        self.network = IPPOActorCriticRNNV3(self.action_dim, config=self.policy_config)
        if self.deterministic:
            def _infer(params, hidden, obs_t, done_t):
                new_hidden, pi, _ = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.mode()
        else:
            def _infer(params, hidden, obs_t, done_t, rng):
                new_hidden, pi, _ = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.sample(seed=rng)
        self._infer = jax.jit(_infer)
        self._carry_init = IPPOScannedRNNV3.initialize_carry

    def _init_mappo_v2(self):
        self.network = MAPPOActorRNN(self.action_dim, config=self.policy_config)
        if self.deterministic:
            def _infer(params, hidden, obs_t, done_t):
                new_hidden, pi = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.mode()
        else:
            def _infer(params, hidden, obs_t, done_t, rng):
                new_hidden, pi = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.sample(seed=rng)
        self._infer = jax.jit(_infer)
        self._carry_init = MAPPOScannedRNN.initialize_carry

    def _init_mappo_v3(self):
        self.network = MAPPOActorRNNV3(self.action_dim, config=self.policy_config)
        if self.deterministic:
            def _infer(params, hidden, obs_t, done_t):
                new_hidden, pi = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.mode()
        else:
            def _infer(params, hidden, obs_t, done_t, rng):
                new_hidden, pi = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.sample(seed=rng)
        self._infer = jax.jit(_infer)
        self._carry_init = MAPPOScannedRNNV3.initialize_carry

    def _init_talora(self):
        self.network = TALORAActorRNN(self.action_dim, config=self.policy_config)
        if self.deterministic:
            def _infer(params, hidden, obs_t, done_t):
                new_hidden, pi = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.mode()
        else:
            def _infer(params, hidden, obs_t, done_t, rng):
                new_hidden, pi = self.network.apply(params, hidden, (obs_t, done_t))
                return new_hidden, pi.sample(seed=rng)
        self._infer = jax.jit(_infer)
        self._carry_init = TALORAScannedRNN.initialize_carry

    def init_hidden(self):
        return self._carry_init(1, self.hidden_dim)

    def act(self, obs_agent, done_agent, hidden, rng):
        obs_t = obs_agent[jnp.newaxis, jnp.newaxis, ...]
        done_t = jnp.asarray(done_agent, dtype=bool).reshape(1, 1)
        if self.deterministic:
            new_hidden, action = self._infer(self.params, hidden, obs_t, done_t)
            return _to_int(action.squeeze()), new_hidden, rng
        rng, action_rng = jax.random.split(rng)
        new_hidden, action = self._infer(
            self.params, hidden, obs_t, done_t, action_rng
        )
        return _to_int(action.squeeze()), new_hidden, rng


def _build_montage_gif(
    gif_paths: List[str],
    output_path: str,
    columns: int = 3,
    duration: float = 0.25,
):
    if len(gif_paths) == 0:
        return

    sequences = [imageio.mimread(path) for path in gif_paths]
    sequences = [[np.asarray(frame)[..., :3] for frame in seq] for seq in sequences]

    max_frames = max(len(seq) for seq in sequences)
    tile_h, tile_w = sequences[0][0].shape[:2]
    columns = max(1, int(columns))
    rows = int(math.ceil(len(sequences) / columns))

    montage_frames = []
    for frame_idx in range(max_frames):
        canvas = np.zeros((rows * tile_h, columns * tile_w, 3), dtype=np.uint8)
        for idx, seq in enumerate(sequences):
            row = idx // columns
            col = idx % columns
            frame = seq[min(frame_idx, len(seq) - 1)]
            canvas[
                row * tile_h : (row + 1) * tile_h,
                col * tile_w : (col + 1) * tile_w,
            ] = frame
        montage_frames.append(canvas)

    imageio.mimsave(output_path, montage_frames, format="GIF", duration=duration)


def _write_matrix_csv(
    output_path: str,
    matrix: np.ndarray,
    left_names: List[str],
    right_names: List[str],
):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left\\right"] + right_names)
        for i, left_name in enumerate(left_names):
            row = [left_name]
            for j in range(len(right_names)):
                value = matrix[i, j]
                row.append("" if np.isnan(value) else f"{value:.6f}")
            writer.writerow(row)


def evaluate_pair(
    env,
    step_fn,
    reset_fn,
    left_policy: PartnerPolicy,
    right_policy: PartnerPolicy,
    num_eval_episodes: int,
    seed: int,
    save_gif: bool,
    gif_path: str,
    gif_duration: float,
    gif_agent_view_size,
    gif_episodes_per_pair: int,
    gif_episode_pause_frames: int,
):
    viz = OvercookedV2Visualizer()
    rng = jax.random.PRNGKey(seed)

    episode_returns = []
    episode_shaped_returns = []
    episode_lengths = []
    saved_gif = ""
    gif_frames = []
    gif_episodes_to_save = max(
        0, min(int(gif_episodes_per_pair), int(num_eval_episodes))
    )
    pause_frames = max(0, int(gif_episode_pause_frames))

    for ep in range(num_eval_episodes):
        rng, reset_key = jax.random.split(rng)
        obs, state = reset_fn(reset_key)
        done_prev = {"agent_0": jnp.array(False), "agent_1": jnp.array(False)}

        left_hidden = left_policy.init_hidden()
        right_hidden = right_policy.init_hidden()

        ep_return = 0.0
        ep_shaped_return = 0.0
        save_this_episode = save_gif and ep < gif_episodes_to_save
        states = [state] if save_this_episode else None

        for t in range(env.max_steps):
            rng, k_left, k_right, k_step = jax.random.split(rng, 4)
            action_left, left_hidden, _ = left_policy.act(
                obs["agent_0"], done_prev["agent_0"], left_hidden, k_left
            )
            action_right, right_hidden, _ = right_policy.act(
                obs["agent_1"], done_prev["agent_1"], right_hidden, k_right
            )

            actions = {
                "agent_0": jnp.asarray(action_left, dtype=jnp.int32),
                "agent_1": jnp.asarray(action_right, dtype=jnp.int32),
            }

            obs, state, reward, done, info = step_fn(k_step, state, actions)

            ep_return += _to_float(reward["agent_0"])
            ep_shaped_return += _to_float(info["shaped_reward"]["agent_0"])

            done_prev = {"agent_0": done["agent_0"], "agent_1": done["agent_1"]}
            if states is not None:
                states.append(state)
            if bool(np.asarray(done["__all__"])):
                break

        episode_returns.append(ep_return)
        episode_shaped_returns.append(ep_shaped_return)
        episode_lengths.append(t + 1)

        if states is not None:
            state_seq = _stack_state_sequence(states)
            frames = viz.render_sequence(state_seq, agent_view_size=gif_agent_view_size)
            frames = np.asarray(frames).astype(np.uint8)
            gif_frames.extend(list(frames))
            if pause_frames > 0 and ep + 1 < gif_episodes_to_save:
                gif_frames.extend([frames[-1]] * pause_frames)

    if save_gif and len(gif_frames) > 0:
        imageio.mimsave(
            gif_path,
            np.asarray(gif_frames).astype(np.uint8),
            format="GIF",
            duration=gif_duration,
        )
        saved_gif = gif_path

    mean_return, std_return = _resolve_metric_stats(episode_returns)
    mean_shaped, std_shaped = _resolve_metric_stats(episode_shaped_returns)
    mean_length, _ = _resolve_metric_stats(episode_lengths)

    return {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_shaped_return": mean_shaped,
        "std_shaped_return": std_shaped,
        "mean_length": mean_length,
        "num_episodes": num_eval_episodes,
        "gif_path": saved_gif,
    }


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="eval_sp_xp_overcooked_v2",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    if env.num_agents != 2:
        raise ValueError(
            f"SP/XP evaluator currently assumes 2 agents, got num_agents={env.num_agents}"
        )

    action_dim = env.action_space(env.agents[0]).n
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    layout = config["ENV_KWARGS"]["layout"]
    output_root = to_absolute_path(config["OUTPUT_ROOT"])
    run_dir = os.path.join(output_root, f"{layout}_{now}")
    metrics_dir = os.path.join(run_dir, "metrics")
    gif_single_dir = os.path.join(run_dir, "gifs", "single")
    gif_montage_dir = os.path.join(run_dir, "gifs", "montage")
    _ensure_dir(metrics_dir)
    _ensure_dir(gif_single_dir)
    _ensure_dir(gif_montage_dir)

    left_pool = config["MODELS"]["LEFT_POOL"]
    right_pool = config["MODELS"]["RIGHT_POOL"]
    default_ckpt_root = config["DEFAULT_CHECKPOINT_ROOT"]

    for model_cfg in left_pool:
        model_cfg["resolved_checkpoint"] = _resolve_checkpoint_path(
            model_cfg, default_ckpt_root, config["ENV_NAME"]
        )
    for model_cfg in right_pool:
        model_cfg["resolved_checkpoint"] = _resolve_checkpoint_path(
            model_cfg, default_ckpt_root, config["ENV_NAME"]
        )

    left_policies = [
        PartnerPolicy(model_cfg, action_dim, deterministic=config["DETERMINISTIC"])
        for model_cfg in left_pool
    ]
    right_policies = [
        PartnerPolicy(model_cfg, action_dim, deterministic=config["DETERMINISTIC"])
        for model_cfg in right_pool
    ]

    results = []
    return_matrix = np.full((len(left_policies), len(right_policies)), np.nan)
    shaped_matrix = np.full((len(left_policies), len(right_policies)), np.nan)

    for i, left_policy in enumerate(left_policies):
        for j, right_policy in enumerate(right_policies):
            pair_name = (
                f"L{i}_{_safe_name(left_policy.name)}__R{j}_{_safe_name(right_policy.name)}"
            )
            pair_seed = int(config["SEED"]) + i * 10007 + j * 1009
            pair_is_sp = left_policy.model_id == right_policy.model_id
            gif_path = os.path.join(gif_single_dir, f"{pair_name}.gif")
            pair_metrics = evaluate_pair(
                env=env,
                step_fn=step_fn,
                reset_fn=reset_fn,
                left_policy=left_policy,
                right_policy=right_policy,
                num_eval_episodes=int(config["NUM_EVAL_EPISODES"]),
                seed=pair_seed,
                save_gif=bool(config["SAVE_SINGLE_GIFS"]),
                gif_path=gif_path,
                gif_duration=float(config["GIF_DURATION"]),
                gif_agent_view_size=config.get("GIF_AGENT_VIEW_SIZE", env.agent_view_size),
                gif_episodes_per_pair=int(config.get("GIF_EPISODES_PER_PAIR", 1)),
                gif_episode_pause_frames=int(config.get("GIF_EPISODE_PAUSE_FRAMES", 0)),
            )

            return_matrix[i, j] = pair_metrics["mean_return"]
            shaped_matrix[i, j] = pair_metrics["mean_shaped_return"]

            results.append(
                {
                    "pair_name": pair_name,
                    "left_name": left_policy.name,
                    "right_name": right_policy.name,
                    "left_algo": left_policy.algo,
                    "right_algo": right_policy.algo,
                    "left_checkpoint": left_pool[i]["resolved_checkpoint"],
                    "right_checkpoint": right_pool[j]["resolved_checkpoint"],
                    "is_sp": pair_is_sp,
                    **pair_metrics,
                }
            )

    sp_returns = [r["mean_return"] for r in results if r["is_sp"]]
    xp_returns = [r["mean_return"] for r in results if not r["is_sp"]]
    sp_shaped_returns = [r["mean_shaped_return"] for r in results if r["is_sp"]]
    xp_shaped_returns = [r["mean_shaped_return"] for r in results if not r["is_sp"]]

    sp_mean, sp_std = _resolve_metric_stats(sp_returns)
    xp_mean, xp_std = _resolve_metric_stats(xp_returns)
    sp_shaped_mean, sp_shaped_std = _resolve_metric_stats(sp_shaped_returns)
    xp_shaped_mean, xp_shaped_std = _resolve_metric_stats(xp_shaped_returns)

    summary = {
        "num_pairs": len(results),
        "num_sp_pairs": len(sp_returns),
        "num_xp_pairs": len(xp_returns),
        "sp_mean_return": sp_mean,
        "sp_std_return": sp_std,
        "xp_mean_return": xp_mean,
        "xp_std_return": xp_std,
        "sp_xp_gap_return": sp_mean - xp_mean if len(sp_returns) and len(xp_returns) else float("nan"),
        "sp_mean_shaped_return": sp_shaped_mean,
        "sp_std_shaped_return": sp_shaped_std,
        "xp_mean_shaped_return": xp_shaped_mean,
        "xp_std_shaped_return": xp_shaped_std,
        "sp_xp_gap_shaped_return": sp_shaped_mean - xp_shaped_mean if len(sp_shaped_returns) and len(xp_shaped_returns) else float("nan"),
        "env_kwargs": config["ENV_KWARGS"],
        "num_eval_episodes": int(config["NUM_EVAL_EPISODES"]),
        "deterministic": bool(config["DETERMINISTIC"]),
    }

    with open(os.path.join(metrics_dir, "pair_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(metrics_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_fields = [
        "pair_name",
        "left_name",
        "right_name",
        "left_algo",
        "right_algo",
        "left_checkpoint",
        "right_checkpoint",
        "is_sp",
        "mean_return",
        "std_return",
        "mean_shaped_return",
        "std_shaped_return",
        "mean_length",
        "num_episodes",
        "gif_path",
    ]
    with open(
        os.path.join(metrics_dir, "pair_results.csv"), "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    _write_matrix_csv(
        os.path.join(metrics_dir, "return_matrix.csv"),
        return_matrix,
        [p.name for p in left_policies],
        [p.name for p in right_policies],
    )
    _write_matrix_csv(
        os.path.join(metrics_dir, "shaped_return_matrix.csv"),
        shaped_matrix,
        [p.name for p in left_policies],
        [p.name for p in right_policies],
    )

    if config["SAVE_MONTAGE_GIF"]:
        gif_paths = [r["gif_path"] for r in results if r.get("gif_path")]
        _build_montage_gif(
            gif_paths=gif_paths,
            output_path=os.path.join(gif_montage_dir, "all_pairs_montage.gif"),
            columns=int(config["MONTAGE_COLUMNS"]),
            duration=float(config["GIF_DURATION"]),
        )

    print(f"[SP/XP] run_dir={run_dir}")
    print(
        f"[SP/XP] SP(mean/std)={summary['sp_mean_return']:.4f}/{summary['sp_std_return']:.4f}, "
        f"XP(mean/std)={summary['xp_mean_return']:.4f}/{summary['xp_std_return']:.4f}"
    )
    print(
        f"[SP/XP][shaped] SP(mean/std)={summary['sp_mean_shaped_return']:.4f}/{summary['sp_std_shaped_return']:.4f}, "
        f"XP(mean/std)={summary['xp_mean_shaped_return']:.4f}/{summary['xp_std_shaped_return']:.4f}"
    )


if __name__ == "__main__":
    main()
