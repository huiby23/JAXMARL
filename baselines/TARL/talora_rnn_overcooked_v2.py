import datetime
import functools
import os
from typing import Any, Callable, Dict, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import numpy as np
import optax
import wandb
from flax import traverse_util
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

try:
    from tensorboardX import SummaryWriter
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        SummaryWriter = None

from jaxmarl.wrappers.baselines import JaxMARLWrapper, OvercookedV2LogWrapper, save_params


class OvercookedV2WorldStateWrapper(JaxMARLWrapper):
    """Attachs a critic world_state for MAPPO under two modes.

    - obs_concat: concat all agents' observations along channel dim.
    - env_state: encode underlying environment state as a global vector.
    """

    def __init__(self, env, source: str = "env_state"):
        super().__init__(env)
        if source not in ("env_state", "obs_concat"):
            raise ValueError(f"Unsupported WORLD_STATE_SOURCE: {source}")
        self.source = source

        obs_shape = tuple(self._env.observation_space().shape)
        self._obs_concat_shape = (*obs_shape[:-1], obs_shape[-1] * self._env.num_agents)

        # grid + (pos, dir_one_hot, inventory) per agent + (recipe, time, terminal, delivery_flag)
        self._env_state_size = self._env.height * self._env.width * 3 + self._env.num_agents * 7 + 4

        if self.source == "obs_concat":
            self._world_state_shape = self._obs_concat_shape
            self._world_state_size = int(np.prod(self._world_state_shape))
        else:
            self._world_state_shape = (self._env_state_size,)
            self._world_state_size = int(self._env_state_size)

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs, env_state)
        return obs, env_state

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs, env_state)
        return obs, env_state, reward, done, info

    @functools.partial(jax.jit, static_argnums=0)
    def world_state(self, obs, env_state):
        if self.source == "obs_concat":
            all_obs = jnp.stack([obs[a] for a in self._env.agents], axis=0)
            all_obs = jnp.transpose(all_obs, (1, 2, 0, 3))
            world_state = all_obs.reshape(all_obs.shape[0], all_obs.shape[1], -1)
            return jnp.broadcast_to(world_state[None, ...], (self._env.num_agents, *world_state.shape))

        grid = env_state.grid.astype(jnp.float32).reshape(-1) / 255.0

        pos = env_state.agents.pos.to_array().astype(jnp.float32)
        pos_scale = jnp.array(
            [max(self._env.width - 1, 1), max(self._env.height - 1, 1)],
            dtype=jnp.float32,
        )
        pos = (pos / pos_scale).reshape(-1)

        dir_one_hot = jax.nn.one_hot(env_state.agents.dir, 4, dtype=jnp.float32).reshape(-1)
        inventory = env_state.agents.inventory.astype(jnp.float32).reshape(-1) / 255.0

        recipe = jnp.asarray(env_state.recipe, dtype=jnp.float32).reshape((1,)) / 255.0
        time = jnp.asarray(env_state.time, dtype=jnp.float32).reshape((1,)) / float(max(self._env.max_steps, 1))
        terminal = jnp.asarray(env_state.terminal, dtype=jnp.float32).reshape((1,))
        delivery = jnp.asarray(env_state.new_correct_delivery, dtype=jnp.float32).reshape((1,))

        world_state = jnp.concatenate(
            [grid, pos, dir_one_hot, inventory, recipe, time, terminal, delivery],
            axis=0,
        )
        return jnp.broadcast_to(world_state[None, :], (self._env.num_agents, world_state.shape[0]))

    def world_state_size(self):
        return self._world_state_size

    def world_state_shape(self):
        return self._world_state_shape


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])
        rnn_state = jnp.where(resets[:, np.newaxis], new_carry, rnn_state)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


def _adapter_targets(config: Dict) -> set:
    raw = config.get("ADAPTER_TARGETS", config.get("LORA_TARGETS", []))
    if isinstance(raw, str):
        return {x.strip() for x in raw.split(",") if x.strip()}
    if isinstance(raw, (tuple, list)):
        return {str(x).strip() for x in raw if str(x).strip()}
    return set()


def _adapter_enabled(config: Dict) -> bool:
    return bool(config.get("ADAPTER_ENABLED", config.get("LORA_ENABLED", False)))


def _adapter_type(config: Dict) -> str:
    return str(config.get("ADAPTER_TYPE", "lora")).strip().lower()


def _adapter_rank(config: Dict) -> int:
    return int(config.get("ADAPTER_RANK", config.get("LORA_RANK", 0)))


def _adapter_alpha(config: Dict) -> float:
    return float(config.get("ADAPTER_ALPHA", config.get("LORA_ALPHA", 1.0)))


def _use_adapter_layer(config: Dict, layer_name: str) -> bool:
    if not _adapter_enabled(config):
        return False
    rank = _adapter_rank(config)
    if rank <= 0:
        return False
    targets = _adapter_targets(config)
    return "all" in targets or layer_name in targets


def _train_adapter_a_flag(config: Dict, phase: str, default: bool) -> bool:
    return bool(
        config.get(
            f"{phase}_TRAIN_ADAPTER_A",
            config.get(f"{phase}_TRAIN_LORA_A", default),
        )
    )


def _train_adapter_b_flag(config: Dict, phase: str, default: bool) -> bool:
    return bool(
        config.get(
            f"{phase}_TRAIN_ADAPTER_B",
            config.get(f"{phase}_TRAIN_LORA_B", default),
        )
    )


class LoRADense(nn.Module):
    features: int
    rank: int
    alpha: float
    use_bias: bool = True
    kernel_init: Callable[..., Any] = orthogonal(1.0)
    bias_init: Callable[..., Any] = constant(0.0)
    lora_a_init: Callable[..., Any] = nn.initializers.normal(stddev=0.01)
    lora_b_init: Callable[..., Any] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        kernel = self.param("kernel", self.kernel_init, (in_features, self.features))
        y = jnp.matmul(x, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            y = y + bias

        if self.rank > 0:
            lora_a = self.param("lora_A", self.lora_a_init, (in_features, self.rank))
            lora_b = self.param("lora_B", self.lora_b_init, (self.rank, self.features))
            scale = self.alpha / float(self.rank)
            y = y + scale * jnp.matmul(jnp.matmul(x, lora_a), lora_b)
        return y


class ResidualAdapterDense(nn.Module):
    features: int
    rank: int
    alpha: float
    activation: Callable[..., Any] = nn.relu
    use_bias: bool = True
    kernel_init: Callable[..., Any] = orthogonal(1.0)
    bias_init: Callable[..., Any] = constant(0.0)
    adapter_a_init: Callable[..., Any] = nn.initializers.normal(stddev=0.01)
    adapter_b_init: Callable[..., Any] = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        in_features = x.shape[-1]
        kernel = self.param("kernel", self.kernel_init, (in_features, self.features))
        y = jnp.matmul(x, kernel)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (self.features,))
            y = y + bias

        if self.rank > 0:
            adapter_a = self.param(
                "adapter_A", self.adapter_a_init, (in_features, self.rank)
            )
            adapter_b = self.param(
                "adapter_B", self.adapter_b_init, (self.rank, self.features)
            )
            scale = self.alpha / float(self.rank)
            residual = self.activation(jnp.matmul(x, adapter_a))
            y = y + scale * jnp.matmul(residual, adapter_b)
        return y


def _dense(
    x,
    config: Dict,
    layer_name: str,
    features: int,
    *,
    kernel_init=orthogonal(1.0),
    bias_init=constant(0.0),
    use_bias: bool = True,
):
    if _use_adapter_layer(config, layer_name):
        adapter_type = _adapter_type(config)
        common_kwargs = dict(
            features=features,
            rank=_adapter_rank(config),
            alpha=_adapter_alpha(config),
            use_bias=use_bias,
            kernel_init=kernel_init,
            bias_init=bias_init,
            name=layer_name,
        )
        if adapter_type == "lora":
            return LoRADense(**common_kwargs)(x)
        if adapter_type == "residual":
            activation = nn.relu if config["ACTIVATION"] == "relu" else nn.tanh
            return ResidualAdapterDense(
                activation=activation,
                **common_kwargs,
            )(x)
        raise ValueError(f"Unsupported ADAPTER_TYPE: {adapter_type}")
    return nn.Dense(
        features=features,
        kernel_init=kernel_init,
        bias_init=bias_init,
        use_bias=use_bias,
        name=layer_name,
    )(x)


class CNN(nn.Module):
    config: Dict
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))
        x = _dense(
            x,
            self.config,
            "cnn_out",
            self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        x = self.activation(x)
        return x


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        embed_model = CNN(
            config=self.config,
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        actor_mean = _dense(
            embedding,
            self.config,
            "actor_fc",
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )
        actor_mean = nn.relu(actor_mean)
        action_logits = _dense(
            actor_mean,
            self.config,
            "actor_logits",
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )
        return hidden, distrax.Categorical(logits=action_logits)


class CriticCNNRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        activation = nn.relu if self.config["ACTIVATION"] == "relu" else nn.tanh

        embed_model = CNN(
            config=self.config,
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(world_state)
        embedding = nn.LayerNorm()(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        critic = _dense(
            embedding,
            self.config,
            "critic_fc",
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )
        critic = nn.relu(critic)
        critic = _dense(
            critic,
            self.config,
            "critic_value",
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )
        return hidden, jnp.squeeze(critic, axis=-1)


class CriticMLPRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x

        embedding = _dense(
            world_state,
            self.config,
            "critic_input_fc",
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )
        embedding = nn.relu(embedding)

        hidden, embedding = ScannedRNN()(hidden, (embedding, dones))

        critic = _dense(
            embedding,
            self.config,
            "critic_fc",
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )
        critic = nn.relu(critic)
        critic = _dense(
            critic,
            self.config,
            "critic_value",
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )
        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _to_wandb_loggable(tree):
    def _convert(x):
        x_np = np.asarray(x)
        if x_np.shape == ():
            return x_np.item()
        return x_np

    return jax.tree_util.tree_map(_convert, tree)


def _flatten_nested_dict(tree, parent_key: str = ""):
    flat = {}
    for key, value in tree.items():
        key = str(key)
        full_key = f"{parent_key}_{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(_flatten_nested_dict(value, full_key))
        else:
            flat[full_key] = value
    return flat


def _configure_wandb_run(run):
    # Ensure all user metrics share a consistent x-axis in W&B charts.
    run.define_metric("env_step")
    run.define_metric("*", step_metric="env_step")


def _to_scalar(value):
    value_np = np.asarray(value)
    if value_np.shape == ():
        return float(value_np)
    return None


def _count_params(tree) -> int:
    return int(sum(np.prod(np.asarray(x).shape) for x in jax.tree_util.tree_leaves(tree)))


def _count_adapter_params(tree) -> int:
    flat = traverse_util.flatten_dict(tree)
    total = 0
    for path, value in flat.items():
        if len(path) > 0 and path[-1] in {"lora_A", "lora_B", "adapter_A", "adapter_B"}:
            total += int(np.prod(np.asarray(value).shape))
    return total


def _mask_grads_by_train_flags(
    tree,
    *,
    train_base: bool = True,
    train_adapter_a: bool = True,
    train_adapter_b: bool = True,
):
    flat = traverse_util.flatten_dict(tree)
    masked = {}
    for path, value in flat.items():
        leaf_name = path[-1] if len(path) > 0 else ""
        if leaf_name in {"lora_A", "adapter_A"}:
            keep = train_adapter_a
        elif leaf_name in {"lora_B", "adapter_B"}:
            keep = train_adapter_b
        else:
            keep = train_base

        if keep:
            masked[path] = value
        else:
            masked[path] = jnp.zeros_like(value)
    return traverse_util.unflatten_dict(masked)


def _stable_hash(text: str) -> int:
    # FNV-1a 32-bit hash for deterministic path-specific projections.
    h = 2166136261
    for ch in text.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def _context_features_from_traj(traj: Dict[str, Any], action_dim: int) -> np.ndarray:
    obs = np.asarray(traj["obs"], dtype=np.float32)
    done = np.asarray(traj["done"], dtype=np.float32)
    action = np.asarray(traj["action"], dtype=np.int32)
    rewards = np.asarray(traj["rewards"], dtype=np.float32)

    if rewards.size == 0:
        rewards = np.zeros((1,), dtype=np.float32)
    if action.size == 0:
        action = np.zeros((1,), dtype=np.int32)

    action_hist = np.bincount(action.reshape(-1), minlength=action_dim).astype(np.float32)
    action_hist_sum = float(action_hist.sum())
    if action_hist_sum > 0:
        action_hist /= action_hist_sum

    obs_flat = obs.reshape((obs.shape[0], -1))
    obs_mean = float(obs_flat.mean())
    obs_std = float(obs_flat.std())
    done_ratio = float(done.mean()) if done.size > 0 else 0.0
    rew_mean = float(rewards.mean())
    rew_std = float(rewards.std())
    rew_max = float(rewards.max())
    rew_min = float(rewards.min())
    ep_len = float(rewards.shape[0])

    stats = np.asarray(
        [
            obs_mean,
            obs_std,
            done_ratio,
            rew_mean,
            rew_std,
            rew_max,
            rew_min,
            ep_len / 100.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([stats, action_hist], axis=0)


def _init_adapter_b_from_context(
    params,
    context_vec: np.ndarray,
    *,
    seed: int = 0,
    scale: float = 0.15,
    blend: float = 0.7,
):
    flat = traverse_util.flatten_dict(params)
    context = np.asarray(context_vec, dtype=np.float32).reshape(-1)
    if context.size == 0:
        return params

    context = context / (np.linalg.norm(context) + 1e-6)
    ctx_dim = int(context.shape[0])
    scale = float(max(scale, 0.0))
    blend = float(np.clip(blend, 0.0, 1.0))

    updated = {}
    for path, value in flat.items():
        if len(path) == 0 or path[-1] not in {"lora_B", "adapter_B"}:
            updated[path] = value
            continue

        value_np = np.asarray(value)
        if value_np.ndim != 2:
            updated[path] = value
            continue

        rank, out_dim = value_np.shape
        key = "/".join([str(x) for x in path])
        rs = np.random.RandomState((int(seed) + _stable_hash(key)) % (2**32 - 1))

        proj_u = rs.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(ctx_dim, 1)),
            size=(ctx_dim, rank),
        ).astype(np.float32)
        proj_v = rs.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(max(ctx_dim, 1)),
            size=(ctx_dim, out_dim),
        ).astype(np.float32)

        u = np.tanh(context @ proj_u)
        v = np.tanh(context @ proj_v)
        b_ctx = np.outer(u, v).astype(np.float32)

        b_old = value_np.astype(np.float32, copy=False)
        b_new = blend * b_old + (1.0 - blend) * scale * b_ctx
        updated[path] = jnp.asarray(b_new, dtype=value.dtype)

    return traverse_util.unflatten_dict(updated)


def _discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


class VmapCheckpointManager:
    def __init__(self, save_dir: str, prefix: str, best_mode: str = "max"):
        self.save_dir = save_dir
        self.prefix = prefix
        self.best_mode = best_mode
        self.best_metric = -np.inf if best_mode == "max" else np.inf
        os.makedirs(self.save_dir, exist_ok=True)

    def _is_better(self, metric: float) -> bool:
        if self.best_mode == "min":
            return metric < self.best_metric
        return metric > self.best_metric

    def _save(self, params, suffix: str):
        save_path = os.path.join(self.save_dir, f"{self.prefix}_{suffix}.safetensors")
        save_params(params, save_path)

    def maybe_save_best(self, params, metric):
        metric = float(np.asarray(metric))
        if self._is_better(metric):
            self.best_metric = metric
            self._save(params, "best")

    def save_final(self, params):
        self._save(params, "final")


def make_train(
    config,
    checkpoint_callback=None,
    checkpoint_metric_key: str = "returned_episode_returns",
    wandb_callback=None,
):
    world_state_source = config.get("WORLD_STATE_SOURCE", "env_state")
    if world_state_source not in ("env_state", "obs_concat"):
        raise ValueError(
            f"WORLD_STATE_SOURCE must be one of ['env_state', 'obs_concat'], got {world_state_source}"
        )

    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config.get("SCALE_CLIP_EPS", False)
        else config["CLIP_EPS"]
    )
    log_interval_updates = max(int(config.get("METRIC_LOG_INTERVAL_UPDATES", 1)), 1)
    checkpoint_interval_updates = max(
        int(config.get("CHECKPOINT_INTERVAL_UPDATES", 1)), 1
    )

    env = OvercookedV2WorldStateWrapper(env, source=world_state_source)
    world_state_shape = tuple(env.world_state_shape())
    world_state_size = int(env.world_state_size())
    world_state_is_image = world_state_source == "obs_concat"

    env = OvercookedV2LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_lr = config["LR"]
        if not config.get("ANNEAL_LR", False):
            return lambda _: base_lr

        warmup_ratio = float(config.get("LR_WARMUP", 0.0))
        warmup_updates = int(warmup_ratio * config["NUM_UPDATES"])
        steps_per_update = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
        warmup_steps = warmup_updates * steps_per_update

        cosine_updates = max(config["NUM_UPDATES"] - warmup_updates, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_lr, decay_steps=cosine_updates * steps_per_update
        )
        if warmup_steps <= 0:
            return cosine_fn

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_lr,
            transition_steps=warmup_steps,
        )
        return optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps],
        )

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def format_world_state(world_state_obs):
        # vmapped env returns [num_envs, num_agents, ...]
        world_state = world_state_obs.swapaxes(0, 1)
        if world_state_is_image:
            return world_state.reshape((config["NUM_ACTORS"], *world_state_shape))
        return world_state.reshape((config["NUM_ACTORS"], world_state_size))

    def train(rng, seed_idx):
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticCNNRNN(config=config) if world_state_is_image else CriticMLPRNN(config=config)

        obs_shape = env.observation_space().shape
        critic_obs_shape = world_state_shape if world_state_is_image else (world_state_size,)

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *obs_shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        cr_init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *critic_obs_shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )
        actor_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        critic_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)

        lr_schedule = create_learning_rate_fn()
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_params,
            tx=critic_tx,
        )

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        def _update_step(update_runner_state, unused):
            runner_state, update_step = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = runner_state

                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    config["NUM_ACTORS"], *env.observation_space().shape
                )
                ac_in = (obs_batch[None, :], last_done[None, :])
                ac_hstate, pi = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act = {k: v.flatten() for k, v in env_act.items()}

                world_state = format_world_state(last_obs["world_state"])
                cr_in = (world_state[None, :], last_done[None, :])
                cr_hstate, value = critic_network.apply(train_states[1].params, hstates[1], cr_in)

                rng, _rng = jax.random.split(rng)
                step_rng = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_rng, env_state, env_act)

                original_reward = jnp.array([reward[a] for a in env.agents])
                current_timestep = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward = jnp.array([info["shaped_reward"][a] for a in env.agents])
                combined_reward = jnp.array([reward[a] for a in env.agents])
                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    done=last_done,
                    action=action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    world_state=world_state,
                    info=info,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            train_states, env_state, last_obs, last_done, hstates, rng = runner_state
            last_world_state = format_world_state(last_obs["world_state"])
            cr_in = (last_world_state[None, :], last_done[None, :])
            _, last_val = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        actor_loss_1 = ratio * gae
                        actor_loss_2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        policy_loss = -jnp.minimum(actor_loss_1, actor_loss_2).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1.0) > config["CLIP_EPS"])
                        actor_loss = policy_loss - config["ENT_COEF"] * entropy
                        return actor_loss, (
                            policy_loss,
                            entropy,
                            approx_kl,
                            clip_frac,
                            ratio.mean(),
                        )

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        ac_init_hstate,
                        traj_batch,
                        advantages,
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        cr_init_hstate,
                        traj_batch,
                        targets,
                    )

                    actor_grads = _mask_grads_by_train_flags(
                        actor_grads,
                        train_base=bool(config.get("MAIN_TRAIN_BASE", True)),
                        train_adapter_a=_train_adapter_a_flag(config, "MAIN", True),
                        train_adapter_b=_train_adapter_b_flag(config, "MAIN", True),
                    )
                    critic_grads = _mask_grads_by_train_flags(
                        critic_grads,
                        train_base=bool(config.get("MAIN_TRAIN_BASE", True)),
                        train_adapter_a=_train_adapter_a_flag(config, "MAIN", True),
                        train_adapter_b=_train_adapter_b_flag(config, "MAIN", True),
                    )

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

                    loss_info = {
                        "total_loss": actor_loss[0] + critic_loss[0],
                        "actor_loss": actor_loss[0],
                        "policy_loss": actor_loss[1][0],
                        "value_loss": critic_loss[1],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][4],
                        "approx_kl": actor_loss[1][2],
                        "clip_frac": actor_loss[1][3],
                    }
                    return (actor_train_state, critic_train_state), loss_info

                train_states, init_hstates, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )
                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree_util.tree_map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric["loss"] = loss_info
            rng = update_state[-1]

            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            update_step = update_step + 1
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]

            is_last_update = jnp.equal(update_step, config["NUM_UPDATES"])
            do_metric_log = jnp.logical_or(
                jnp.equal(jnp.mod(update_step, log_interval_updates), 0),
                is_last_update,
            )
            do_checkpoint = jnp.logical_or(
                jnp.equal(jnp.mod(update_step, checkpoint_interval_updates), 0),
                is_last_update,
            )

            if checkpoint_callback is not None:
                ckpt_metric = metric
                for key in checkpoint_metric_key.split("."):
                    ckpt_metric = ckpt_metric[key]
                ckpt_metric = jnp.asarray(ckpt_metric).mean()

                def _emit_checkpoint(_):
                    jax.debug.callback(
                        checkpoint_callback,
                        train_states[0].params,
                        ckpt_metric,
                        metric["update_step"],
                        metric["env_step"],
                        seed_idx,
                        ordered=True,
                    )
                    return jnp.int32(0)

                _ = jax.lax.cond(
                    do_checkpoint,
                    _emit_checkpoint,
                    lambda _: jnp.int32(0),
                    operand=jnp.int32(0),
                )

            if wandb_callback is not None:
                def _emit_metric(_):
                    jax.debug.callback(
                        wandb_callback,
                        metric,
                        seed_idx,
                    )
                    return jnp.int32(0)

                _ = jax.lax.cond(
                    do_metric_log,
                    _emit_metric,
                    lambda _: jnp.int32(0),
                    operand=jnp.int32(0),
                )

            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                hstates,
                rng,
            )
            return (runner_state, update_step), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(version_base=None, config_path="config", config_name="talora_rnn_overcooked_v2")
def main(config):
    config = OmegaConf.to_container(config)
    layout_name = config["ENV_KWARGS"]["layout"]
    ws_source = config.get("WORLD_STATE_SOURCE", "env_state")
    num_seeds = config["NUM_SEEDS"]
    metric_key = config.get("SAVE_BEST_BY", "returned_episode_returns")
    save_path = config.get("SAVE_PATH", "")
    save_best_mode = config.get("SAVE_BEST_MODE", "max")
    alg_name = "talora_rnn_overcooked_v2"
    run_name_prefix = f"talora_rnn_overcooked_v2_{layout_name}_{ws_source}"
    wandb_enabled = config["WANDB_MODE"] != "disabled"
    wandb_group_override = config.get("WANDB_GROUP", "")
    algo_label = str(config.get("WANDB_ALGO_LABEL", "TALORA")).strip() or "TALORA"
    seed_runs = {}
    seed_group = None
    tensorboard_enabled = bool(config.get("TENSORBOARD_ENABLED", False))
    tb_log_seed_runs = bool(config.get("TENSORBOARD_LOG_SEED_RUNS", True))
    tb_log_aggregate = bool(config.get("TENSORBOARD_LOG_AGGREGATE", True))
    tb_flush_secs = int(config.get("TENSORBOARD_FLUSH_SECS", 10))
    tb_root_dir = str(config.get("TENSORBOARD_DIR", os.path.join("tb_logs", alg_name)))
    tb_run_name = str(config.get("TENSORBOARD_RUN_NAME", "")).strip()
    tb_seed_writers = {}
    tb_aggregate_writer = None
    tb_aggregate_buffer = {}
    tb_log_std_line = bool(config.get("TENSORBOARD_LOG_STD_LINE", False))
    tb_margin_chart_enabled = bool(config.get("TENSORBOARD_MARGIN_CHART", True))
    tb_margin_layout_written = False
    tta_enabled = bool(config.get("TTA_ENABLE", False))

    if tensorboard_enabled:
        if SummaryWriter is None:
            raise ImportError(
                "TensorBoard logging is enabled, but neither tensorboardX nor torch.utils.tensorboard is installed."
            )
        if not tb_run_name:
            tb_run_name = f"{run_name_prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        tb_run_dir = os.path.join(tb_root_dir, tb_run_name)
        os.makedirs(tb_run_dir, exist_ok=True)
        print(f"[TensorBoard] Logging to: {tb_run_dir}")

        if tb_log_seed_runs:
            for i in range(num_seeds):
                tb_seed_writers[i] = SummaryWriter(
                    log_dir=os.path.join(tb_run_dir, f"seed_{i}"),
                    flush_secs=tb_flush_secs,
                )
        if tb_log_aggregate:
            tb_aggregate_writer = SummaryWriter(
                log_dir=os.path.join(tb_run_dir, "aggregate"),
                flush_secs=tb_flush_secs,
            )

    if wandb_enabled:
        if num_seeds <= 1:
            seed_runs[0] = wandb.init(
                entity=config["ENTITY"],
                project=config["PROJECT"],
                tags=[algo_label, "RNN", "OvercookedV2", ws_source],
                config=config,
                mode=config["WANDB_MODE"],
                name=run_name_prefix,
            )
            seed_runs[0].config.update({"algo": algo_label, "seed_index": 0}, allow_val_change=True)
            _configure_wandb_run(seed_runs[0])
        else:
            # Use a stable experiment-level group by default so multi-seed runs
            # are consistently grouped in W&B without requiring CLI overrides.
            seed_group = (
                str(wandb_group_override).strip()
                if str(wandb_group_override).strip()
                else run_name_prefix
            )
            for i in range(num_seeds):
                run_i = wandb.init(
                    entity=config["ENTITY"],
                    project=config["PROJECT"],
                    tags=[algo_label, "RNN", "OvercookedV2", ws_source],
                    config=config,
                    mode=config["WANDB_MODE"],
                    group=seed_group,
                    name=f"{run_name_prefix}_s{i}",
                    reinit="create_new",
                )
                run_i.config.update({"seed_index": i, "algo": algo_label}, allow_val_change=True)
                _configure_wandb_run(run_i)
                seed_runs[i] = run_i

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        seed_ids = jnp.arange(num_seeds, dtype=jnp.int32)

        checkpoint_managers = {}
        if save_path:
            for i in range(num_seeds):
                run_rng = int(np.asarray(rngs[i][0]))
                run_dir = os.path.join(
                    save_path,
                    alg_name,
                    config["ENV_NAME"],
                    f"{layout_name}_{ws_source}_seed{config['SEED']}_vmap{i}_rng{run_rng}",
                )
                checkpoint_managers[i] = VmapCheckpointManager(
                    save_dir=run_dir,
                    prefix=alg_name,
                    best_mode=save_best_mode,
                )
                OmegaConf.save(
                    OmegaConf.create(config),
                    os.path.join(run_dir, "config.yaml"),
                )

        def _extract_params_at(tree, idx):
            return jax.tree_util.tree_map(lambda x: x[idx], tree)

        def _checkpoint_callback(params, metric, update_step, env_step, seed_idx):
            if not checkpoint_managers:
                return
            seed_idx_arr = np.asarray(seed_idx)
            if seed_idx_arr.ndim == 0:
                checkpoint_managers[int(seed_idx_arr)].maybe_save_best(params, metric)
                return
            metric_arr = np.asarray(metric)
            for i, sid in enumerate(seed_idx_arr):
                params_i = _extract_params_at(params, i)
                checkpoint_managers[int(sid)].maybe_save_best(params_i, metric_arr[i])

        def _iter_seed_metrics(metric_loggable, seed_idx):
            seed_idx_arr = np.asarray(seed_idx)
            if seed_idx_arr.ndim == 0:
                return [(int(seed_idx_arr), metric_loggable)]

            num_seed_entries = seed_idx_arr.shape[0]

            def _metric_for_seed(idx):
                return jax.tree_util.tree_map(
                    lambda x: np.asarray(x)[idx]
                    if np.asarray(x).ndim > 0 and np.asarray(x).shape[0] == num_seed_entries
                    else x,
                    metric_loggable,
                )

            return [
                (int(seed_idx_arr[i]), _metric_for_seed(i))
                for i in range(num_seed_entries)
            ]

        def _wandb_callback(metric, seed_idx):
            if not wandb_enabled:
                return
            metric_loggable = _to_wandb_loggable(metric)
            seed_metric_pairs = _iter_seed_metrics(metric_loggable, seed_idx)

            for seed_id, metric_i in seed_metric_pairs:
                flat_metric = _flatten_nested_dict(metric_i)
                if "returned_episode_returns" in flat_metric:
                    flat_metric["returns"] = flat_metric["returned_episode_returns"]

                env_step = int(np.asarray(flat_metric["env_step"]))
                run_i = seed_runs.get(seed_id)
                if run_i is None and num_seeds <= 1:
                    run_i = seed_runs.get(0)
                if run_i is not None:
                    run_i.log(flat_metric, step=env_step)

        def _tensorboard_callback(metric, seed_idx):
            if not tensorboard_enabled:
                return
            nonlocal tb_margin_layout_written

            metric_loggable = _to_wandb_loggable(metric)
            seed_metric_pairs = _iter_seed_metrics(metric_loggable, seed_idx)
            expected_seed_count = max(int(num_seeds), 1)

            for seed_id, metric_i in seed_metric_pairs:
                flat_metric = _flatten_nested_dict(metric_i)
                if "returned_episode_returns" in flat_metric:
                    flat_metric["returns"] = flat_metric["returned_episode_returns"]

                env_step = int(np.asarray(flat_metric["env_step"]))

                writer = tb_seed_writers.get(seed_id)
                if writer is None and num_seeds <= 1:
                    writer = tb_seed_writers.get(0)

                for key, value in flat_metric.items():
                    scalar = _to_scalar(value)
                    if scalar is None:
                        continue
                    if writer is not None:
                        writer.add_scalar(key, scalar, env_step)

                if tb_aggregate_writer is not None:
                    step_bucket = tb_aggregate_buffer.setdefault(env_step, {})
                    step_bucket[int(seed_id)] = flat_metric
                    if len(step_bucket) >= expected_seed_count:
                        aggregate_values = {}
                        for metric_seed in step_bucket.values():
                            for key, value in metric_seed.items():
                                scalar = _to_scalar(value)
                                if scalar is None:
                                    continue
                                aggregate_values.setdefault(key, []).append(scalar)

                        aggregate_stats = {}
                        for key, values in aggregate_values.items():
                            values_np = np.asarray(values, dtype=np.float64)
                            mean_value = float(values_np.mean())
                            std_value = float(values_np.std(ddof=0))
                            aggregate_stats[key] = (mean_value, std_value)
                            tb_aggregate_writer.add_scalar(f"mean/{key}", mean_value, env_step)
                            tb_aggregate_writer.add_scalar(
                                f"band_lower/{key}", mean_value - std_value, env_step
                            )
                            tb_aggregate_writer.add_scalar(
                                f"band_upper/{key}", mean_value + std_value, env_step
                            )
                            if tb_log_std_line:
                                tb_aggregate_writer.add_scalar(f"std/{key}", std_value, env_step)

                        if tb_margin_chart_enabled and not tb_margin_layout_written:
                            margin_keys = sorted(
                                key
                                for key in aggregate_stats.keys()
                                if key not in {"env_step", "update_step"}
                            )
                            if margin_keys:
                                layout = {"Aggregate Mean-Std Band": {}}
                                for key in margin_keys:
                                    layout["Aggregate Mean-Std Band"][key] = [
                                        "Margin",
                                        [f"mean/{key}", f"band_lower/{key}", f"band_upper/{key}"],
                                    ]
                                tb_aggregate_writer.add_custom_scalars(layout)
                                tb_margin_layout_written = True

                        del tb_aggregate_buffer[env_step]

        def _metrics_callback(metric, seed_idx):
            _wandb_callback(metric, seed_idx)
            _tensorboard_callback(metric, seed_idx)

        tta_num_updates = int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
        tta_base_env_step = int(tta_num_updates * config["NUM_STEPS"] * config["NUM_ENVS"])

        def _run_tta_for_seed(seed_id: int, actor_params):
            tta_env_kwargs = dict(config["ENV_KWARGS"])
            tta_env_override = config.get("TTA_ENV_KWARGS", {})
            if isinstance(tta_env_override, dict):
                tta_env_kwargs.update(tta_env_override)
            tta_layout = str(config.get("TTA_LAYOUT", "")).strip()
            if tta_layout:
                tta_env_kwargs["layout"] = tta_layout

            tta_env = jaxmarl.make(config["ENV_NAME"], **tta_env_kwargs)
            tta_env = OvercookedV2LogWrapper(tta_env, replace_info=False)
            tta_num_envs = int(config.get("TTA_NUM_ENVS", 1))
            tta_num_actors = int(tta_env.num_agents * tta_num_envs)
            obs_shape = tuple(tta_env.observation_space().shape)

            actor_network = ActorRNN(tta_env.action_space(tta_env.agents[0]).n, config=config)
            gamma = float(config.get("TTA_GAMMA", 0.99))
            normalize_returns = bool(config.get("TTA_NORMALIZE_RETURNS", True))
            deterministic_eval = bool(config.get("TTA_DETERMINISTIC_EVAL", True))
            use_shaped_reward = bool(config.get("TTA_USE_SHAPED_REWARD", False))
            tta_train_base = bool(config.get("TTA_TRAIN_BASE", False))
            tta_train_adapter_a = _train_adapter_a_flag(config, "TTA", False)
            tta_train_adapter_b = _train_adapter_b_flag(config, "TTA", True)
            tta_action_dim = int(tta_env.action_space(tta_env.agents[0]).n)

            @jax.jit
            def _policy_step_sample(params, hstate, obs_batch, done_batch, sample_rng):
                ac_in = (obs_batch[None, :], done_batch[None, :])
                hstate, pi = actor_network.apply(params, hstate, ac_in)
                action = pi.sample(seed=sample_rng)
                return hstate, action

            @jax.jit
            def _policy_step_mode(params, hstate, obs_batch, done_batch):
                ac_in = (obs_batch[None, :], done_batch[None, :])
                hstate, pi = actor_network.apply(params, hstate, ac_in)
                action = pi.mode()
                return hstate, action

            if (
                (not tta_train_base)
                and (not tta_train_adapter_a)
                and tta_train_adapter_b
                and _count_adapter_params(actor_params) == 0
            ):
                return {
                    "seed_id": seed_id,
                    "layout": tta_env_kwargs.get("layout", ""),
                    "before_return_mean": float("nan"),
                    "after_return_mean": float("nan"),
                    "gain_return_mean": float("nan"),
                    "num_updates": int(config.get("TTA_NUM_ADAPT_EPISODES", 0)),
                    "warning": "No adapter parameters found while TTA updates are set to B-only.",
                }

            def _collect_episode(params, rng_key, deterministic: bool):
                rng_key, _reset_rng = jax.random.split(rng_key)
                reset_rng = jax.random.split(_reset_rng, tta_num_envs)
                obsv, env_state = jax.vmap(tta_env.reset, in_axes=(0,))(reset_rng)
                done_batch = jnp.zeros((tta_num_actors,), dtype=bool)
                hstate = ScannedRNN.initialize_carry(
                    tta_num_actors, int(config["GRU_HIDDEN_DIM"])
                )

                traj_obs = []
                traj_done = []
                traj_action = []
                rewards = []
                episode_return = 0.0

                for _ in range(int(tta_env.max_steps)):
                    obs_batch = jnp.stack([obsv[a] for a in tta_env.agents]).reshape(
                        tta_num_actors, *obs_shape
                    )

                    rng_key, _sample_rng, _step_rng = jax.random.split(rng_key, 3)
                    if deterministic:
                        hstate, action = _policy_step_mode(params, hstate, obs_batch, done_batch)
                    else:
                        hstate, action = _policy_step_sample(
                            params, hstate, obs_batch, done_batch, _sample_rng
                        )

                    env_act = unbatchify(
                        action, tta_env.agents, tta_num_envs, tta_env.num_agents
                    )
                    env_act = {k: v.flatten() for k, v in env_act.items()}
                    step_rng = jax.random.split(_step_rng, tta_num_envs)
                    obsv, env_state, reward, done, info = jax.vmap(
                        tta_env.step, in_axes=(0, 0, 0)
                    )(step_rng, env_state, env_act)

                    reward_batch = batchify(reward, tta_env.agents, tta_num_actors).squeeze()
                    if use_shaped_reward and "shaped_reward" in info:
                        shaped_batch = batchify(
                            info["shaped_reward"], tta_env.agents, tta_num_actors
                        ).squeeze()
                        reward_batch = reward_batch + shaped_batch

                    reward_scalar = float(np.asarray(reward_batch.mean()))
                    episode_return += reward_scalar

                    traj_obs.append(obs_batch)
                    traj_done.append(done_batch)
                    traj_action.append(action.squeeze(0))
                    rewards.append(reward_scalar)

                    done_batch = batchify(done, tta_env.agents, tta_num_actors).squeeze()
                    if bool(np.asarray(done["__all__"]).all()):
                        break

                if len(traj_obs) == 0:
                    traj_obs = [jnp.zeros((tta_num_actors, *obs_shape), dtype=jnp.float32)]
                    traj_done = [jnp.zeros((tta_num_actors,), dtype=bool)]
                    traj_action = [jnp.zeros((tta_num_actors,), dtype=jnp.int32)]
                    rewards = [0.0]

                traj = {
                    "obs": jnp.stack(traj_obs, axis=0),
                    "done": jnp.stack(traj_done, axis=0),
                    "action": jnp.stack(traj_action, axis=0),
                    "rewards": np.asarray(rewards, dtype=np.float32),
                }
                return traj, episode_return, rng_key

            def _tta_loss_fn(params, obs_seq, done_seq, action_seq, returns_seq):
                init_h = ScannedRNN.initialize_carry(
                    tta_num_actors, int(config["GRU_HIDDEN_DIM"])
                )
                _, pi = actor_network.apply(params, init_h, (obs_seq, done_seq))
                log_prob = pi.log_prob(action_seq)
                advantage = returns_seq
                if normalize_returns:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                policy_loss = -(log_prob.mean(axis=-1) * advantage).mean()
                entropy = pi.entropy().mean()
                return policy_loss - float(config.get("TTA_ENT_COEF", 0.0)) * entropy

            grad_fn = jax.value_and_grad(_tta_loss_fn)
            tta_state = TrainState.create(
                apply_fn=actor_network.apply,
                params=actor_params,
                tx=optax.adam(float(config.get("TTA_LR", 1e-4))),
            )

            rng_tta = jax.random.PRNGKey(
                int(config["SEED"]) + 10_000 + int(seed_id) * 1_003
            )

            context_norm = float("nan")
            if bool(config.get("TTA_INIT_B_FROM_CONTEXT", True)):
                num_context_eps = max(1, int(config.get("TTA_NUM_CONTEXT_EPISODES", 2)))
                context_deterministic = bool(config.get("TTA_CONTEXT_DETERMINISTIC", False))
                context_feats = []
                for _ in range(num_context_eps):
                    context_traj, _, rng_tta = _collect_episode(
                        tta_state.params,
                        rng_tta,
                        deterministic=context_deterministic,
                    )
                    context_feats.append(_context_features_from_traj(context_traj, tta_action_dim))
                context_vec = np.mean(np.stack(context_feats, axis=0), axis=0)
                context_norm = float(np.linalg.norm(context_vec))
                tta_state = tta_state.replace(
                    params=_init_adapter_b_from_context(
                        tta_state.params,
                        context_vec,
                        seed=int(config["SEED"]) + int(seed_id) * 1009,
                        scale=float(config.get("TTA_B_INIT_SCALE", 0.15)),
                        blend=float(config.get("TTA_B_INIT_BLEND", 0.7)),
                    )
                )

            eval_episodes = int(config.get("TTA_NUM_EVAL_EPISODES", 10))
            before_returns = []
            for _ in range(eval_episodes):
                _, ep_ret, rng_tta = _collect_episode(
                    tta_state.params, rng_tta, deterministic_eval
                )
                before_returns.append(ep_ret)

            num_adapt = int(config.get("TTA_NUM_ADAPT_EPISODES", 10))
            for upd in range(num_adapt):
                traj, ep_ret, rng_tta = _collect_episode(
                    tta_state.params, rng_tta, deterministic=False
                )
                returns = _discounted_returns(traj["rewards"], gamma)
                returns_jnp = jnp.asarray(returns, dtype=jnp.float32)
                loss, grads = grad_fn(
                    tta_state.params,
                    traj["obs"],
                    traj["done"],
                    traj["action"],
                    returns_jnp,
                )
                grads = _mask_grads_by_train_flags(
                    grads,
                    train_base=tta_train_base,
                    train_adapter_a=tta_train_adapter_a,
                    train_adapter_b=tta_train_adapter_b,
                )
                tta_state = tta_state.apply_gradients(grads=grads)

                run_i = seed_runs.get(seed_id)
                if run_i is None and num_seeds <= 1:
                    run_i = seed_runs.get(0)
                if run_i is not None:
                    tta_env_step = int(tta_base_env_step + upd + 1)
                    run_i.log(
                        {
                            "env_step": tta_env_step,
                            "tta_update": int(upd + 1),
                            "tta_loss": float(np.asarray(loss)),
                            "tta_episode_return": float(ep_ret),
                            "tta_context_norm": context_norm,
                        },
                        step=tta_env_step,
                    )

            after_returns = []
            for _ in range(eval_episodes):
                _, ep_ret, rng_tta = _collect_episode(
                    tta_state.params, rng_tta, deterministic_eval
                )
                after_returns.append(ep_ret)

            before_mean = float(np.mean(before_returns)) if before_returns else float("nan")
            after_mean = float(np.mean(after_returns)) if after_returns else float("nan")
            return {
                "seed_id": seed_id,
                "layout": tta_env_kwargs.get("layout", ""),
                "before_return_mean": before_mean,
                "after_return_mean": after_mean,
                "gain_return_mean": after_mean - before_mean,
                "num_updates": num_adapt,
                "context_norm": context_norm,
                "warning": "",
            }

        train_jit = jax.jit(
            make_train(
                config,
                checkpoint_callback=_checkpoint_callback if checkpoint_managers else None,
                checkpoint_metric_key=metric_key,
                wandb_callback=_metrics_callback
                if (wandb_enabled or tensorboard_enabled)
                else None,
            )
        )
        out = jax.block_until_ready(jax.vmap(train_jit)(rngs, seed_ids))

        final_actor_train_state = out["runner_state"][0][0][0]
        total_actor_params = _count_params(final_actor_train_state.params)
        adapter_actor_params = _count_adapter_params(final_actor_train_state.params)
        adapter_type = _adapter_type(config).upper()
        if adapter_actor_params > 0:
            print(
                f"[{adapter_type}] Actor params: total={total_actor_params}, adapter={adapter_actor_params} "
                f"({100.0 * adapter_actor_params / max(total_actor_params, 1):.2f}%)."
            )
        else:
            print(f"[{adapter_type}] Actor params: total={total_actor_params}, adapter=0.")

        if checkpoint_managers:
            for i in range(num_seeds):
                params_i = _extract_params_at(final_actor_train_state.params, i)
                checkpoint_managers[i].save_final(params_i)

        if tta_enabled:
            print("[TTA] Running test-time adaptation phase...")
            tta_results = []
            for i in range(num_seeds):
                params_i = _extract_params_at(final_actor_train_state.params, i)
                result_i = _run_tta_for_seed(i, params_i)
                tta_results.append(result_i)

                print(
                    f"[TTA][seed={i}] layout={result_i['layout']} "
                    f"before={result_i['before_return_mean']:.4f} "
                    f"after={result_i['after_return_mean']:.4f} "
                    f"gain={result_i['gain_return_mean']:.4f}"
                )
                if result_i["warning"]:
                    print(f"[TTA][seed={i}] warning: {result_i['warning']}")

                run_i = seed_runs.get(i)
                if run_i is None and num_seeds <= 1:
                    run_i = seed_runs.get(0)
                if run_i is not None:
                    tta_summary_step = int(tta_base_env_step + int(result_i["num_updates"]) + 1)
                    run_i.log(
                        {
                            "env_step": tta_summary_step,
                            "tta_before_return_mean": result_i["before_return_mean"],
                            "tta_after_return_mean": result_i["after_return_mean"],
                            "tta_gain_return_mean": result_i["gain_return_mean"],
                        },
                        step=tta_summary_step,
                    )

            valid_gains = [
                x["gain_return_mean"] for x in tta_results if np.isfinite(x["gain_return_mean"])
            ]
            if len(valid_gains) > 0:
                print(
                    f"[TTA] Mean gain over seeds: {float(np.mean(valid_gains)):.4f} "
                    f"(std={float(np.std(valid_gains)):.4f})"
                )

    for run_i in seed_runs.values():
        run_i.finish()

    for writer in tb_seed_writers.values():
        writer.close()
    if tb_aggregate_writer is not None:
        tb_aggregate_writer.close()


if __name__ == "__main__":
    main()
