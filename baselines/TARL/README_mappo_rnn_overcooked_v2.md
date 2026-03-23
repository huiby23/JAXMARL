# MAPPO + Overcooked_v2 代码速读文档

对应代码文件：`baselines/TARL/mappo_rnn_overcooked_v2.py`  
目标：让你在不了解实现细节时，快速知道“这段代码在做什么、怎么跑、从哪里改”。

## 1. 先记住这个版本的算法结构

这份实现是 **MAPPO-RNN**：
- actor：每个 agent 用自己的局部观测做动作采样。
- critic：集中式 critic，输入是所有 agent 观测拼接后的 `world_state`。
- 训练：PPO clipping + GAE + RNN hidden state。
- 环境：`overcooked_v2`，保留了 shaped reward 的退火逻辑。

一句话流程：
1. 用 actor 采样动作；
2. 用 critic 估值；
3. 与环境交互得到轨迹；
4. 用 GAE 算优势；
5. 分别更新 actor/critic；
6. 记录日志到 wandb。

---

## 2. 推荐阅读顺序（5 分钟）

1. `main`（596-620）：脚本入口，知道怎么启动训练。  
2. `make_train`（202-593）：训练主工厂，所有逻辑都在这里。  
3. `_env_step`（318-380）：每一步如何采样和与环境交互。  
4. `_calculate_gae`（393-414）：优势怎么计算。  
5. `_update_minbatch`（417-497）：actor/critic 损失和梯度更新。  
6. 网络定义 `ActorRNN/CriticRNN`（106-165）：模型结构。  

---

## 3. 按代码块解释（含行号）

## 3.1 导入与依赖（1-17）
- 关键依赖：
  - `jax/jax.numpy`：向量化和 JIT。
  - `flax.linen`：网络定义。
  - `optax`：优化器和学习率调度。
  - `distrax`：离散动作分布。
  - `OvercookedV2LogWrapper`：日志与回报统计。

## 3.2 `ScannedRNN`（20-42）
- 用 `nn.scan` 封装 GRU，使其按时间维自动扫描。
- `resets` 为真时重置 hidden state（34 行）。
- 作用：同一套参数在整个时间序列复用，适合 PPO rollout。

## 3.3 `CNN` 编码器（44-103）
- 输入：单帧图像观测（或 world_state 图像）。
- 输出：定长 embedding（`output_size`）。
- Actor 和 Critic 都复用这个 CNN 作为前端特征提取。

## 3.4 `ActorRNN`（106-136）
- 输入：`obs, dones`（112 行）。
- `CNN -> LayerNorm -> GRU -> MLP -> Categorical(logits)`。
- 输出：新 hidden state 和策略分布 `pi`。
- 注意：actor 只看局部观测，不看全局状态（MAPPO 标准做法）。

## 3.5 `CriticRNN`（138-165）
- 输入：`world_state, dones`（143 行）。
- 结构与 actor 类似，但最终输出标量 value。
- 这是集中式 critic：估值时利用更多信息。

## 3.6 `Transition` 与辅助函数（167-199）
- `Transition` 保存 rollout 需要的字段。
- `batchify/unbatchify`：
  - 在 `dict(agent -> tensor)` 与 `[(num_actors), ...]` 间转换。
- `make_world_state`（189-199）：
  - 把所有 agent 观测拼接成 `world_state`；
  - 并复制给每个 agent（每个 actor 样本都能看到同一全局输入）。

## 3.7 训练工厂初始化（202-313）
- 创建环境、计算训练规模参数：
  - `NUM_ACTORS`、`NUM_UPDATES`、`MINIBATCH_SIZE`。
- `CLIP_EPS` 可选按 agent 数缩放（211-215）。
- 包装 `OvercookedV2LogWrapper`（217）。
- 学习率：
  - warmup + cosine（219-242）。
- 奖励塑形退火：
  - `rew_shaping_anneal`（244-246），前期多用 shaped reward，后期逐渐退掉。
- 初始化 actor/critic 参数与优化器（248-302）。
- 初始化环境和 RNN hidden state（304-313）。

## 3.8 采样阶段 `_env_step`（318-380）
- 读当前 `last_obs/last_done/hstates`。
- Actor 采样动作：
  - `pi.sample`（327）；
  - `log_prob`（328）。
- Critic 估值：
  - 先构造 `world_state`（333）；
  - 再输出 `value`（335）。
- 与环境交互（339-341）。
- 奖励处理（343-355）：
  - `reward + anneal_factor * shaped_reward`。
- 组装 `Transition`（361-371）：
  - `global_done`：给 GAE 用；
  - `done`：给 RNN reset 用（这里存的是 `last_done`）。

## 3.9 GAE 计算（393-414）
- 标准递推：
  - `delta = r + gamma * V(next) * (1-done) - V`。
  - `gae = delta + gamma * lambda * (1-done) * gae`。
- 得到：
  - `advantages`
  - `targets = advantages + value`

## 3.10 更新阶段 `_update_epoch/_update_minbatch`（416-554）
- 打乱 actor 维度并切 minibatch（499-528）。
- Actor loss（421-452）：
  - PPO clip policy loss；
  - 熵正则 `- ENT_COEF * entropy`。
- Critic loss（454-467）：
  - clipped value loss（PPO 风格）。
- actor/critic 分别反向传播和更新（469-485）。
- 记录损失统计（487-496）。

## 3.11 日志与外层扫描（556-591）
- 对 info/loss 做均值统计（561）。
- 记录 `update_step` 与 `env_step`（562-564）。
- 如果开启 wandb，回调记录（566-567）。
- 最外层 `jax.lax.scan` 跑 `NUM_UPDATES` 次（588-590）。

## 3.12 主函数 `main`（596-620）
- 读取 hydra 配置。
- `wandb.init(...)`。
- `jax.jit(make_train(config))` 后，对 `NUM_SEEDS` 做 `vmap` 并行训练。
- `wandb.finish()` 收尾。

---

## 4. 关键数据形状（最常用）

- `obs_batch`：`[NUM_ACTORS, H, W, C]`
- `world_state`：`[NUM_ACTORS, H, W, C * num_agents]`
- `action/value/log_prob/reward`：`[NUM_ACTORS]`
- rollout 后 `traj_batch.xxx`：`[NUM_STEPS, NUM_ACTORS, ...]`

理解这 4 个 shape，基本就能看懂 80% 的逻辑。

---

## 5. 最小启动命令

```bash
python baselines/TARL/mappo_rnn_overcooked_v2.py \
  WANDB_MODE=disabled \
  TOTAL_TIMESTEPS=256 \
  NUM_STEPS=32 \
  NUM_ENVS=2 \
  NUM_MINIBATCHES=2 \
  UPDATE_EPOCHS=1 \
  NUM_SEEDS=1
```

完整训练可直接用默认配置：

```bash
python baselines/TARL/mappo_rnn_overcooked_v2.py
```

---

## 6. 你最可能会改的配置项

在 `baselines/TARL/config/mappo_rnn_overcooked_v2.yaml`：
- 环境相关：`ENV_KWARGS.layout`, `agent_view_size`, `random_agent_positions`
- 训练规模：`NUM_ENVS`, `NUM_STEPS`, `TOTAL_TIMESTEPS`
- PPO 超参：`LR`, `CLIP_EPS`, `ENT_COEF`, `VF_COEF`
- RNN/CNN 宽度：`FC_DIM_SIZE`, `GRU_HIDDEN_DIM`
- shaped reward 退火：`REW_SHAPING_HORIZON`

---

## 7. 常见坑位（排错优先看）

- `NUM_ACTORS * NUM_STEPS` 需要能被 `NUM_MINIBATCHES` 整除，否则 reshape 会报错。
- 如果显存/内存紧张，先减小 `NUM_ENVS`，其次减小 `NUM_STEPS`。
- `WANDB_MODE` 设为 `disabled` 可先纯本地调通。
- 若想更“全局”的 critic，可把 `world_state` 从“观测拼接”改成直接编码 `env_state`（当前实现是观测拼接版）。

---

## 8. 一句话总结

这份代码是“**IPPO overcooked_v2 框架 + MAPPO 的集中式 critic 训练逻辑**”：  
采样和环境管线沿用 overcooked_v2 版本，优化阶段改成 actor/critic 解耦并使用 `world_state` 估值。
