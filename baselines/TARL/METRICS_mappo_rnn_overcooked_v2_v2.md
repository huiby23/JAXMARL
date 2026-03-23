# MAPPO Overcooked_v2 指标说明

对应脚本：`baselines/TARL/mappo_rnn_overcooked_v2_v2.py`  
本文档说明训练日志里的主要指标含义，以及实际看算法表现时的关注重点。

## 1. 奖励与任务指标

- `returned_episode_returns`
  - 含义：一个 episode 结束时的累计回报（由 `OvercookedV2LogWrapper` 记录）。
  - 用途：最核心任务表现指标之一，建议作为主对比指标。

- `returned_episode_lengths`
  - 含义：episode 长度。
  - 用途：辅助判断策略是否更快完成有效行为，或是否拖延。

- `returned_episode_recipe_returns`
  - 含义：按 recipe 统计的 episode 回报。
  - 用途：看策略是否偏科（某些菜谱学得好，某些差）。

- `original_reward`
  - 含义：环境原始奖励（未加入 shaped reward）。
  - 用途：更接近最终任务目标，后期评估价值更高。

- `shaped_reward`
  - 含义：环境给的 shaped reward（密集奖励信号）。
  - 用途：看学习速度很有用，尤其训练前中期。

- `anneal_factor`
  - 含义：reward shaping 退火系数（从 1 逐步降到 0）。
  - 用途：理解为何同样行为在不同时期产生不同训练奖励。

- `combined_reward`
  - 含义：训练实际使用的奖励，公式：
  - `combined_reward = original_reward + anneal_factor * shaped_reward`
  - 用途：解释优化器真正收到的 reward 信号。

- `returned_episode`
  - 含义：该步是否发生 episode 结束（bool）。
  - 用途：筛选“只有 episode 结束时才有意义”的统计。

## 2. PPO/MAPPO 优化诊断指标（`metric["loss"]`）

- `policy_loss`
  - 含义：PPO clip 后的策略损失（不含熵正则项）。
  - 用途：观察策略目标是否持续优化。

- `entropy`
  - 含义：策略熵。
  - 用途：看探索强度；过快下降可能过早收敛。

- `actor_loss`
  - 含义：`policy_loss - ENT_COEF * entropy`。
  - 用途：actor 真正优化目标。

- `value_loss`
  - 含义：critic 的 clipped value loss。
  - 用途：看 critic 是否在稳定拟合回报目标。

- `total_loss`
  - 含义：总损失（actor + critic 加权组合）。
  - 用途：总览训练优化趋势，单独解释能力较弱。

- `ratio`
  - 含义：`exp(logp_new - logp_old)` 的均值。
  - 用途：看策略更新幅度，长期非常接近 1 往往说明更新偏小。

- `approx_kl`
  - 含义：近似 KL，用于衡量新旧策略差异。
  - 用途：判断步长是否合理，过大可能不稳定，过小可能学得慢。

- `clip_frac`
  - 含义：PPO 裁剪触发比例。
  - 用途：0 附近长期不变通常表示约束太紧或更新太小；太高表示更新太激进。

## 3. 训练进度指标

- `update_step`
  - 含义：第几个 PPO update。

- `env_step`
  - 含义：累计环境步数，`update_step * NUM_STEPS * NUM_ENVS`。
  - 用途：跨配置对比时优先用 `env_step` 对齐横轴。

## 4. 看“算法性能”建议重点关注哪些指标

建议按优先级看：

1. `returned_episode_returns`
   - 主指标，判断最终任务性能。
2. `shaped_reward`
   - 学习速度指标，前中期最有参考价值。
3. `original_reward`
   - 后期更接近真实任务目标，建议和 `shaped_reward` 同时看。
4. `returned_episode_recipe_returns`
   - 看是否存在 recipe 维度偏科问题。
5. `approx_kl + clip_frac + entropy + value_loss`
   - 训练稳定性诊断四件套，出现性能异常时优先排查。

## 5. 实战看板（推荐）

- 性能主图：
  - `returned_episode_returns` vs `env_step`
  - `shaped_reward` vs `env_step`
  - `original_reward` vs `env_step`

- 稳定性主图：
  - `approx_kl` vs `env_step`
  - `clip_frac` vs `env_step`
  - `entropy` vs `env_step`
  - `value_loss` vs `env_step`

- 诊断补充：
  - `returned_episode_recipe_returns`（按 recipe 分组）

## 6. 一句话总结

看“最终好不好”主要看 `returned_episode_returns`；  
看“学得快不快”主要看 `shaped_reward`；  
看“为什么表现变差”优先看 `approx_kl / clip_frac / entropy / value_loss`。
