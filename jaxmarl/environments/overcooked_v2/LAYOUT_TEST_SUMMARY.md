# OvercookedV2 布局测试总结

本文档总结了 `layouts.py` 里已注册的每个 layout 主要适合测试什么能力。

说明：
- 这里的“测试目标”是基于地图几何结构、物体摆放和 `layouts.py` 中分类注释做的工程化归纳。
- 一个 layout 往往可用于多种能力测试，下表只强调最主要用途，便于实验设计。

## 术语说明
- **拓扑结构**：走廊/环形/分房间/对称性/路径约束。
- **协作机制**：分工、交接、防碰撞、同步时机。
- **信息因素**：菜谱指示器位置、非对称可见性、具身通信（grounded communication）。
- **任务复杂度**：食材/锅数量、菜谱多样性。

## 布局逐项说明

| Layout | 代码分类 | 主要测试目标 | 菜谱设置（来自注册表） |
|---|---|---|---|
| `cramped_room` | Overcooked-AI | 小空间下的基础协作烹饪与拥挤碰撞处理 | `[[0,0,0]]` |
| `asymm_advantages` | Overcooked-AI | 非对称地图优势下的工位占有与角色专业化 | `[[0,0,0]]` |
| `coord_ring` | Overcooked-AI | 环形路径+较长移动距离下的路径规划与时序协作 | `[[0,0,0]]` |
| `forced_coord` | Overcooked-AI | 受限通路和瓶颈导致的强制协作与更强角色互依 | `[[0,0,0]]` |
| `counter_circuit` | Overcooked-AI | 绕台面运输回路与中心障碍下的多锅物流调度 | `[[0,0,0]]` |
| `cramped_room_v2` | Adapted | 小图上多食材+菜谱指示器条件下的条件决策 | 自动（`Layout.from_string`） |
| `asymm_advantages_recipes_center` | Adapted | 非对称布局且菜谱信息居中时的信息利用能力 | 自动 |
| `asymm_advantages_recipes_right` | Adapted | 菜谱信息偏右，测试右侧信息优势与分工策略 | 自动 |
| `asymm_advantages_recipes_left` | Adapted | 菜谱信息偏左，测试左侧信息优势与分工策略 | 自动 |
| `two_rooms` | Adapted | 分离工作区之间的跨房间协作与路径受限移动 | 自动 |
| `two_rooms_both` | Other | 双房间且资源更丰富时的任务分配与角色弹性 | 自动 |
| `long_room` | Other | 长走廊场景中的远距离运输效率与延迟协作 | `[[0,0,0]]` |
| `fun_coordination` | Other | 多食材菜谱选择（`0/1/2/3`）与菜谱驱动取材 | `[[0,0,2],[1,1,3]]` |
| `more_fun_coordination` | Other | 更复杂配方/锅位协同下的多动作决策协调 | `[[0,1,1],[0,2,2]]` |
| `fun_symmetries` | Other | 对称地图+食材非对称时的对称性打破与协议形成 | `[[0,0,0],[1,1,1]]` |
| `fun_symmetries_plates` | Other | 对称结构+重复盘子资源下的盘子管理策略 | `[[0,0,0],[1,1,1]]` |
| `fun_symmetries1` | Other | 增加墙体后的更难对称瓶颈，测对称破缺鲁棒性 | `[[0,0,0],[1,1,1]]` |
| `grounded_coord_simple` | Extended Cat-Dog | 含 `R/L` 指示器与按钮的具身通信，部分可观测下的角色分工 | `[[0,0,0],[1,1,1]]` |
| `grounded_coord_ring` | Extended Cat-Dog | 环形拓扑上的大尺度具身通信任务，长时序信号与执行 | `[[0,0,0],[1,1,1]]` |
| `test_time_simple` | Test-Time Protocol Formation | 修改通信拓扑后的测试时协议迁移/泛化能力 | `[[0,0,0],[1,1,1]]` |
| `test_time_wide` | Test-Time Protocol Formation | 更宽阔开放地图上的测试时协议泛化能力 | `[[0,0,0],[1,1,1]]` |
| `demo_cook_simple` | Demo Cook | 演示规模地图下的快速定性策略检查 | `[[0,0,0],[1,1,1]]` |
| `demo_cook_wide` | Demo Cook | 更大演示地图下的策略可扩展性与可视化压力测试 | `[[0,0,0],[1,1,1]]` |

## 额外说明：未注册的布局字符串

`layouts.py` 中有 `overcookedv2_demo` 这个模板/示例 ASCII 地图，但默认未加入 `overcooked_v2_layouts` 注册表。

## 实验选型建议

- **算法基本可用性检查**：`cramped_room`、`cramped_room_v2`
- **非对称与专业化分工**：`asymm_advantages*`、`forced_coord`
- **通信/协议研究**：`grounded_coord_simple`、`grounded_coord_ring`、`test_time_*`
- **长时程路径与调度**：`coord_ring`、`long_room`、`counter_circuit`
- **对称性打破能力**：`fun_symmetries*`
