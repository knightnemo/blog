---
title: "Fantastic Mega-Scale RL Recipes and where to find them"
date: 2026-10-11T00:00:00+08:00
draft: false
ShowToc: false
math: true
guideLang: zh
tags: ["reinforcement-learning", "robotics", "sim-to-real", "massively-parallel", "paper-reading"]
summary: "Mega-Scale RL 的论文笔记：batch、多策略、reset 分布，以及 off-policy。"
---

{{< mega-header >}}

{{< mega-overview >}}

## 1. On-Policy，优化 batch {#batch}

### [CoRL'21] Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning  {#rudin}

Comment: 大名鼎鼎的rsl_rl/legged_gym出处

论文：[arXiv:2109.11978](https://arxiv.org/abs/2109.11978)

代码：[legged_gym](https://github.com/leggedrobotics/legged_gym), [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

{{< mega-figure src="rudin-scaling.png" caption="环境数、batch size 与训练时间" paper="2109.11978" >}}

{{< mega-figure src="rudin-bootstrapping.png" caption="超时截断的 value bootstrapping" paper="2109.11978" >}}

> **核心结论**
>
>
> 1. 大 batch size，大 {{< mega-math inline=true >}}N_{\mathrm{env}}{{< /mega-math >}}
>
> 2. 超时截断应该做 value bootstrapping
>
### [ICLR'22] What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study {#andrychowicz}

论文：[arXiv:2006.05990](https://arxiv.org/abs/2006.05990)

> **核心结论**
>
> trick小合集，没有什么特别的结论。

### [ICML'25] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks {#parallel-collection}

论文: [arXiv:2506.03404](https://arxiv.org/abs/2506.03404)

{{< mega-figure src="parallel-collection-training.png" caption="环境数、rollout length 与训练表现" paper="2506.03404" >}}

{{< mega-figure src="parallel-collection-environments.png" caption="更多环境与更短 rollout 的比较" paper="2506.03404" >}}

> **核心结论**
>
> 并行 on-policy RL 里，batch 大小 {{< mega-math inline=true >}}|B| = N_{\mathrm{envs}} \times N_{\mathrm{RO}}{{< /mega-math >}}。其中用**更多环境，更少 rollout length**比反之更好

### [NIPS'25] Staggered Environment Resets Improve Massively Parallel On-Policy Reinforcement Learning {#staggered-resets}

论文：[arXiv:2511.21011](https://arxiv.org/abs/2511.21011)

代码：[staggered-resets](https://github.com/siddharthbharthulwar/staggered-resets)

{{< mega-figure src="staggered-resets-schematic.png" caption="同步与错峰重置的数据分布" paper="2511.21011" >}}

{{< mega-figure src="staggered-resets-results.png" caption="错峰重置的实验结果" paper="2511.21011" >}}

但如果所有环境同时在 t=0 开始、同时在 t=H 超时重置，第 j 次更新的 batch 就**只包含** {{< mega-math inline=true >}}[(j-1)K,  jK-1]{{< /mega-math >}} 这一小段时间窗的状态：一批全是"刚开局"，下一批全是"抓取中"，……过 {{< mega-math inline=true >}}\lceil H/K \rceil{{< /mega-math >}} 次更新后又全部跳回开局。作者称之为**周期性批次非平稳**（cyclical nonstationarity）。后果是 critic 一直在追一个循环变化的数据分布，学了后段忘了前段（灾难性遗忘）。

> **核心结论**
>
> 应该初始reset的时候就把distribution错开

## 2. On-Policy，优化怎么用这个 fleet {#fleet}

### [RSS'23] DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training {#dexpbt}

论文：[arXiv:2305.12127](https://arxiv.org/abs/2305.12127)

代码：[Isaac Lab PBT](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/population_based_training.html)

手臂+多指手的任务探索难，同时跑 8–32 个 PPO 训练（每个占一张 GPU、各 8192 个环境），定期把差的复制成好的、并随机扰动超参数和奖励系数，用"种群 + 进化"来解决高维灵巧操作里 PPO 单次运行容易卡在局部最优的问题。

{{< mega-figure src="dexpbt-method.png" caption="DexPBT 的种群训练流程" paper="2305.12127" >}}

{{< mega-figure src="dexpbt-results.png" caption="有无 PBT 的双臂任务表现" paper="2305.12127" >}}

{{< mega-figure src="dexpbt-hyperparameters.png" caption="种群规模与超参数变化" paper="2305.12127" >}}

> **核心结论**
>
> PPO 是抽奖，尤其是高维度 task，多抽抽奖然后把 work 的多铺开（外环类似遗传算法）。

### [ICML'24] SAPG：Split and Aggregate Policy Gradients {#sapg}

论文：[arXiv:2407.20230](https://arxiv.org/abs/2407.20230)
代码：[sapg](https://github.com/jayeshs999/sapg)

PPO 的 {{< mega-math inline=true >}}N_{\mathrm{env}}{{< /mega-math >}} 大到一定程度之后，batch size 增大到一定程度就饱和了 （蓝:PPO, 红: SAPG）

{{< mega-figure src="sapg-saturation.png" caption="PPO 的 batch size 饱和现象" paper="2407.20230" >}}

作者给的解释是：所有环境都从同一个高斯策略采样，动作大多落在均值附近，环境越多，重复数据越多，"多出来的环境被浪费了"

**方法**：

1. **拆分**：N 个环境平均分给 M 个策略 {{< mega-math inline=true >}}\pi_1\ldots\pi_M{{< /mega-math >}}（论文用 M=6、N=24576，即每个策略 4096 个环境）。所有策略共享 actor 主干 {{< mega-math inline=true >}}B_\theta{{< /mega-math >}} 和 critic 主干 {{< mega-math inline=true >}}C_\psi{{< /mega-math >}}，第 j 个策略额外有一个只属于自己的可学习向量 {{< mega-math inline=true >}}\phi_j{{< /mega-math >}}（复杂任务 {{< mega-math inline=true >}}\phi_j\in\mathbb R^{32}{{< /mega-math >}}，简单任务 {{< mega-math inline=true >}}\mathbb R^{16}{{< /mega-math >}}），作为网络输入条件。{{< mega-math inline=true >}}\phi_j{{< /mega-math >}} 只被自己的损失更新，{{< mega-math inline=true >}}\theta/\psi{{< /mega-math >}} 被所有损失更新。

2. **leader–follower 聚合**：{{< mega-math inline=true >}}\pi_1{{< /mega-math >}} 是 leader，{{< mega-math inline=true >}}\pi_2\ldots\pi_M{{< /mega-math >}} 是 follower。

    - follower：只用自己块内的 on-policy 数据做普通 PPO。

    - leader：on-policy PPO 损失 + {{< mega-math inline=true >}}\lambda\cdot{{< /mega-math >}}off-policy 损失，{{< mega-math inline=true >}}\lambda=1{{< /mega-math >}}。off-policy 数据从 follower 的数据并集中**下采样到与 leader 自己的数据同样多**（{{< mega-math inline=true >}}|D'_1|=|D_1|{{< /mega-math >}}），避免噪声更大的 off-policy 梯度淹没 on-policy 梯度。

    - off-policy 截断代理目标（来自 Meng et al. 2023 的 off-policy PPO）：

        {{< mega-math >}}
        \begin{aligned}
        L_{\mathrm{off}}(\pi_i;X)
        &= \frac{1}{|X|}\sum_{j\in X}\mathbb E_{(s,a)\sim\pi_j}
        \left[\min\left(rA, \operatorname{clip}\left(r,\mu(1-\epsilon),\mu(1+\epsilon)\right)A\right)\right],\\
        r &= \frac{\pi_i(a\mid s)}{\pi_j(a\mid s)},\qquad
        \mu = \frac{\pi_{i,\mathrm{old}}(a\mid s)}{\pi_j(a\mid s)}.
        \end{aligned}
        {{< /mega-math >}}

        把 clip 区间以“旧策略相对于行为策略的比值”为中心平移，{{< mega-math inline=true >}}A{{< /mega-math >}} 是 leader 旧 critic 估计的优势。当 {{< mega-math inline=true >}}j=i{{< /mega-math >}} 时退化为普通 PPO。

    - critic：on-policy 数据用 3-step return 目标，off-policy 数据只用 1-step 目标：

        {{< mega-math >}}
        \begin{aligned}
        V^{\mathrm{tgt}}_{\mathrm{on}}
        &=\sum_{k=t}^{t+2}\gamma^{k-t}r_k+\gamma^3V_{\mathrm{old}}(s_{t+3}),\\
        V^{\mathrm{tgt}}_{\mathrm{off}}
        &=r_t+\gamma V_{\mathrm{old}}(s_{t+1}).
        \end{aligned}
        {{< /mega-math >}}

        总 critic 损失 = on + {{< mega-math inline=true >}}\lambda\cdot{{< /mega-math >}}off。

3. **多样性**：除了 {{< mega-math inline=true >}}\phi_j{{< /mega-math >}} 条件化，可选给 follower 加不同强度的熵正则：第 i 个策略的熵系数 = {{< mega-math inline=true >}}\lambda_{\mathrm{ent}}(i-1){{< /mega-math >}}，leader 不加熵；此时每个块有自己的可学习 σ 向量。{{< mega-math inline=true >}}\lambda_{\mathrm{ent}}{{< /mega-math >}} 从 {{< mega-math inline=true >}}\{0,0.003,0.005\}{{< /mega-math >}} 中挑（Reorientation、ShadowHand 用 0.005，其余 0）。

4. 每次迭代：所有块各 rollout 16 步（AllegroKuka）→ 拼成一个总损失 → 对 θ, ψ, φ 一起做梯度步（Algorithm 1）。

{{< mega-figure src="sapg-method.png" caption="SAPG 的 leader–follower 架构" paper="2407.20230" >}}

{{< mega-figure src="sapg-results.png" caption="SAPG 的任务表现" paper="2407.20230" >}}

> **核心结论**
>
> 在 on-policy 算法中避免重复采样能提升策略表现。

### [arxiv'25] EPO：Evolutionary Policy Optimization {#epo}

论文：[arXiv:2503.19037](https://arxiv.org/abs/2503.19037)

代码：[EPO](https://github.com/YifanSu1301/EPO)

在 SAPG 的"leader 吸收 follower 数据"框架上，把 follower 数量扩到 64 个，并对 follower 的潜变量 φ 做遗传算法（淘汰差的、交叉+变异好的），让"多策略"能随环境数一起扩展。

{{< mega-figure src="epo-method.png" caption="EPO 的进化与策略更新" paper="2503.19037" >}}

{{< mega-figure src="epo-results.png" caption="EPO 的任务表现" paper="2503.19037" >}}

{{< mega-figure src="epo-scaling.png" caption="策略数与环境数的扩展" paper="2503.19037" >}}

## 3. On-Policy，优化怎么做 reset {#reset}

### [ICLR'26] OmniReset {#omnireset}

论文: [arXiv:2603.15789](https://arxiv.org/abs/2603.15789)

代码：[UWLab](https://github.com/UW-Lab/UWLab)

标准 PPO 在大规模并行下会「饱和」——env 再多，也只是在同一片狭窄状态区域里重复采样，长程任务里成功信号太稀有，策略困在局部最优（如只学会伸手）。

方法：程序化生成的多样重置状态 + 大量并行 env 的 PPO

{{< mega-figure src="omnireset-method.png" caption="OmniReset 的训练与迁移流程" paper="2603.15789" >}}

{{< mega-figure src="omnireset-resets.png" caption="不同重置状态与环境规模" paper="2603.15789" >}}

{{< mega-figure src="omnireset-results.png" caption="OmniReset 的实验结果" paper="2603.15789" >}}

> **核心结论**
>
> 大N_env是对的，避免重复采样是对的，通过手动调整reset distribution来去增广coverage/调整buffer里的transition distribution是对的

### [CoRL'26] SGS-RL {#sgs}

A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control

论文：[arXiv:2610.12465](https://arxiv.org/abs/2610.12465)

有了 OmniReset 式的多样重置后，均匀采样会把越来越多 env 花在「已经掌握」或「暂时做不到」的配置上，这些配置对 PPO 梯度没有贡献（优势函数接近 0）。env 越多，浪费的绝对量越大，所以「加 env」的收益消失甚至变差。

Related: DemoStart Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots ([arXiv:2409.06613](https://arxiv.org/abs/2409.06613)) 是从demonstration start的地方filter掉之前从这里开始的全成功/全失败，是一个这个idea在另一个problem setup下的极端/退化版本。

方法：

- 训练前从任务分布采一个固定集合 N 个配置 {{< mega-math inline=true >}}\tau_i=(s_0,g,e){{< /mega-math >}}（初始状态、目标、环境）；操作任务 N = 32,768，运动 N = 104,000。

- 每个配置维护最近 H = 100 次的成败（环形缓冲），成功率 {{< mega-math inline=true >}}p_i{{< /mega-math >}} 取均值。

- 打分（Beta 核的「众数-集中度」形式，目标成功率 t、集中度 κ）：

    {{< mega-math >}}
    \begin{aligned}
    w_i&=(p_i+\epsilon)^{\kappa t}(1-p_i+\epsilon)^{\kappa(1-t)},\\
    \ell_i&=\log(\max(w_i,\epsilon)+\epsilon).
    \end{aligned}
    {{< /mega-math >}}

- 每个 episode 结束时按 softmax 采下一个配置：

    {{< mega-math >}}
    p(i)=\frac{\exp(\ell_i/T)}{\sum_j\exp(\ell_j/T)},\qquad
    T_{\mathrm{eff}}=\max(T,1)\quad\text{(implementation)}.
    {{< /mega-math >}}

- 超参：H = 100，T = 2；操作任务 t = 0.5, κ = 1, ε = 1e-4；运动任务 t = 0.66, κ = 5, ε = 1e-8。ε 保证每个配置都有非零概率，从而能不断更新成功率估计。

- 其它配合：每个领域共用一个奖励（终局成功 + 小的正则项）；PPO 的 mini-batch 数固定，所以 batch 随 env 数等比例增大。

- 操作任务的重置池沿用 OmniReset 三类：Reaching、Stable Grasp、Near-Goal，各占三分之一，预先碰撞检查。

- 和已有方法的关系：与 Sampling for Learnability（SFL，按 {{< mega-math inline=true >}}p(1-p){{< /mega-math >}} 打分）同一思想，区别是滑动窗口成功率、可调目标成功率、所有配置保底概率；对比的 PLR 用 {{< mega-math inline=true >}}|\mathrm{GAE}|{{< /mega-math >}} 作分数。

{{< mega-figure src="sgs-sampling.png" caption="SGS 对成功率的采样权重" paper="2610.12465" >}}

{{< mega-figure src="sgs-scaling.png" caption="SGS 随环境数的扩展" paper="2610.12465" >}}

> **核心结论**
>
> 过难/过容易的states都没learning signal，这相当于是adaptive sampling 在难度适中的state上做reset

### [CoRL'24] DextrAH-G / DextrAH-RGB {#dextrah}

DextrAH-G: Pixels-to-Action Dexterous Arm-Hand Grasping with Geometric Fabrics

- arXiv：[arXiv:2407.02274](https://arxiv.org/abs/2407.02274)

DextrAH-RGB: Visuomotor Policies to Grasp Anything with Dexterous Hands

- arXiv：[arXiv:2412.01791](https://arxiv.org/abs/2412.01791)

- 代码：[DEXTRAH](https://github.com/NVlabs/DEXTRAH) （「DextrAH on Isaac Lab」，含特权 RL + 在线蒸馏）

臂+多指手的高维动作直接用关节空间 PPO：训练慢、学出不自然的抓法（夹在中指和无名指之间）、多物体根本学不会、真机不安全。

方法：把 23 维关节动作换成 11 维「掌位姿 + 手部 PCA 协同」动作，并负责避碰/关节限位；

{{< mega-figure src="dextrah-method.png" caption="DextrAH-G 的训练与蒸馏流程" paper="2407.02274" >}}

> **核心结论**
>
> 高维度动作空间 => 降维度，可以有效提升学习速度, Dexterous Functional Grasping (EigenGrasp [https://arxiv.org/pdf/2312.02975](https://arxiv.org/pdf/2312.02975) 也是类似idea)

后续工作: EigenDEXplore ([EigenDEXplore](https://eigendexplore.github.io/)) 使用eigenvectors帮助exploration，稍微constraint soft 一些

## 4. Off-Policy 罗列 {#off-policy}

MEGA-Scale目前主要还是on-policy, off-policy最近尤其是在FlashSAC之后让大家开始重新提起兴趣考虑，简单罗列代表性工作和核心insight：

### [ICML'23] PQL — Parallel Q-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation {#pql}

把 DDPG 拆成三个并行进程（采数据的 Actor、学 Q 的 V-learner、学策略的 P-learner），在单台工作站上让 off-policy 吃满上万个 Isaac Gym 环境，墙钟时间比 PPO 快、样本效率也更高。

arXiv：[arXiv:2307.12983](https://arxiv.org/abs/2307.12983)

代码：[pql](https://github.com/Improbable-AI/pql)

> **核心结论**
>
> 每个 env 用不同探索强度可以试试，

### [ICLR'25] PQN: Simplifying Deep Temporal Difference Learning {#pqn}

arXiv: [arXiv:2407.04811](https://arxiv.org/abs/2407.04811)

代码: [purejaxql](https://github.com/mttga/purejaxql)

证明 LayerNorm（+ 少量 ℓ2）能让 TD 学习在无目标网络、无回放池时也收敛；于是把 DQN 简化成「大量并行 env + 小段 λ-回报 + 像 PPO 一样分 mini-batch 更新」的纯 GPU Q 学习。

> **核心结论**
>
> layernorm, lambda-回报都挺重要

### [arxiv'25] FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control {#fasttd3}

arXiv[arXiv:2505.22642](https://arxiv.org/abs/2505.22642)

代码[FastTD3](https://github.com/younggyoseo/FastTD3)

不发明新东西，把 PQL 的经验做成一个同步、简单、调好参数的 TD3：并行仿真 +**超大 batch**（32768）+ **categorical critic** + **CDQ**，单张 A100 3 小时内解 HumanoidBench 多数任务，并支持 IsaacLab / MuJoCo Playground。

> **核心结论**
>
> 大batch和categorical critic是对的

### [arxiv'25] FastSAC: Learning Sim-to-Real Humanoid Locomotion in 15 Minutes {#fastsac}

arXiv[arXiv:2512.01996](https://arxiv.org/abs/2512.01996)

代码[holosoma](https://github.com/amazon-far/holosoma)

把 FastTD3 论文里不稳定的 FastSAC 调稳，配合极简奖励，在单张 RTX 4090 上 15 分钟训出带强域随机化、可上真机的 G1/T1 全身（29 自由度）行走策略；全身动作跟踪用 4×L40S、16384 env 也比 PPO 快。

> **核心结论**
>
> LayerNorm，取 Q-mean 而不是 CDQ，自动 alpha。

### [RSS'26] FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control {#flashsac}

- **arXiv**：[arXiv:2604.04539](https://arxiv.org/abs/2604.04539)

- **代码**：[FlashSAC](https://github.com/Holiday-Robot/FlashSAC)

> **核心结论**
>
> 用更大的网络（2.5M 参数、6 层）、更大 batch、更大回放池、**更少的梯度更新（UTD = 2/1024）**，再用 BatchNorm/RMSNorm/权重投影/分布式 critic 把各种范数约束住，让 SAC 在高维任务上既快又稳。

后续工作： **WarpSAC: Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation**

- **arXiv**：[arXiv:2608.24479](https://arxiv.org/abs/2608.24479)

- **代码**：[warprl](https://github.com/wzhhasadream/warprl)

> **核心结论**
>
> GPU 大规模并行时**关掉权重投影归一化、只用单个 Q** 反而更好
