---
title: "Fantastic Mega-Scale RL Recipes and where to find them"
date: 2026-10-11T00:00:00+08:00
draft: false
ShowToc: false
math: true
guideLang: en
tags: ["reinforcement-learning", "robotics", "sim-to-real", "massively-parallel", "paper-reading"]
summary: "Reading notes on Mega-Scale RL: batches, multiple policies, reset distributions and off-policy methods."
---

{{< mega-header >}}

{{< mega-overview >}}

## 1. On-Policy: getting the batch right {#batch}

### [CoRL'21] Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning {#rudin}

Comment: This is where the well-known rsl_rl / legged_gym comes from.

Paper: [arXiv:2109.11978](https://arxiv.org/abs/2109.11978)

Code: [legged_gym](https://github.com/leggedrobotics/legged_gym) · [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

{{< mega-figure src="rudin-scaling.png" caption="Environment count, batch size and training time" paper="2109.11978" >}}

{{< mega-figure src="rudin-bootstrapping.png" caption="Value bootstrapping at time-limit truncation" paper="2109.11978" >}}

> **Main takeaway**
>
>
> 1. Large batch size, large {{< mega-math inline=true >}}N_{\mathrm{env}}{{< /mega-math >}}.
> 2. Time-limit truncation should use value bootstrapping.
>
### [ICLR'22] What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study {#andrychowicz}

Paper: [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)

> **Main takeaway**
>
> A collection of tricks. I don't see one standout conclusion.

### [ICML'25] The Impact of On-Policy Parallelized Data Collection on Deep Reinforcement Learning Networks {#parallel-collection}

Paper: [arXiv:2506.03404](https://arxiv.org/abs/2506.03404)

{{< mega-figure src="parallel-collection-training.png" caption="Environment count, rollout length and training performance" paper="2506.03404" >}}

{{< mega-figure src="parallel-collection-environments.png" caption="Comparing more environments with shorter rollouts" paper="2506.03404" >}}

> **Main takeaway**
>
> In parallel on-policy RL, batch size is {{< mega-math inline=true >}}|B|=N_{\mathrm{envs}}\times N_{\mathrm{RO}}{{< /mega-math >}}. **More environments and shorter rollouts** work better than the reverse allocation.

### [NIPS'25] Staggered Environment Resets Improve Massively Parallel On-Policy Reinforcement Learning {#staggered-resets}

Paper: [arXiv:2511.21011](https://arxiv.org/abs/2511.21011)

Code: [staggered-resets](https://github.com/siddharthbharthulwar/staggered-resets)

{{< mega-figure src="staggered-resets-schematic.png" caption="Batch distributions with synchronous and staggered resets" paper="2511.21011" >}}

{{< mega-figure src="staggered-resets-results.png" caption="Staggered-reset results" paper="2511.21011" >}}

If all environments start together at {{< mega-math inline=true >}}t=0{{< /mega-math >}} and hit the time limit together at {{< mega-math inline=true >}}t=H{{< /mega-math >}}, the batch for update {{< mega-math inline=true >}}j{{< /mega-math >}} **only contains** states in the short window {{< mega-math inline=true >}}[(j-1)K,\;jK-1]{{< /mega-math >}}. One batch is entirely “just started,” the next is entirely “in the middle of grasping,” and so on. After {{< mega-math inline=true >}}\lceil H/K\rceil{{< /mega-math >}} updates, everything jumps back to the beginning. The authors call this **cyclical nonstationarity**. The critic keeps chasing a changing distribution, learning later stages and forgetting earlier ones (catastrophic forgetting).

> **Main takeaway**
>
> Stagger the distribution at the initial reset.

## 2. On-Policy: making use of the fleet {#fleet}

### [RSS'23] DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training {#dexpbt}

Paper: [arXiv:2305.12127](https://arxiv.org/abs/2305.12127)

Code: [Population Based Training in Isaac Lab](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/features/population_based_training.html)

Exploration is difficult in arm-and-multifinger-hand tasks. Run 8–32 PPO trainings at once, each on one GPU with 8,192 environments. Periodically copy good runs into bad ones and randomly perturb hyperparameters and reward coefficients. A population with evolution helps address the local optima that a single PPO run can get stuck in on high-dimensional dexterous manipulation.

{{< mega-figure src="dexpbt-method.png" caption="DexPBT population training" paper="2305.12127" >}}

{{< mega-figure src="dexpbt-results.png" caption="Dual-arm performance with and without PBT" paper="2305.12127" >}}

{{< mega-figure src="dexpbt-hyperparameters.png" caption="Population size and changes in hyperparameters" paper="2305.12127" >}}

> **Main takeaway**
>
> PPO is a lottery, especially on high-dimensional tasks. Buy more tickets, then give the runs that work more room. The outer loop is much like a genetic algorithm.

### [ICML'24] SAPG: Split and Aggregate Policy Gradients {#sapg}

Paper: [arXiv:2407.20230](https://arxiv.org/abs/2407.20230)

Code: [sapg](https://github.com/jayeshs999/sapg)

Once PPO's {{< mega-math inline=true >}}N_{\mathrm{env}}{{< /mega-math >}} gets large enough, increasing batch size further saturates performance (blue: PPO; red: SAPG).

{{< mega-figure src="sapg-saturation.png" caption="PPO performance saturates with batch size" paper="2407.20230" >}}

The authors' explanation is that all environments sample from the same Gaussian policy, so most actions stay near the mean. More environments produce more redundant data: the extra environments are wasted.

**Method:**

1. **Split:** Divide {{< mega-math inline=true >}}N{{< /mega-math >}} environments evenly among {{< mega-math inline=true >}}M{{< /mega-math >}} policies {{< mega-math inline=true >}}\pi_1\ldots\pi_M{{< /mega-math >}}. The paper uses {{< mega-math inline=true >}}M=6{{< /mega-math >}} and {{< mega-math inline=true >}}N=24576{{< /mega-math >}}, giving each policy 4,096 environments. All policies share actor trunk {{< mega-math inline=true >}}B_\theta{{< /mega-math >}} and critic trunk {{< mega-math inline=true >}}C_\psi{{< /mega-math >}}. Policy {{< mega-math inline=true >}}j{{< /mega-math >}} also has its own learnable vector {{< mega-math inline=true >}}\phi_j{{< /mega-math >}}, used as an input condition: {{< mega-math inline=true >}}\phi_j\in\mathbb R^{32}{{< /mega-math >}} for complex tasks and {{< mega-math inline=true >}}\mathbb R^{16}{{< /mega-math >}} for simple tasks. Only its own loss updates {{< mega-math inline=true >}}\phi_j{{< /mega-math >}}; all losses update {{< mega-math inline=true >}}\theta{{< /mega-math >}} and {{< mega-math inline=true >}}\psi{{< /mega-math >}}.

2. **Leader–follower aggregation:** {{< mega-math inline=true >}}\pi_1{{< /mega-math >}} is the leader; {{< mega-math inline=true >}}\pi_2\ldots\pi_M{{< /mega-math >}} are followers.

    - Followers use ordinary PPO on their own block's on-policy data.

    - The leader uses an on-policy PPO loss plus {{< mega-math inline=true >}}\lambda{{< /mega-math >}} times an off-policy loss, with {{< mega-math inline=true >}}\lambda=1{{< /mega-math >}}. Off-policy data from the followers' combined data are **subsampled to match the amount of the leader's own data**, {{< mega-math inline=true >}}|D'_1|=|D_1|{{< /mega-math >}}, so noisier off-policy gradients do not overwhelm the on-policy gradients.

    - The clipped off-policy surrogate comes from Meng et al. (2023)'s off-policy PPO:

        {{< mega-math >}}
        \begin{aligned}
        L_{\mathrm{off}}(\pi_i;X)
        &=\frac{1}{|X|}\sum_{j\in X}\mathbb E_{(s,a)\sim\pi_j}
        \left[\min\left(rA,\;\operatorname{clip}\left(r,\mu(1-\epsilon),\mu(1+\epsilon)\right)A\right)\right],\\
        r&=\frac{\pi_i(a\mid s)}{\pi_j(a\mid s)},\qquad
        \mu=\frac{\pi_{i,\mathrm{old}}(a\mid s)}{\pi_j(a\mid s)}.
        \end{aligned}
        {{< /mega-math >}}

        The clipping interval is shifted around the ratio of the old policy to the behavior policy. {{< mega-math inline=true >}}A{{< /mega-math >}} is the advantage estimated by the leader's old critic. When {{< mega-math inline=true >}}j=i{{< /mega-math >}}, this reduces to ordinary PPO.

    - The critic uses a 3-step return target for on-policy data and a 1-step target for off-policy data:

        {{< mega-math >}}
        \begin{aligned}
        V^{\mathrm{tgt}}_{\mathrm{on}}
        &=\sum_{k=t}^{t+2}\gamma^{k-t}r_k+\gamma^3V_{\mathrm{old}}(s_{t+3}),\\
        V^{\mathrm{tgt}}_{\mathrm{off}}
        &=r_t+\gamma V_{\mathrm{old}}(s_{t+1}).
        \end{aligned}
        {{< /mega-math >}}

        Total critic loss = on + {{< mega-math inline=true >}}\lambda\cdot{{< /mega-math >}}off.

3. **Diversity:** Besides conditioning on {{< mega-math inline=true >}}\phi_j{{< /mega-math >}}, followers can optionally receive different strengths of entropy regularization. Policy {{< mega-math inline=true >}}i{{< /mega-math >}} has entropy coefficient {{< mega-math inline=true >}}\lambda_{\mathrm{ent}}(i-1){{< /mega-math >}}; the leader has none. Each block then has its own learnable {{< mega-math inline=true >}}\sigma{{< /mega-math >}} vector. Choose {{< mega-math inline=true >}}\lambda_{\mathrm{ent}}{{< /mega-math >}} from {{< mega-math inline=true >}}\{0,0.003,0.005\}{{< /mega-math >}}: Reorientation and ShadowHand use 0.005; the others use 0.

4. Each iteration: every block rolls out 16 steps (AllegroKuka), the losses are combined, and {{< mega-math inline=true >}}\theta{{< /mega-math >}}, {{< mega-math inline=true >}}\psi{{< /mega-math >}} and {{< mega-math inline=true >}}\phi{{< /mega-math >}} are updated together (Algorithm 1).

{{< mega-figure src="sapg-method.png" caption="SAPG's leader–follower architecture" paper="2407.20230" >}}

{{< mega-figure src="sapg-results.png" caption="SAPG task performance" paper="2407.20230" >}}

> **Main takeaway**
>
> Avoiding redundant sampling in on-policy algorithms improves policy performance.

### [arxiv'25] EPO: Evolutionary Policy Optimization {#epo}

Paper: [arXiv:2503.19037](https://arxiv.org/abs/2503.19037)

Code: [EPO](https://github.com/YifanSu1301/EPO)

Building on SAPG's framework where the leader absorbs follower data, EPO increases the number of followers to 64 and applies a genetic algorithm to their latent vectors {{< mega-math inline=true >}}\phi{{< /mega-math >}}: discard poor ones, then cross and mutate good ones. This lets the number of policies scale with the environment count.

{{< mega-figure src="epo-method.png" caption="Evolution and policy updates in EPO" paper="2503.19037" >}}

{{< mega-figure src="epo-results.png" caption="EPO task performance" paper="2503.19037" >}}

{{< mega-figure src="epo-scaling.png" caption="Scaling policy and environment counts" paper="2503.19037" >}}

## 3. On-Policy: choosing how to reset {#reset}

### [ICLR'26] OmniReset {#omnireset}

Paper: [arXiv:2603.15789](https://arxiv.org/abs/2603.15789)

Code: [UWLab](https://github.com/UW-Lab/UWLab)

Standard PPO saturates under massive parallelism: more environments keep sampling the same narrow part of the state space. Success signals are too rare in long-horizon tasks, and the policy gets stuck in local optima—for example, only learning to reach.

Method: Procedurally generated diverse reset states + PPO with many parallel environments.

{{< mega-figure src="omnireset-method.png" caption="OmniReset training and transfer" paper="2603.15789" >}}

{{< mega-figure src="omnireset-resets.png" caption="Reset states and environment scales" paper="2603.15789" >}}

{{< mega-figure src="omnireset-results.png" caption="OmniReset results" paper="2603.15789" >}}

> **Main takeaway**
>
> Large {{< mega-math inline=true >}}N_{\mathrm{env}}{{< /mega-math >}} makes sense. Avoiding redundant sampling makes sense. Manually adjusting the reset distribution to increase coverage and change the transition distribution in the buffer makes sense.

### [CoRL'26] SGS-RL {#sgs}

A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control

Paper: [arXiv:2610.12465](https://arxiv.org/abs/2610.12465)

After introducing OmniReset-style diverse resets, uniform sampling spends more and more environments on configurations the policy has already mastered or cannot yet solve. These configurations contribute little to the PPO gradient (advantages are near zero). More environments mean more wasted experience in absolute terms, so the benefit of adding environments disappears or even reverses.

Related: [DemoStart: Demonstration-led auto-curriculum applied to sim-to-real with multi-fingered robots](https://arxiv.org/abs/2409.06613) filters out demonstration starting states from which previous attempts were all successes or all failures. This is an extreme / degenerate version of the same idea under a different problem setup.

Method:

- Before training, sample a fixed set of {{< mega-math inline=true >}}N{{< /mega-math >}} configurations {{< mega-math inline=true >}}\tau_i=(s_0,g,e){{< /mega-math >}} from the task distribution: initial state, goal and environment. {{< mega-math inline=true >}}N=32768{{< /mega-math >}} for manipulation and {{< mega-math inline=true >}}N=104000{{< /mega-math >}} for locomotion.

- For each configuration, keep the outcomes of the last {{< mega-math inline=true >}}H=100{{< /mega-math >}} attempts in a ring buffer. Their mean gives success rate {{< mega-math inline=true >}}p_i{{< /mega-math >}}.

- Score configurations with a Beta kernel in mode–concentration form, using target success rate {{< mega-math inline=true >}}t{{< /mega-math >}} and concentration {{< mega-math inline=true >}}\kappa{{< /mega-math >}}:

    {{< mega-math >}}
    \begin{aligned}
    w_i&=(p_i+\epsilon)^{\kappa t}(1-p_i+\epsilon)^{\kappa(1-t)},\\
    \ell_i&=\log(\max(w_i,\epsilon)+\epsilon).
    \end{aligned}
    {{< /mega-math >}}

- At the end of each episode, sample the next configuration with a softmax. The implementation uses {{< mega-math inline=true >}}T_{\mathrm{eff}}=\max(T,1){{< /mega-math >}}:

    {{< mega-math >}}
    p(i)=\frac{\exp(\ell_i/T)}{\sum_j\exp(\ell_j/T)},\qquad
    T_{\mathrm{eff}}=\max(T,1)\quad\text{(implementation)}.
    {{< /mega-math >}}

- Hyperparameters: {{< mega-math inline=true >}}H=100{{< /mega-math >}}, {{< mega-math inline=true >}}T=2{{< /mega-math >}}. Manipulation uses {{< mega-math inline=true >}}t=0.5{{< /mega-math >}}, {{< mega-math inline=true >}}\kappa=1{{< /mega-math >}}, {{< mega-math inline=true >}}\epsilon=10^{-4}{{< /mega-math >}}; locomotion uses {{< mega-math inline=true >}}t=0.66{{< /mega-math >}}, {{< mega-math inline=true >}}\kappa=5{{< /mega-math >}}, {{< mega-math inline=true >}}\epsilon=10^{-8}{{< /mega-math >}}. {{< mega-math inline=true >}}\epsilon{{< /mega-math >}} ensures every configuration has nonzero probability, allowing its success-rate estimate to keep being updated.

- Other choices: one shared reward within each domain, consisting of terminal success plus small regularization terms. The number of PPO minibatches stays fixed, so batch size grows proportionally with environment count.

- For manipulation, the reset pool uses OmniReset's three categories: Reaching, Stable Grasp and Near-Goal, each making up one third. Collision checks are done in advance.

- Relation to existing methods: the idea is shared with Sampling for Learnability (SFL), which scores with {{< mega-math inline=true >}}p(1-p){{< /mega-math >}}. The differences are a sliding-window success rate, an adjustable target success rate and a minimum probability for every configuration. PLR, another comparison, scores with {{< mega-math inline=true >}}|\mathrm{GAE}|{{< /mega-math >}}.

{{< mega-figure src="sgs-sampling.png" caption="SGS sampling weights over success rates" paper="2610.12465" >}}

{{< mega-figure src="sgs-scaling.png" caption="SGS scaling with environment count" paper="2610.12465" >}}

> **Main takeaway**
>
> States that are too hard or too easy have little learning signal. This is adaptive sampling that resets into states of moderate difficulty.

### [CoRL'24] DextrAH-G / DextrAH-RGB {#dextrah}

DextrAH-G: Pixels-to-Action Dexterous Arm-Hand Grasping with Geometric Fabrics

- Paper: [arXiv:2407.02274](https://arxiv.org/abs/2407.02274)

DextrAH-RGB: Visuomotor Policies to Grasp Anything with Dexterous Hands

- Paper: [arXiv:2412.01791](https://arxiv.org/abs/2412.01791)
- Code: [DEXTRAH](https://github.com/NVlabs/DEXTRAH) (“DextrAH on Isaac Lab,” including privileged RL and online distillation)

Direct joint-space PPO for high-dimensional arm-and-multifinger-hand actions trains slowly, learns unnatural grasps (such as holding an object between the middle and ring fingers), fails to learn with multiple objects and is unsafe on hardware.

Method: Replace 23-dimensional joint actions with 11-dimensional palm-pose + hand-PCA-synergy actions, while handling collision avoidance and joint limits.

{{< mega-figure src="dextrah-method.png" caption="DextrAH-G training and distillation" paper="2407.02274" >}}

> **Main takeaway**
>
> Reducing a high-dimensional action space can substantially improve learning speed. [Dexterous Functional Grasping (EigenGrasp)](https://arxiv.org/pdf/2312.02975) uses a similar idea.

Follow-up: [EigenDEXplore](https://eigendexplore.github.io/) uses eigenvectors to aid exploration, with somewhat softer constraints.

## 4. Off-Policy notes {#off-policy}

Mega-Scale RL is still mostly on-policy. Off-policy has attracted renewed interest recently, especially after FlashSAC. Here are representative works and their main insights.

### [ICML'23] PQL — Parallel Q-Learning: Scaling Off-policy Reinforcement Learning under Massively Parallel Simulation {#pql}

Split DDPG into three parallel processes: an actor collects data, a V-learner learns Q and a P-learner learns the policy. This lets off-policy RL use tens of thousands of Isaac Gym environments on one workstation, with shorter wall-clock time and better sample efficiency than PPO.

Paper: [arXiv:2307.12983](https://arxiv.org/abs/2307.12983)

Code: [pql](https://github.com/Improbable-AI/pql)

> **Main takeaway**
>
> Different exploration strengths for different environments are worth trying.

### [ICLR'25] PQN: Simplifying Deep Temporal Difference Learning {#pqn}

Paper: [arXiv:2407.04811](https://arxiv.org/abs/2407.04811)

Code: [purejaxql](https://github.com/mttga/purejaxql)

Shows that LayerNorm, plus a small amount of {{< mega-math inline=true >}}\ell_2{{< /mega-math >}} regularization, lets TD learning converge without a target network or replay buffer. DQN becomes pure GPU Q-learning with many parallel environments, short {{< mega-math inline=true >}}\lambda{{< /mega-math >}}-returns and PPO-like minibatch updates.

> **Main takeaway**
>
> LayerNorm and {{< mega-math inline=true >}}\lambda{{< /mega-math >}}-returns both matter.

### [arxiv'25] FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control {#fasttd3}

Paper: [arXiv:2505.22642](https://arxiv.org/abs/2505.22642)

Code: [FastTD3](https://github.com/younggyoseo/FastTD3)

No new invention: turn PQL's lessons into a synchronous, simple, well-tuned TD3. Parallel simulation + **huge batches** (32,768) + **categorical critic** + **CDQ** solves most HumanoidBench tasks within three hours on one A100, with support for IsaacLab / MuJoCo Playground.

> **Main takeaway**
>
> Large batches and a categorical critic make sense.

### [arxiv'25] FastSAC: Learning Sim-to-Real Humanoid Locomotion in 15 Minutes {#fastsac}

Paper: [arXiv:2512.01996](https://arxiv.org/abs/2512.01996)

Code: [holosoma](https://github.com/amazon-far/holosoma)

Stabilizes the FastSAC variant that was unstable in the FastTD3 paper. With a minimal reward, one RTX 4090 trains a G1/T1 whole-body walking policy (29 degrees of freedom) in 15 minutes, with strong domain randomization and hardware transfer. Whole-body motion tracking with 4×L40S and 16,384 environments is also faster than PPO.

> **Main takeaway**
>
> LayerNorm, Q-mean rather than CDQ, and automatic alpha.

### [RSS'26] FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control {#flashsac}

- Paper: [arXiv:2604.04539](https://arxiv.org/abs/2604.04539)
- Code: [FlashSAC](https://github.com/Holiday-Robot/FlashSAC)

> **Main takeaway**
>
> Use a larger network (2.5M parameters, six layers), larger batches and a larger replay buffer, with **fewer gradient updates (UTD = 2/1024)**. Then constrain norms with BatchNorm, RMSNorm, weight projection and a distributional critic, making SAC fast and stable on high-dimensional tasks.

Follow-up: **WarpSAC: Towards the Pinnacle of Scalable Off-policy RL by Rethinking Exploration and Exploitation**

- Paper: [arXiv:2608.24479](https://arxiv.org/abs/2608.24479)
- Code: [warprl](https://github.com/wzhhasadream/warprl)

> **Main takeaway**
>
> Under massive GPU parallelism, **turning off weight-projection normalization and using a single Q** works better.
