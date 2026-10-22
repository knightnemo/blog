---
title: "What Matters for Dexterous RL?"
date: 2026-10-23T00:00:00+08:00
draft: false
ShowToc: false
math: true
guideLang: en
tags: ["reinforcement-learning", "robotics", "dexterous-manipulation", "sim-to-real", "paper-reading"]
summary: "Reading notes on dexterous RL: reset distributions, goal-conditioned pretraining, structured exploration, motion tracking, contact priors and simulation."
---

{{< dex-header >}}

For in-hand reorientation, tool use and tight-tolerance assembly with a multifingered hand, a common recipe is to train with RL in massively parallel simulation, then transfer to a real robot. Much of this work starts from the same baseline: PPO, a diagonal Gaussian policy, and resets to the task's default initial state. On many dexterous tasks, that baseline gets nowhere. What I want to understand is **what these papers add to PPO to make learning work, and why each addition is needed.**

I will start with the problem and the failure modes, then go through six groups of methods according to which part of the RL problem they change.

## 1 Problem setup {#s1}

Consider an MDP $(\mathcal S,\mathcal A,P,r,\rho_0,\gamma)$, optionally conditioned on a goal $g$. A Gaussian policy $\pi_\theta(a\mid s,g)=\mathcal N(\mu_\theta(s,g),\operatorname{diag}(\sigma^2))$ outputs joint targets for the arm and hand. Here $\rho_0$ is the initial-state distribution and $P$ is the simulator's transition model. Leaving aside PPO's clipping and importance ratios, the policy-gradient signal has the form

{{< robo-math >}}
\nabla_\theta J \approx \mathbb{E}_{s_0\sim\rho_0,\,g,\,\tau\sim(\pi_\theta,P)}\!\left[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t,g)\,\hat A_t\right]
{{< /robo-math >}}

The advantage estimate $\hat A_t$ comes from the reward $r$. Each group below changes one part of this expression: $\rho_0$, the goal distribution, policy noise, rewards, or transitions.

Two properties of dexterous hands are useful to keep in mind:

1. **The action space is large.** A hand with 16–22 actuated joints on a 7-DoF arm has 23–29 action dimensions. Two arms and two hands can exceed 50.
2. **Human data can provide a prior.** The resemblance to human hands makes motion capture, hand–object videos and hand-pose datasets useful after retargeting.

## 2 Why plain PPO is often not enough {#s2}

The failures reported in these papers fall into four groups. They are related, but their remedies differ.

### 2.1 Almost no success signal {#no-success}

The successful region is narrow. A stable grasp requires coordinated contacts and balanced forces; rotation and insertion can also require millimetre-level pose accuracy. From a default reset, Gaussian perturbations around a randomly initialized policy are unlikely to reach that region. If every rollout fails under a sparse success reward, there is little useful advantage signal pointing towards success. [DAPG](https://arxiv.org/abs/1709.10087) reported that random exploration under sparse rewards never encountered success on its Adroit tasks, apart from the in-hand task. In DemoStart's plug-insertion ablation, plain RL achieved 0% success.

### 2.2 Local optima from dense rewards {#reward-traps}

The obvious response is reward shaping: distance from the hand to the object, lift height, distance from the object to its target. But a high-dimensional contact system offers plenty of cheap ways to collect those rewards. OmniReset observed policies that only learned to reach. In Play2Perfect, a policy trained from scratch with dense rewards balanced the object on its fingers instead of grasping it, and failed as soon as an external force was applied (Figure 1). In Twisting Lids, a gait reward without contact guidance produced disorganized finger movements. A reward can provide a strong signal and still point towards the wrong behavior.

{{< dex-figure src="p2p_brittle.jpg" caption="Figure 1. Left: on a simplified assembly task, dense-reward training from scratch (orange) takes 33 times as long as Play2Perfect (blue). Middle: Play2Perfect grasps with several fingers; the scratch policy lifts with its thumb. Right: the latter loses success rapidly under external forces." paper="2606.26428" >}}

### 2.3 Independent noise wastes exploration in joint space {#joint-noise}

A diagonal Gaussian perturbs each joint independently. Human hand poses, however, are strongly correlated. In DextrAH-G's analysis of DexYCB, the first five principal components explained 98% of the variance. Independent perturbations across twenty-odd joints often cancel each other or produce poses rarely seen in human hands. DextrAH-G's joint-space policy learned unusual grasps, such as pinching with the middle and ring fingers, and converged much more slowly (Figure 2). [HOP](https://arxiv.org/abs/2409.08273) reported that its PPO baseline trained from scratch with joint-position actions consistently failed.

{{< dex-figure src="dextrah_cspace.jpg" narrow=true caption="Figure 2. DextrAH-G with joint-space actions (red) versus palm-pose and PCA hand actions (green). The vertical axis measures the fraction of runs reaching enough consecutive successes; the horizontal axis is training time in hours." paper="2407.02274" >}}

### 2.4 Diminishing returns from more environments {#scaling}

GPU simulation can run tens of thousands of environments on one card. If all of them sample from the same Gaussian policy, though, the additional data may provide little extra coverage. SAPG found that PPO saturated quickly as the environment count increased on AllegroKuka tasks, with almost no progress on bimanual reorientation (Figure 3). DexPBT reported large differences across independent PPO seeds: none of eight seeds managed to tighten the reorientation tolerance to 1 cm.

{{< dex-figure src="sapg_curves.jpg" caption="Figure 3. AllegroKuka regrasping, throwing, reorientation and bimanual reorientation. SAPG (blue) and PBT (green) outperform PPO (red) and PQL, especially on the harder tasks." paper="2407.20230" >}}

The first two failures motivate work on resets, goals, tracking and contact priors. The third motivates structured exploration; the fourth motivates population training.

## 3 Solutions {#solutions}

### 3.1 Changing the reset distribution {#s3}

The first group changes where an episode starts. Resets that include an object already grasped, an aligned insertion, or a state close to the goal allow some episodes to reach success quickly. The policy can learn the last few steps first, then extend that ability backwards. A curriculum emerges from the changing set of states it can solve.

Under the usual discounted-MDP assumptions, an unrestricted policy can be optimal from every state. Changing $\rho_0$ therefore need not change that statewise optimal policy; it changes where training samples are spent. With function approximation and finite data, performance can still deteriorate in regions that receive too few samples. This is why the training mixture usually retains some default resets.

The cost is generating valid intermediate states. For a parallel-jaw gripper, one can sample a grasp pose and solve inverse kinematics. Generating stable simultaneous contacts for several fingers is much harder. The papers below differ mainly in where their intermediate states come from and how they select among them.

#### OmniReset [ICLR'26] {#omnireset}

*Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning* · [arXiv](https://arxiv.org/abs/2603.15789) · [Project](https://weirdlabuw.github.io/omnireset/) · [Code](https://github.com/UW-Lab/UWLab)

OmniReset generates four reset families offline: random arm poses, states near the object, stable grasps in the air, and states near the goal. All are validated in simulation and mixed uniformly during training. The tasks share rewards and hyperparameters, with neither a curriculum nor demonstrations. State-based simulation success rates on peg insertion, table-leg screwing and drawer insertion are 95.0%, 93.9% and 89.6%. After distillation to RGB policies, the real-robot rates are 85%, 56% and 15%; behavior cloning from 100 demonstrations achieves about 2%.

Its ablations favor more environments and more diverse grasps. With too few environments, the policy only succeeds from near-goal resets (Figure 4b). Reward weights have little effect within the tested settings. One important limitation for this discussion: OmniReset uses a parallel-jaw gripper, not a multifingered hand.

{{< dex-figure src="omni_pipeline.jpg" caption="Figure 4a. OmniReset generates reaching, near-object, grasped and near-goal states, trains a state-based RL policy, then distills it into a visual policy for deployment." paper="2603.15789" >}}

{{< dex-figure src="omni_envs.jpg" caption="Figure 4b. Success from different reset families as the environment count increases. Completing the whole task from the beginning (far right) requires 32k–65k environments in these experiments." paper="2603.15789" >}}

> **Main takeaway**
>
> With broad enough reset coverage, reward tuning matters much less. The difficult part becomes generating diverse intermediate states. That is cheap for a gripper and expensive for a multifingered hand.

#### DemoStart [ICRA'25] {#demostart}

*DemoStart: Demonstration-led Auto-curriculum Applied to Sim-to-Real with Multi-fingered Robots* · [arXiv](https://arxiv.org/abs/2409.06613) · [Project](https://sites.google.com/view/demostart)

DemoStart uses a three-fingered hand. It extracts full simulator states from 2–60 demonstrations and applies a zero-variance filter: retain a reset only if repeated attempts produce both successes and failures. The plug-insertion ablation is unusually clear. Plain RL: 0%. Demonstration resets alone: still 0%. Keeping resets that have succeeded at least once: also 0%. With zero-variance filtering: 97.2%. During training, selected resets move from the end of a demonstration towards its beginning (Figure 5).

{{< dex-figure src="demostart_curr.jpg" caption="Figure 5. Selected reset positions within a demonstration versus policy updates; 100% denotes the demonstration's end. The zero-variance filter moves the selected starts backwards as learning progresses." paper="2409.06613" >}}

> **Main takeaway**
>
> Having a reset pool is not enough. The selected states need to produce a useful difference between successful and failed attempts. Under this sparse-reward setup, resets that always succeed or always fail offer little of that signal.

#### SGS [CoRL'26] {#sgs}

*A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control* · [arXiv](https://arxiv.org/abs/2610.12465) · [Project](https://sgs-rl.github.io/)

SGS replaces binary filtering with continuous weighting. For each reset configuration, it records the success rate $p$ over the last 100 attempts, scores it with a Beta kernel that favors intermediate success rates, and samples in proportion to that score. On Franka nut screwing with one million environments, SGS reaches 0.70 success, compared with 0.06 for uniform sampling and 0.05 for PLR. Removing near-goal resets reduces success to zero. Once those resets are available, a gravity curriculum adds little: 69.4% versus 70.2%. The experiments cover grippers and legged robots.

{{< dex-figure src="sgs_sampling.jpg" caption="Figure 6. SGS sampling distributions early, midway and late in training on a locomotion terrain task. Sampling mass follows the boundary of the policy's current ability." paper="2610.12465" >}}

> **Main takeaway**
>
> At large scale, uniform sampling can waste a great deal of computation. A reset pool still needs a sampling rule that follows what the policy is currently learning.

#### Lessons from Learning to Spin “Pens” [CoRL'24] {#penspin}

[arXiv](https://arxiv.org/abs/2407.18902) · [Project](https://penspin.github.io/) · [Code](https://github.com/HaozhiQi/penspin) · Allegro hand

Pen spinning requires the fingers to make and break contact in sequence. The authors design six canonical initial grasps at key moments when the index finger, thumb or middle finger releases and re-establishes contact (Figure 7). With only one of those resets, the policy does not learn to switch fingers.

{{< dex-figure src="penspin_canon.jpg" caption="Figure 7. Six canonical initial grasps inspired by human pen spinning, chosen at key moments when fingers release and re-establish contact." paper="2407.18902" >}}

{{< dex-figure src="penspin_curves.jpg" caption="Figure 8. The two left plots compare episode reward and length for six canonical resets (blue) and a single reset (red)." paper="2407.18902" >}}

> **Main takeaway**
>
> For in-hand manipulation, useful intermediate states can be defined by contact transitions, rather than coarse task stages.

#### Human2Sim2Robot [arXiv'25]: human priors in resets {#human2sim2robot}

*Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration* · [arXiv](https://arxiv.org/abs/2504.12609) · [Project](https://human2sim2robot.github.io/) · [Code](https://github.com/tylerlum/human2sim2robot)

This work takes two things from one human RGB-D video: an object-pose trajectory for the reward, and the hand pose just before manipulation, retargeted to initialize RL. The human pre-manipulation pose works substantially better than a default reset, a reset above the object, or the same hand pose placed farther away (Figure 9). [Lin et al.'s 2025 humanoid study](https://arxiv.org/abs/2502.20396) reports a related result: initializing the hands from human data raises the fraction of seeds that learn a bimanual handover from 0% to 30%.

{{< dex-figure src="h2s2r_premanip.jpg" narrow=true caption="Figure 9. Human2Sim2Robot object-tracking rewards with human pre-manipulation initialization (blue), default initialization (orange), an overhead start (green), and the pre-manipulation pose placed farther from the object (red)." paper="2504.12609" >}}

> **Main takeaway**
>
> A direct use of human data is to supply a state in which the hand is already in the right place. For multifingered hands, that is precisely the sort of state that is difficult to generate automatically.

### 3.2 Changing the goal distribution {#s4}

The second group changes what the policy learns. Instead of training for one task, it trains $\pi(a\mid s,g)$ over a broad distribution of goals and chains goals within each episode. Nearby goals make some successes attainable, keeping a learning signal available. Across tasks, the policy learns a reusable skill for moving an object to a requested pose. A downstream task can then supply a goal sequence instead of a new reward function.

Unlike changing resets, changing the goal distribution changes the training objective. A policy optimized for random goal reaching need not be optimal for a particular downstream task. Either that task must be expressible as a goal sequence for zero-shot execution, or the policy needs fine-tuning. Here human data can enter purely as object-pose trajectories, without requiring a correspondence between human and robot fingers.

#### SimToolReal [arXiv'26] {#simtoolreal}

*SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation* · [arXiv](https://arxiv.org/abs/2602.16863) · [Project](https://simtoolreal.github.io/) · [Code](https://github.com/tylerlum/simtoolreal) · KUKA arm + 22-DoF Sharpa hand

Training objects are procedural tools: a handle and a head, each made from a box or capsule, with randomized dimensions and densities. The first target is sampled uniformly in the workspace. Subsequent targets are within 10 cm and 90° of the previous one. Success requires the maximum error across four keypoints to be below 1 cm. At deployment, tool-pose trajectories estimated from human RGB-D videos become goal sequences for zero-shot execution (Figure 10).

Across 12 real tools and 24 tasks, task progress is 37% higher than the fixed-grasp and kinematic-retargeting baselines, and comparable to policies trained for individual tool categories. Training uses roughly 120 billion frames. Replacing SAPG with PPO, or removing the asymmetric critic, substantially reduces performance (Figure 11).

{{< dex-figure src="str_overview.jpg" caption="Figure 10. SimToolReal trains a goal-conditioned policy with procedural objects and random target poses in simulation. At deployment, object-pose sequences extracted from human videos provide goals for zero-shot tool use." paper="2602.16863" >}}

{{< dex-figure src="str_ablation.jpg" narrow=true caption="Figure 11. Training reward versus environment steps, averaged over five seeds: SAPG with an asymmetric critic (cyan), PPO with an asymmetric critic (gray), and SAPG with a symmetric critic (orange)." paper="2602.16863" >}}

> **Main takeaway**
>
> A family of nearby goals supplies a learning signal where a single sparse task may not. Human videos only need to provide object trajectories. The cost is a huge sample budget and, in these experiments, a strong dependence on SAPG and an asymmetric critic.

#### Play2Perfect [arXiv'26] {#play2perfect}

*Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly?* · [arXiv](https://arxiv.org/abs/2606.26428) · [Project](https://play2perfect.github.io/) · [Code](https://github.com/kushal2000/play2perfect) · Same platform

Play2Perfect uses SimToolReal-style goal reaching as play pretraining, then fine-tunes on each assembly task with sparse task rewards. The reward includes a smoothness term, one-time subgoal rewards, and a reward for withdrawing the hand after completion. Subgoals are generated from CAD models by reversing a disassembly sequence. Each task takes 2–5 hours of fine-tuning. Sparse- and dense-reward baselines trained from scratch achieve no success after 24 hours on these tasks (Figure 12). On the real robot, insertion with 10 mm, 2 mm and 0.5 mm clearances succeeds in 10/10, 9/10 and 6/10 trials.

{{< dex-figure src="p2p_speed.jpg" caption="Figure 12. Success versus training time on four assembly tasks: Play2Perfect with sparse-reward fine-tuning (blue), dense rewards from scratch (orange), and sparse rewards from scratch (gray)." paper="2606.26428" >}}

The most useful part of the paper is the pretraining ablation (Figure 13). More objects help: 1,000 > 100 > 10. Full 6D goals outperform rotation-only goals by about ten percentage points; translation-only goals do not teach in-hand reorientation. Online random goal trajectories beat a fixed trajectory library. A 1 cm target tolerance works best. At 5 cm, downstream performance stalls around 60%; at 10 cm, the learned behavior barely transfers.

{{< dex-figure src="p2p_matters.jpg" caption="Figure 13. Effects of object diversity, training goals, trajectory diversity and target precision on assembly fine-tuning, averaged over four tasks and three seeds." paper="2606.26428" >}}

> **Main takeaway**
>
> The goal distribution determines the skill learned in pretraining. If a fixed grasp and arm motion can solve the goals, that is what the policy will learn. The goals must demand enough precision and variety to require finger-level manipulation.

DexterityGen ([arXiv:2502.04307](https://arxiv.org/abs/2502.04307)) provides a related observation. It uses RL to generate transitions between grasps, and reports that targets need to be near the current state. Distant targets cause learning to stall early.

### 3.3 Structured exploration {#s5}

The third group addresses inefficient joint-space noise and limited diversity at scale. There are three approaches: constrain actions to a low-dimensional subspace; retain the full action space but change the noise covariance; or train a population of policies. In the first two, human priors can enter as hand synergies, often the principal components of human hand poses.

#### DextrAH-G [CoRL'24] {#dextrah-g}

*DextrAH-G: Pixels-to-Action Dexterous Arm-Hand Grasping with Geometric Fabrics* · [arXiv](https://arxiv.org/abs/2407.02274) · [Project](https://sites.google.com/view/dextrah-g) · [Code](https://github.com/NVlabs/DEXTRAH) · KUKA + Allegro

The policy outputs a 6D palm pose and five PCA hand coordinates, giving 11 action dimensions instead of 23 motor commands. The PCA basis comes from retargeted DexYCB human-hand data. A lower-level geometric fabrics controller handles collision avoidance and joint limits. Compared with joint-space actions, PCA actions converge faster and produce more human-like grasps (Figures 2 and 14). Joint-space training is about four times slower on one object and fails in the multi-object setting. The authors also acknowledge the limitation: PCA constrains in-hand dexterity.

{{< dex-figure src="dextrah_pca.jpg" narrow=true caption="Figure 14. Grasps in 2D and 5D PCA action spaces. Five dimensions allow more grasp diversity while retaining the human grasp-pose structure." paper="2407.02274" >}}

> **Main takeaway**
>
> Restricting actions to a hand-synergy subspace makes exploration easier, but the same subspace limits what the policy can eventually do.

#### EigenDEXplore [arXiv'26] {#eigendexplore}

*EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors* · [arXiv](https://arxiv.org/abs/2610.07681) · [Project](https://eigendexplore.github.io/) · [Code](https://github.com/hgupt3/eigendexplore)

EigenDEXplore treats the action space and noise space separately. It retargets a large human hand-pose dataset, performs PCA in robot joint space, and retains 90% of the variance: nine dimensions for the 22-DoF Sharpa hand. Actions remain in the full joint space. Exploration adds both independent joint noise and noise along the principal directions:

{{< robo-math >}}
\begin{aligned}a &= \mu_\theta(s)+\sigma_q\odot\eta_q+E^{\mathsf T}(\sigma_e\odot\eta_e),\\ \eta_q,\eta_e &\sim\mathcal N(0,I).\end{aligned}
{{< /robo-math >}}

PPO learns both sets of noise scales. Deployment uses only the mean, so the basis is no longer needed. Figure 15 compares joint-space actions, reduced eigen-space actions, eigen-residual actions and EigenDEXplore. After 60 billion SimToolReal frames, their consecutive-success counts are approximately 4.9, 6, 12.0 and 14.1 (Figure 16). Reduced eigen-space actions lead early but fall behind when precision becomes important. On real tool use, task progress is 82% versus 33% for joint-space exploration.

Two controls clarify the source of the gain. An orthogonal basis with the same spectrum but random directions performs worse than joint-space exploration. On DexMachina, where a curriculum and auxiliary rewards already help learning, gains only appear after those aids are removed.

{{< dex-figure src="eig_actions.jpg" caption="Figure 15. Four action/noise combinations. Independent joint noise produces awkward poses; eigen-space actions coordinate the hand but restrict expressiveness; eigen-residual actions are redundant. EigenDEXplore keeps joint-space actions and adds noise along human synergy directions. Original paper: Figure 1." paper="2610.07681" >}}

{{< dex-figure src="eig_str.jpg" caption="Figure 16. EigenDEXplore on SimToolReal. (a) Consecutive successes and reward at nine billion frames. (b) Median runs trained to sixty billion frames with a tolerance curriculum. Original paper: Figure 3." paper="2610.07681" >}}

> **Main takeaway**
>
> Human priors can improve exploration without restricting the final action space. The useful ingredient is the direction of the human synergies, not low dimensionality by itself. And when other training aids already solve the exploration problem, the gain can disappear.

#### SAR [RSS'23] and Lattice [NeurIPS'23] {#sar-lattice}

SAR · [arXiv](https://arxiv.org/abs/2307.03716) · [Project](https://sites.google.com/view/sar-rl) · [Code](https://github.com/MyoHub/myosuite/blob/main/docs/source/tutorials/SAR/SAR_tutorial.ipynb)  
Lattice · [arXiv](https://arxiv.org/abs/2305.20065) · [Code](https://github.com/amathislab/lattice) · Musculoskeletal hands

These papers study musculoskeletal control, where redundancy is even greater: MyoHand has 39 muscles. SAR first trains a policy on an easy task, extracts synergies from its muscle activations, and uses them as the action representation for new tasks. Across 100 object-reorientation tasks, SAR exceeds 70% success; end-to-end and curriculum baselines remain below 20% (Figure 17). Lattice injects temporally correlated noise into the policy's hidden layers, inducing correlations across actuators. Its argument is that independent actuator noise is inefficient in an overactuated system.

{{< dex-figure src="sar_100obj.jpg" narrow=true caption="Figure 17. Reorientation across 100 objects: SAR-RL (blue), RL with a curriculum (orange), and end-to-end RL (green)." paper="2307.03716" >}}

> **Main takeaway**
>
> Synergies need not come from human data. They can be extracted from a policy on an easier task. What matters here is coordination across actuators during exploration.

#### DexPBT [RSS'23] and SAPG [ICML'24] {#population}

DexPBT · [arXiv](https://arxiv.org/abs/2305.12127) · [Project](https://sites.google.com/view/dexpbt) · [Code](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)  
SAPG · [arXiv](https://arxiv.org/abs/2407.20230) · [Project](https://sapg-rl.github.io/) · [Code](https://github.com/jayeshs999/sapg)

Both use multiple policies to address the scaling problem. DexPBT trains 8–32 PPO policies, periodically copies stronger runs into weaker ones, and perturbs hyperparameters and reward coefficients. On bimanual reorientation, it approaches 40 consecutive successes while the best independent seed remains below ten.

SAPG divides environments among follower policies with a shared backbone and separate learnable embeddings. A leader uses importance sampling to learn from their data (Figure 18). In its bimanual-reorientation comparison, SAPG reaches 28.6 consecutive successes, DexPBT 14.5 and PPO 1.7. There is also a counterpoint: [Beukman et al.](https://arxiv.org/abs/2603.06009) (RLC 2026) report that PPO can approach SAPG when its minibatch size is brought back into a suitable range. Part of the apparent saturation comes from hyperparameters that were not scaled appropriately with the environment count.

{{< dex-figure src="sapg_method.jpg" narrow=true caption="Figure 18. SAPG's leader and followers share a backbone, with a separate learnable embedding for each policy. The leader uses importance sampling to learn from follower data." paper="2407.20230" >}}

> **Main takeaway**
>
> A single Gaussian policy may provide too little diversity for a high-dimensional task. Policy-level diversity can help, but poor PPO hyperparameter scaling should be ruled out before attributing the entire gain to exploration.

### 3.4 Motion tracking {#s6}

The fourth group uses human priors more directly. Retarget a human hand–object interaction to a robot hand, then train a policy to track the reference. Rewards usually include object-pose, wrist and joint errors, plus contact terms. Tracking supplies a target at every time step, addressing sparse successes, reward ambiguity and some exploration difficulties. Tasks can also be generated from data instead of receiving individually designed rewards.

The difficulties are quite specific. Kinematic retargeting often produces physically infeasible motions because the hands differ. Accurate tracking is itself an exploration problem: the robot must establish the right contacts first. A policy may track its reference well but generalize poorly away from it. And it is not obvious whether the reference should constrain the whole hand or only the object. Each paper below addresses some of these issues.

#### ManipTrans [CVPR'25] {#maniptrans}

*ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning* · [arXiv](https://arxiv.org/abs/2503.21860) · [Project](https://maniptrans.github.io/) · [Code](https://github.com/ManipTrans/ManipTrans)

ManipTrans separates transfer into two stages. It pretrains a hand-motion imitator on large-scale human data, then learns a task-specific residual policy to handle the physical constraints of object interaction (Figure 19). Its comparison quantifies the embodiment gap. Single-hand / bimanual success is 4.6% / 0.0% for kinematic retargeting, 34.3% / 12.1% for RL alone, 47.8% / 13.9% for retargeting plus residual learning, and 58.1% / 39.5% for the full method.

{{< dex-figure src="maniptrans_pipe.jpg" caption="Figure 19. ManipTrans pretrains a hand-motion imitator on motion capture, then fine-tunes a residual policy to satisfy physical constraints during manipulation." paper="2503.21860" >}}

> **Main takeaway**
>
> Learn how the hand moves first, then learn how that motion must change when an object is involved. The two stages separate motion imitation from contact dynamics.

#### DexTrack [ICLR'25] {#dextrack}

*DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References* · [arXiv](https://arxiv.org/abs/2502.09614) · [Project](https://meowuu7.github.io/DexTrack/) · [Code](https://github.com/Meowuu7/DexTrack)

DexTrack trains a general tracking controller with RL and imitation of successful tracking demonstrations. The problem is obtaining those demonstrations for difficult references. It uses homotopy optimization: solve a related easy trajectory first, then progressively move towards the difficult one. The resulting demonstrations improve the controller, which helps mine more demonstrations, and the two steps alternate (Figure 20). The paper reports success more than 10% above its strongest baseline.

{{< dex-figure src="dextrack_pipe.jpg" caption="Figure 20. DexTrack alternates between mining successful tracking demonstrations and training a general tracking controller from them." paper="2502.09614" >}}

> **Main takeaway**
>
> A general tracker needs demonstrations of difficult trajectories. Those demonstrations can be obtained by gradually deforming easier problems into harder ones.

#### DexMachina [ICML'26] {#dexmachina}

*DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation* · [arXiv](https://arxiv.org/abs/2505.24853) · [Project](https://project-dexmachina.github.io/) · [Code](https://github.com/MandiZhao/dexmachina) · Genesis · Six robot hands

DexMachina studies functional retargeting: reproduce the object motion in a human bimanual demonstration without requiring identical hand motion. Its main mechanism is a virtual object controller. Springs and dampers pull the object along the reference, so the simulator initially performs some of the manipulation for the policy. Whenever reward reaches a threshold, controller gains decay exponentially, handing control back to the policy (Figure 21).

The curriculum is the decisive component. Task reward alone is insufficient; auxiliary imitation, behavior-cloning and contact rewards provide inconsistent gains. ManipTrans-style curricula over physical parameters and thresholds add little in this comparison (Figure 22). There are no real-robot experiments.

{{< dex-figure src="dexmachina_overview.jpg" narrow=true caption="Figure 21. DexMachina trains from human hand–object references with task, motion and contact rewards. The virtual object controller provides strong assistance early, weaker assistance later, and is eventually removed." paper="2505.24853" >}}

{{< dex-figure src="dexmachina_results.jpg" caption="Figure 22. Success across seven demonstrations and four robot hands: kinematic retargeting, task reward alone, task plus auxiliary rewards, DexMachina, and ManipTrans." paper="2505.24853" >}}

> **Main takeaway**
>
> Helping the object move early in training, then removing that help, can solve a tracking exploration problem more effectively than adding auxiliary rewards.

#### DONIC [arXiv'26] {#donic}

*DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations* · [arXiv](https://arxiv.org/abs/2610.29997) · [Project](https://ropedia.github.io/DONIC/) · [Code](https://github.com/Ropedia/DONIC) · Two Wuji Hand V1 hands

DONIC trains a general bimanual tracker on roughly 3,400 clips. These combine public datasets with its new BiHOI dataset: 1,575 clips and 229 objects. The method has three components (Figure 23): residual actions added to the next reference frame; object-centric rewards computed separately for each hand, with fingertip contact terms enabled according to the reference; and advantage centering within each clip to reduce differences caused by clip difficulty. Successfully replayed trajectories are also added to a demonstration buffer for PPO plus behavior cloning.

One model reaches 86.8% success on unseen trajectories and 57.3% on unseen objects, while the strongest baselines are around 9–12%. Two individual design changes each contribute 5–9 percentage points. There are no real-robot experiments.

{{< dex-figure src="donic_overview.jpg" caption="Figure 23. DONIC combines reconstruction and retargeting with a shared bimanual policy, per-hand object-centric rewards, clip-centered advantages, and behavior cloning." paper="2610.29997" >}}

> **Main takeaway**
>
> Much of the method combines existing components. The ablations suggest that the large overall gain depends on the data scale and the complete training recipe, not one isolated trick. Whole-hand tracking also works here with a hand whose morphology is close to a human's.

#### AdaDex [arXiv'26] {#adadex}

*AdaDex: Learning Adaptable Dexterous Skills via Hybrid Dexterous Mimicking* · [arXiv](https://arxiv.org/abs/2610.28628) · [Project](https://yuxuank.com/AdaDex/) · XHand, Wuji, Sharpa and others

AdaDex directly addresses poor generalization away from a reference. One policy trains in two types of parallel environments (Figure 24). Tracking environments use dense tracking rewards and resets sampled from the reference, with adaptive allocation based on success-rate variation across segments. Generalization environments heavily perturb both resets and goals, use sparse object-success rewards, and retain dense guidance only for the wrist. Each skill needs just a few seconds of human motion.

On GRAB, progress is 96.7% for seen motions and 80.0% for unseen ones, compared with 35.2% and 7.3% for ManipTrans. The ablations separate the two roles neatly. Removing tracking environments drops every metric to around 8%. Removing generalization environments leaves seen motions largely unchanged at 97.5%, but reduces unseen-motion progress to 31.8%.

{{< dex-figure src="adadex_method.jpg" narrow=true caption="Figure 24. AdaDex shares one actor–critic across tracking environments with dense reference rewards and small reset deviations, and generalization environments with larger perturbations and sparse goal rewards." paper="2610.28628" >}}

> **Main takeaway**
>
> Tracking tells the policy where to go; perturbed resets and goals make it learn beyond the reference. Training both in the same batch is useful, and the ablation shows why neither is sufficient alone.

#### TeleDexter [arXiv'26] {#teledexter}

*Towards Human-level Dexterous Teleoperation* · [arXiv](https://arxiv.org/abs/2607.11481) · [Project](https://bigai-dex.github.io/blog/teledexter/) · Franka + LeapHand / Sharpa · Four RTX 5090 GPUs, about 62,000 environments and $10^{10}$ steps

TeleDexter learns a low-level teleoperation controller. Motion capture supplies fingertip and object targets; the policy supplies joint commands. Human references are divided into sequences of subgoals, rather than treated as trajectories to match at every frame. Episodes begin at random reference frames and can switch between trajectories (Figure 25).

The reward is mainly sparse subgoal success, plus a dense tracking term weighted by 0.1. This mixture achieves 32.6–186.6 consecutive subgoals, compared with about 2.7 for pure dense tracking. Across seven tasks, teleoperation success is 75.2%, versus 0–9% for other methods. Random action masking is also important for transfer: selected action dimensions hold their previous commands for a random duration. Removing it drops real-robot screwdriver success from 73% to zero.

{{< dex-figure src="teledexter_method.jpg" caption="Figure 25. TeleDexter divides human hand–object references into successive subgoals and trains with SAPG. Its main components are mixed rewards, action masking and curricula." paper="2607.11481" >}}

> **Main takeaway**
>
> In this setting, treating human motion as a sequence of goals works much better than matching every frame. It brings the goal-reaching idea from Section 3.2 into tracking.

#### Tracking the object or the hand? {#object-or-hand}

Several papers compare these choices. Human2Sim2Robot finds little benefit from adding hand-tracking rewards to object-trajectory rewards, or from learning residuals on replayed demonstrations (Figure 26). Its conclusion is that highly accurate human hand data is unnecessary for its tasks. ObjDex ([arXiv:2411.04005](https://arxiv.org/abs/2411.04005)) uses human data for wrist trajectories and leaves the fingers to RL, noting that human finger motion is not always helpful across embodiments.

DONIC and AdaDex, meanwhile, obtain good results with whole-hand tracking. Their hands—Wuji, Sharpa and XHand—are closer to human proportions and thumb structure. My reading is that the closer the morphology, the lower the cost of following the hand. With a larger embodiment gap, object motion plus wrist or pre-grasp guidance may be a better starting point. This is an interpretation across papers, not a controlled comparison of hand designs.

{{< dex-figure src="h2s2r_fullhand.jpg" narrow=true caption="Figure 26. Human2Sim2Robot object-tracking rewards: object trajectory only (blue), additional hand-tracking rewards (orange), and residual learning on demonstrations (green)." paper="2504.12609" >}}

#### Do as I Do [arXiv'26]: where references come from {#do-as-i-do}

*Do as I Do: Dexterous Manipulation Data from Everyday Human Videos* · [arXiv](https://arxiv.org/abs/2606.19333) · [Project](https://www.do-as-i-do.com/) · [Code](https://github.com/malik-group/do-as-i-do) · Two UR3e arms + Sharpa hands · Not RL

Tracking depends on good references. Do as I Do tries to obtain them from monocular internet videos. Visual foundation models reconstruct hand and object motion; sampling-based trajectory optimization in simulation, using [SPIDER](https://arxiv.org/abs/2511.09484)-style MPPI, repairs the kinematic retargeting into physically feasible motion; the result is replayed on the robot (Figure 27). It does not train a policy.

To handle reconstruction noise, the authors add warm-up steps, random perturbations and transition rewards. Retargeting success rises from 0.25 to 0.71. Data yield remains low: among 2,000 pre-screened videos, only 9% contain valid interactions and about 4% pass quality checks.

{{< dex-figure src="doasido_method.jpg" caption="Figure 27. Do as I Do reconstructs hand and object motion with visual foundation models, then uses sampling-based optimization in simulation to retarget it to the robot." paper="2606.19333" >}}

> **Main takeaway**
>
> Human video is plentiful, but physically usable references are not. Reconstruction and feasibility filtering discard roughly twenty times as much data as they retain. For tracking methods, obtaining the reference is itself a major cost.

### 3.5 Contact priors and modified physics {#s7}

Two further changes appear repeatedly, each addressing something the previous four groups do not fully resolve.

#### 3.5.1 Contact priors {#contact-priors}

A task can admit many grasps and finger configurations. A dense reward that only constrains the object leaves room for strange contacts, as in Section 2.2. A contact prior specifies desired contact locations in the reward without prescribing the entire hand trajectory. It supplies more information than an object goal but less than full tracking. Because it refers to contact locations, it can be less sensitive to hand morphology than joint-level imitation.

##### Twisting Lids Off with Two Hands [CoRL'24] {#twisting-lids}

[arXiv](https://arxiv.org/abs/2403.02338) · [Project](https://toruowo.github.io/bimanual-twist/) · [Code](https://github.com/ToruOwO/twisting-lids) · Two Allegro hands

The task is bimanual lid twisting. Reference contact points are placed on the bottle and lid, and fingers are rewarded for approaching them. A gait reward without contact guidance produces disorganized motion; reducing the contact coefficient prevents the policy from learning to twist (Figure 28). The physical model is also interesting: the lid is a revolute joint with brake friction, which the authors use to reproduce the required static-friction behavior.

{{< dex-figure src="twist_reward.jpg" caption="Figure 28. Left: behavior with the full reward (top), a gait reward without contact guidance (middle), and half the contact-reward weight (bottom). Right: real-robot perturbations applied from different directions." paper="2403.02338" >}}

##### Vision-based dexterous manipulation on humanoids [CoRL'25] {#humanoid-contact}

*Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids* · [arXiv](https://arxiv.org/abs/2502.20396) · [Project](https://toruowo.github.io/recipe/)

Contact stickers on the object define targets that enter the reward alongside object goals. Their positions determine the contact pattern that emerges (Figure 29). The same reward form can be reused for grasping, lifting a box and bimanual handover.

{{< dex-figure src="lin_contact.jpg" narrow=true caption="Figure 29. Contact stickers at the left/right centers (top), top/bottom centers (middle), or bottom edges (bottom) produce different contact patterns." paper="2502.20396" >}}

> **Main takeaway**
>
> Contact locations are a compact prior that transfers more readily across hand designs than joint trajectories. They constrain exactly what object-only rewards leave unspecified: where the fingers should go.

#### 3.5.2 Changing the physics {#physics}

The last group changes the transition model $P$, for two reasons. One is learnability: relax the physics, make success easier, and gradually restore the intended dynamics. DexMachina's virtual controller does this. QuasiSim ([arXiv:2404.07988](https://arxiv.org/abs/2404.07988)) transfers human motion through a curriculum of quasi-physical simulators, moving from relaxed, differentiable contacts towards realistic dynamics (Figure 30). Reduced gravity early in training is another example.

The other reason is practical simulation. Threads and static friction are difficult or expensive to reproduce accurately, so some methods deliberately use a simpler model.

{{< dex-figure src="quasisim_curr.jpg" caption="Figure 30. QuasiSim begins with relaxed contact and joint constraints and a point-based object representation, then gradually tightens the simulator towards rigid-body physics." paper="2404.07988" >}}

##### DexScrew [arXiv'25] {#dexscrew}

*Learning Dexterous Manipulation Skills from Imperfect Simulations* · [arXiv](https://arxiv.org/abs/2512.02011) · [Project](https://dexscrew.github.io/) · [Code](https://github.com/x-robotics-lab/dexscrew) · UR5e + 12-DoF XHand + fingertip touch

DexScrew studies nut screwing and screwdriver use without simulating threads. The nut or screwdriver handle is simply a rigid body attached to a base by a revolute joint (Figure 31). Training has three stages (Figure 32). Privileged PPO first learns finger rotation in the simplified simulator, then is distilled into a policy using proprioceptive history. On the robot, an operator controls the wrist with a VR controller while invoking the learned finger skill to collect data. Finally, behavior cloning uses those real-robot data together with touch.

Screwdriver task progress is 41.6% with direct simulation transfer, with no trial reaching full completion. Behavior cloning without touch or history reaches 69.2%; with both, it reaches 95.0%.

{{< dex-figure src="dexscrew_model.jpg" narrow=true caption="Figure 31. DexScrew's simplified object: a nut or handle connected to a fixed base by a revolute joint, without thread mechanics." paper="2512.02011" >}}

{{< dex-figure src="dexscrew_overview.jpg" caption="Figure 32. DexScrew trains RL in imperfect simulation, uses the learned skill to assist real-robot teleoperation, then trains behavior cloning on multimodal real data." paper="2512.02011" >}}

> **Main takeaway**
>
> Simulation can teach a useful skill without reproducing every contact detail. That skill can then help collect the real data needed to learn what the simulator omitted.

## 4 Comparing the changes {#s8}

The six groups can be compared by what they change in the RL problem and whether that change can alter the original task's optimal solution.

| Method family | What changes | Can it change the original optimum? | Main difficulty | Form of human prior | Examples |
| --- | --- | --- | --- | --- | --- |
| Reset distribution | $\rho_0$ | Not the statewise optimum under the assumptions in Section 3.1 | 2.1, 2.2 | Pre-grasp poses | OmniReset, DemoStart, SGS, Penspin, Human2Sim2Robot |
| Goal distribution | Distribution of $g$ | Yes: a family of goal-reaching tasks replaces the downstream objective | 2.1, 2.2 | Object trajectories | SimToolReal, Play2Perfect |
| Structured exploration | Policy noise or action representation | Noise alone need not; restricting actions can | 2.3, 2.4 | Hand synergies | EigenDEXplore, DextrAH-G, SAR, DexPBT, SAPG |
| Motion tracking | Reward $r$, often resets too | Yes: preferences for a reference are added | 2.1–2.3 | Full hand–object trajectories | ManipTrans, DexTrack, DexMachina, DONIC, AdaDex, TeleDexter |
| Contact priors | Reward $r$ | Yes: specified contacts are favored | 2.2 | Contact locations | Twisting Lids, humanoid manipulation |
| Modified physics | Transition $P$ | Yes: assistance must be removed or mismatch addressed | 2.1, simulation cost | Assistance along a demonstration | DexMachina, QuasiSim, DexScrew |

**Changing sampling versus changing the task.** Reset selection and exploration noise can change where samples come from without changing the deployment task. Population training can add diversity, although mutating reward coefficients also changes what individual members optimize. These ideas can be combined, provided useful reset states or noise directions are available. Tracking, contact rewards and modified physics add stronger guidance, but also change the objective or dynamics. It helps to ask how the policy will cope when that guidance is removed: DexMachina decays assistance to zero; Play2Perfect fine-tunes on the downstream task; DexScrew learns from real data. Residual policies in ManipTrans and DONIC can correct a reference, but do not by themselves remove a tracking reward's preference for it.

**Where human data enters determines how much the embodiment gap matters.** Pre-grasp poses enter resets, object trajectories enter goals, hand synergies enter exploration, and full trajectories enter rewards. Resets and noise can guide learning without forcing the final motion to imitate a person. Object goals still change the task distribution, but avoid prescribing human finger motion. Full hand tracking constrains the solution more directly. The observations above fit this distinction: EigenDEXplore works best when human synergies affect noise alone; reduced eigen-space actions fall behind later; Human2Sim2Robot and ObjDex find limited value in finger tracking; DONIC and AdaDex obtain good whole-hand tracking on more human-like hands.

**Different forms of help can substitute for one another.** A component's benefit depends on the rest of the recipe. With diverse OmniReset resets, reward weights matter little. With near-goal SGS resets, a gravity curriculum becomes unnecessary. EigenDEXplore only helps DexMachina after its curriculum and auxiliary rewards are removed. After Play2Perfect pretraining, sparse task rewards suffice. An ablation establishes what matters within one setup; it does not automatically establish what matters most across papers.

**What I still want to see.** Among manipulation tasks, the broad reset-generation results in OmniReset and SGS are demonstrated with grippers. Generating large numbers of stable multifinger states, especially bimanual contact states, remains much harder. Human pre-grasp poses and bimanual motion capture seem useful here. Tracking methods have those priors but not the reset coverage of OmniReset; broad-reset methods have coverage but make less use of human data. Combining human-generated contact states and goals with learning-progress-based reset sampling looks worth exploring more systematically.

## References {#ref}

- Rajeswaran et al. Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations (DAPG). RSS 2018. [arXiv:1709.10087](https://arxiv.org/abs/1709.10087) · [Project](https://sites.google.com/view/deeprl-dexterous-manipulation) · [Code](https://github.com/aravindr93/hand_dapg).

- Petrenko et al. DexPBT: Scaling up Dexterous Manipulation for Hand-Arm Systems with Population Based Training. RSS 2023. [arXiv:2305.12127](https://arxiv.org/abs/2305.12127) · [Project](https://sites.google.com/view/dexpbt) · [Code](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs).

- Berg et al. SAR: Generalization of Physiological Agility and Dexterity via Synergistic Action Representation. RSS 2023. [arXiv:2307.03716](https://arxiv.org/abs/2307.03716) · [Project](https://sites.google.com/view/sar-rl) · [Code](https://github.com/MyoHub/myosuite/blob/main/docs/source/tutorials/SAR/SAR_tutorial.ipynb).

- Chiappa et al. Latent Exploration for Reinforcement Learning (Lattice). NeurIPS 2023. [arXiv:2305.20065](https://arxiv.org/abs/2305.20065) · [Code](https://github.com/amathislab/lattice).

- Singla, Agarwal, Pathak. SAPG: Split and Aggregate Policy Gradients. ICML 2024. [arXiv:2407.20230](https://arxiv.org/abs/2407.20230) · [Project](https://sapg-rl.github.io/) · [Code](https://github.com/jayeshs999/sapg).

- Lum et al. DextrAH-G: Pixels-to-Action Dexterous Arm-Hand Grasping with Geometric Fabrics. CoRL 2024. [arXiv:2407.02274](https://arxiv.org/abs/2407.02274) · [Project](https://sites.google.com/view/dextrah-g) · [Code](https://github.com/NVlabs/DEXTRAH).

- Lin et al. Twisting Lids Off with Two Hands. CoRL 2024. [arXiv:2403.02338](https://arxiv.org/abs/2403.02338) · [Project](https://toruowo.github.io/bimanual-twist/) · [Code](https://github.com/ToruOwO/twisting-lids).

- Wang et al. Lessons from Learning to Spin “Pens”. CoRL 2024. [arXiv:2407.18902](https://arxiv.org/abs/2407.18902) · [Project](https://penspin.github.io/) · [Code](https://github.com/HaozhiQi/penspin).

- Liu et al. QuasiSim: Parameterized Quasi-Physical Simulators for Dexterous Manipulations Transfer. ECCV 2024. [arXiv:2404.07988](https://arxiv.org/abs/2404.07988) · [Project](https://meowuu7.github.io/QuasiSim/) · [Code](https://github.com/Meowuu7/QuasiSim).

- ObjDex: Object-Centric Dexterous Manipulation from Human Motion Data. 2024. [arXiv:2411.04005](https://arxiv.org/abs/2411.04005) · [Project](https://cypypccpy.github.io/obj-dex.github.io/) · [Code](https://github.com/cypypccpy/ObjDexEnvs).

- HOP: Hand-Object Interaction Pretraining from Videos. 2024. [arXiv:2409.08273](https://arxiv.org/abs/2409.08273) · [Project](https://himanshusingh.ai/hop/) · [Code](https://github.com/hgaurav2k/hop).

- Bauza et al. DemoStart: Demonstration-led Auto-curriculum Applied to Sim-to-Real with Multi-fingered Robots. ICRA 2025. [arXiv:2409.06613](https://arxiv.org/abs/2409.06613) · [Project](https://sites.google.com/view/demostart).

- Liu et al. DexTrack: Towards Generalizable Neural Tracking Control for Dexterous Manipulation from Human References. ICLR 2025. [arXiv:2502.09614](https://arxiv.org/abs/2502.09614) · [Project](https://meowuu7.github.io/DexTrack/) · [Code](https://github.com/Meowuu7/DexTrack).

- Li et al. ManipTrans: Efficient Dexterous Bimanual Manipulation Transfer via Residual Learning. CVPR 2025. [arXiv:2503.21860](https://arxiv.org/abs/2503.21860) · [Project](https://maniptrans.github.io/) · [Code](https://github.com/ManipTrans/ManipTrans).

- Yin et al. DexterityGen: Foundation Controller for Unprecedented Dexterity. 2025. [arXiv:2502.04307](https://arxiv.org/abs/2502.04307) · [Project](https://zhaohengyin.github.io/dexteritygen/).

- Lum et al. Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration (Human2Sim2Robot). 2025. [arXiv:2504.12609](https://arxiv.org/abs/2504.12609) · [Project](https://human2sim2robot.github.io/) · [Code](https://github.com/tylerlum/human2sim2robot).

- Lin et al. Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids. CoRL 2025. [arXiv:2502.20396](https://arxiv.org/abs/2502.20396) · [Project](https://toruowo.github.io/recipe/).

- Hsieh et al. Learning Dexterous Manipulation Skills from Imperfect Simulations (DexScrew). 2025. [arXiv:2512.02011](https://arxiv.org/abs/2512.02011) · [Project](https://dexscrew.github.io/) · [Code](https://github.com/x-robotics-lab/dexscrew).

- Yin et al. Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning (OmniReset). ICLR 2026. [arXiv:2603.15789](https://arxiv.org/abs/2603.15789) · [Project](https://weirdlabuw.github.io/omnireset/) · [Code](https://github.com/UW-Lab/UWLab).

- A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control (SGS). CoRL 2026. [arXiv:2610.12465](https://arxiv.org/abs/2610.12465) · [Project](https://sgs-rl.github.io/).

- Beukman et al. Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments. RLC 2026. [arXiv:2603.06009](https://arxiv.org/abs/2603.06009).

- Mandi et al. DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation. ICML 2026. [arXiv:2505.24853](https://arxiv.org/abs/2505.24853) · [Project](https://project-dexmachina.github.io/) · [Code](https://github.com/MandiZhao/dexmachina).

- Kedia et al. SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation. 2026. [arXiv:2602.16863](https://arxiv.org/abs/2602.16863) · [Project](https://simtoolreal.github.io/) · [Code](https://github.com/tylerlum/simtoolreal).

- Lum et al. Play2Perfect: What Matters in Dexterous Play Pretraining for Precise Assembly? 2026. [arXiv:2606.26428](https://arxiv.org/abs/2606.26428) · [Project](https://play2perfect.github.io/) · [Code](https://github.com/kushal2000/play2perfect).

- Paliwal et al. Do as I Do: Dexterous Manipulation Data from Everyday Human Videos. 2026. [arXiv:2606.19333](https://arxiv.org/abs/2606.19333) · [Project](https://www.do-as-i-do.com/) · [Code](https://github.com/malik-group/do-as-i-do).

- Li et al. Towards Human-level Dexterous Teleoperation (TeleDexter). 2026. [arXiv:2607.11481](https://arxiv.org/abs/2607.11481) · [Project](https://bigai-dex.github.io/blog/teledexter/).

- Gupta et al. EigenDEXplore: Structured Exploration for Dexterous Manipulation with Human Priors. 2026. [arXiv:2610.07681](https://arxiv.org/abs/2610.07681) · [Project](https://eigendexplore.github.io/) · [Code](https://github.com/hgupt3/eigendexplore).

- Kuang et al. AdaDex: Learning Adaptable Dexterous Skills via Hybrid Dexterous Mimicking. 2026. [arXiv:2610.28628](https://arxiv.org/abs/2610.28628) · [Project](https://yuxuank.com/AdaDex/).

- Sun et al. DONIC: Generalist Dexterous Bimanual Tracking Control from Human Demonstrations. 2026. [arXiv:2610.29997](https://arxiv.org/abs/2610.29997) · [Project](https://ropedia.github.io/DONIC/) · [Code](https://github.com/Ropedia/DONIC).

- Pan et al. SPIDER: Scalable Physics-Informed Dexterous Retargeting. 2025. [arXiv:2511.09484](https://arxiv.org/abs/2511.09484) · [Project](https://jc-bao.github.io/spider-project/) · [Code](https://github.com/facebookresearch/spider).

Figures are reproduced from the cited papers. Copyright remains with their authors.
