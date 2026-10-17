---
title: "How to train your Robo-centric VLMs"
date: 2026-10-18T00:00:00+08:00
draft: false
ShowToc: false
math: true
guideLang: en
tags: ["robotics", "vision-language-models", "embodied-ai", "data", "paper-reading"]
summary: "Reading notes on Robo-centric VLMs: data construction, spatial grounding, multi-task training, and closed-loop robot execution."
description: "Reading notes on embodied models and the data behind them."
---

{{< robo-header >}}

## What I would look at first {#takeaways}

> **The main takeaway**
>
> After reading these reports, I think the recipes for Robo-centric VLMs are becoming quite similar. Start with a general VLM, give boxes, points and trajectories a common coordinate interface, and build millions of examples from public datasets, geometry, videos and model-generated annotations. HY, RynnBrain and GroundingPI also use verifiable rewards for GRPO; RoboBrain 2.5 is mainly a two-stage supervised training recipe. The more interesting differences are in **which abilities each team spends its data budget on**.

The first four models below have released weights, including versions worth trying on a single RTX 5090. That does not mean every checkpoint fits in BF16: quantization, context length and video frames still matter.

- **HY-Embodied, Tencent.** The most systematic data engineering of the four, in my view. Perception enters pretraining early: 62M detection examples and 36M depth examples. Data mixtures follow a capability taxonomy, and post-training runs through RL → rejection sampling → RL. The training data is not released.
- **RynnBrain, Alibaba DAMO Academy.** Keep the base architecture and work on the data: first-person videos, SLAM, human pointing annotations, and plans with coordinates. Version 1.1 adds contact points and 3D boxes, and has been deployed on a Tianji arm with a Wuji dexterous hand. The released weights use Apache-2.0; code and benchmarks are available too.
- **RoboBrain, BAAI.** Two representation choices are particularly useful. Image coordinates plus depth, `(u, v, d)`, let 2D and 3D trajectory data share a format. A normalized progress update, called a *hop*, turns the VLM into a dense reward model. The latter uses a data pipeline that produces roughly 35M progress examples before downsampling.
- **GroundingPI, XPeng and collaborators.** A focused 4B grounding backbone. The paper reports 73.68% averaged over 34 grounding benchmarks, versus 71.54% for GPT-6 Astra. Its data engine combines teachers field by field, validates annotations by task, and improves through an expert model. The ablations make a strong case for dense detection and pointing as preparation for action learning.
- **Catching up with Astra.** Among the works in these notes, GroundingPI makes the direct comparison on grounding. Closed-loop execution is a different test: RobotWorld reports Astra completing only 16 of 84 tasks.

## What does a Robo-centric VLM output? {#background}

A Robo-centric VLM, often called an *embodied foundation model*, is usually a general multimodal model trained further on embodied tasks. Its useful outputs describe what an action needs: which object, where it is, where to grasp, which path to follow, what to do next, and how much of the task is done.

Two common uses are to make it a high-level planner that sends grounded subgoals to a low-level policy, or to use its weights to initialize a VLA backbone. HY, RynnBrain and GroundingPI all evaluate the second route. Progress prediction gives a third use: a reward model.

In these notes, *Astra* refers to OpenAI's GPT-6 Astra. Reports describe a substantial improvement in spatial reasoning, especially in 3D tasks, but its training recipe is not public. “Catching up” here means catching up on particular embodied abilities—grounding, spatial perception, planning and progress estimation—not matching every aspect of the model.

| Term | Meaning here |
| --- | --- |
| Grounding | Locate the object described in a query, typically with a bounding box. |
| Pointing | Mark one or more target points. Useful for grasp locations, placement regions and counting. |
| Affordance | A place or part that supports an operation: a handle, a button, or an empty region where an object can be placed. |
| Trajectory / trace | A sequence of waypoints for an end effector or object, in an image or in 3D. |
| Value / progress estimation | Estimate task completion from observations; changes in progress can provide a dense reward. |
| CoT / cold start | First teach a model to produce reasoning with a small supervised set, then apply RL. |
| GRPO | Sample several answers to the same prompt and estimate advantages from their relative rewards. Used by HY, RynnBrain and GroundingPI here. |
| Rejection-sampling fine-tuning (RFT) | Generate answers with a model, keep the correct ones, and train on them. |

## From data to a robot {#recipe}

The common interface is straightforward: an image or video and an instruction go in; text and coordinates come out. The coordinates might specify a box, a grasp point, a placement point, or a path. A large part of the work is making enough reliable data in these formats.

{{< robo-figure id="0" caption="The shared recipe: data sources, annotation, staged training, and three uses on a robot. GRPO is an additional stage in several of these systems, not a requirement shared by all four." >}}

Read the figure from top to bottom. The first two stages build data, the third trains the model, and the last puts it to use. HY adds an RL → rejection sampling → RL loop to training. RynnBrain contributes a substantial first-person video collection. RoboBrain develops dedicated 3D trajectory and progress pipelines. GroundingPI builds a multi-teacher annotation engine around a narrower objective: precise grounding.

### Task inputs and outputs {#tasks}

A “task” in these papers is often a question-answer format. The input is still an image or video plus a query. What changes is the answer: a box, points, a path, a number, or a grounded plan. The twelve examples below cover the main task types in the notes.

{{< robo-figure id="1" caption="Tasks 1–6: perception and spatial understanding. Coordinates and values are illustrative." >}}

{{< robo-figure id="2" caption="Tasks 7–12: manipulation and temporal understanding. The output can specify a contact, a path, a next step, or progress." >}}

Tasks 1–6 ask what is present, where it is and how far away it is. Tasks 7–10 describe where to grasp, where to place and how to move. Tasks 11–12 describe the next step and the progress so far.

Image coordinates are usually quantized to roughly a thousand positions and generated alongside text. This makes different tasks compatible with one autoregressive interface, but the exact representation still matters. Absolute depth keeps its physical units; 3D boxes and masks have their own formats. GroundingPI specifically uses dedicated `<0>`–`<999>` coordinate tokens.

### Three ways to connect it to a robot {#deployment}

A VLM that predicts points or plans still needs an execution mechanism. The three arrangements differ in where the model sits in the control system.

{{< robo-figure id="3" caption="A Robo VLM can plan above a policy, initialize a VLA backbone, or estimate progress for a reward. The blue box identifies its role in each arrangement." >}}

Backbone replacement is a common experiment: keep the action head and demonstration data fixed, swap the VLM, and measure success and generalization. GroundingPI reports that stronger grounding makes better use of a small demonstration budget. The individual tasks have fairly concrete uses:

| Tasks | Use on a robot |
| --- | --- |
| 1–2: detection and pointing | Specify which object the policy should manipulate. GroundingPI's ablations find these abilities especially useful for VLA transfer. |
| 3, 4, 6: depth, spatial QA and 3D boxes | Turn “10 cm to the left” or “how far away?” into metric targets. Camera intrinsics allow image coordinates and depth to be lifted to 3D. |
| 5: video localization | Recall where an object appeared and in which frame, for retrieval or navigation. |
| 7–8: grasp points and placement regions | Supply targets to a motion planner or condition a VLA. |
| 9–10: 2D and 3D trajectories | Provide a reference path, either as a VLA intermediate representation or for a controller to track. |
| 11: grounded planning | Send the next subtask and its coordinates to the low-level executor. |
| 12: progress estimation | Supply dense reward, detect that execution has stalled, or help decide whether to retry. |

## Models and training recipes {#models}

### HY-Embodied: put perception into pretraining {#hy-embodied}

Tencent Robotics X and Hunyuan Vision put substantial detection, depth, segmentation and pointing supervision into pretraining. Embodied and spatial QA follow in mid-training, then reasoning-oriented post-training.

The two reports are [HY-Embodied-0.5](https://arxiv.org/abs/2604.07430), released in April 2026 with MoT-2B and A32B variants, and [Hy-Embodied-VLM-1.0](https://arxiv.org/abs/2607.12894), released in July with a Hy3-A3B MoE backbone and roughly 30B total parameters.

**Model and training.**

- Version 0.5 combines Hunyuan-1.8B with HY-ViT-2.0, a 400M vision encoder. Its Mixture-of-Transformers routes visual and text tokens through different parameters. It also introduces latent visual tokens, with visual reconstruction and global alignment losses during pretraining.
- The sequence is pretraining on over 600B tokens → embodied/spatial mid-training on about 25–30M examples → roughly 100K cold-start CoT examples → GRPO → rejection-sampling self-improvement. The mid-training ratio is general : embodied : spatial = 12 : 5 : 3. The small model finally receives on-policy distillation from the 32B model.
- Version 1.0 replaces the backbone with Hy3-A3B MoE, activating about 3B parameters, while reusing the 0.5 pretraining data. The additions are in SFT and RL: run RL to elicit reasoning, generate about 1M long CoT examples with that policy, restart SFT from the clean pretrained checkpoint, then run another reward-focused RL stage.

**How the data is made.**

Pretraining contains roughly 625B tokens: 389B general and 236B embodied/perceptual. Spatial and robot data account for 43% of the latter; the remainder is visual perception.

{{< robo-figure id="4" caption="HY-Embodied-0.5, Fig. 4. Pretraining data on the left; mid-training data on the right." source="https://arxiv.org/abs/2604.07430" >}}

| Data | Scale | Construction |
| --- | --- | --- |
| Omni-Detection: 2D / 3D detection | 62M | OpenImages, Objects365, RefCOCO and SA-1B. Convert good annotations directly. For missing or weak labels, identify objects with a VLM, localize with SAM plus VLM grounding, then verify with a stronger VLM. Normalize coordinates to integers on a 0–1000 scale. |
| Depth | About 36M | Use indoor/outdoor 3D datasets and autonomous-driving sensor ground truth to ask about absolute or relative depth at selected pixels. Avoid boundaries and infinite-depth regions; normalize focal length. |
| Segmentation | About 5M | Filter SA-1B masks and extend the vocabulary in the manner of PaliGemma so that the model can output masks. |
| Pointing and counting | About 11M | Pixmo-Points and related sources, with particular attention to crowded scenes. |
| Embodied grounding, affordance, trajectories, understanding, planning and reasoning | About 12M QA in mid-training | Combine Molmo, RoboPoint, RefSpatial, RoboAfford, ShareRobot, MolmoAct and RoboVQA with automatic annotation. For trajectories, track arms in Open-X / DROID with CoTracker3, downsample to at most 15 waypoints, overlay them on the first frame, and ask a stronger VLM to check them. For planning, annotate tasks and split videos into next-action targets. For affordances, generate user instructions from existing grounding labels. |
| Spatial correspondence, geometry, configuration, measurement and dynamics | Within the same mid-training mixture | Generate rule-based QA from ScanNet, ScanNet++, ARKitScenes and collected RGB-D sequences with poses. Use the longest 3D-box edge for size and the minimum distance between two boxes for separation. Discard ambiguous cases, including distances below 0.2 m or depth differences below 0.3 m. |

Version 1.0 adds SFT/RL data organized around state understanding, action transitions, and sequences with error recovery:

- **Depth reasoning.** Project LiDAR points into images, retaining at most 100 candidates per image. Normalize focal length to 1000 px, mark the queried point with a red arrow and green dot, and synthesize CoT.
- **Social interaction.** Manually annotate interaction partners, gaze, pointing, pose and emotion in first-person video; add AVA and HICO-DET.
- **Causal reasoning.** Split Ego4D and similar videos into steps with preconditions and effects. Align each step to a time interval, then produce feasibility and effect-prediction questions.
- **Failure awareness.** Use robot failure videos and perturbed plans to ask what went wrong and how to recover.
- **Vision-language navigation.** In Habitat R2R-CE, collect shortest-path oracle demonstrations, then use DAgger to add states the model reaches after deviating from the path.

{{< robo-figure id="5" caption="Hy-Embodied-VLM-1.0, Fig. 3. SFT on the left and RL on the right; the inner rings group state, transition and reasoning tasks." source="https://arxiv.org/abs/2607.12894" >}}

**RL examples are selected dynamically.** Sample several answers from the current model for each candidate. Discard prompts it always gets right and prompts it always gets wrong; keep the partially solved ones. Balance perception, prediction, interaction and planning, with 50K examples per round. The reward follows the output structure.

{{< robo-figure id="6" caption="HY-Embodied-0.5, Fig. 6. Boxes use IoU, points use normalized L2, trajectories use DTW plus endpoint accuracy, and numbers use relative error. Open-ended questions use a model judge." source="https://arxiv.org/abs/2604.07430" >}}

**Results.** Version 0.5 evaluates 22 benchmarks; MoT-2B beats similarly sized models on 16, and the report describes A32B as approaching Gemini 3.0 Pro. The team also connects a VLA to the Xtrainer dual-arm platform. Version 1.0 evaluates 38 benchmarks, leads similarly sized models on 19, and reports an average improvement of 8.4% over 0.5 MoT-2B, as well as state-of-the-art RGB-only R2R-CE results.

**Weights and deployment.** [HY-Embodied-0.5](https://huggingface.co/tencent/HY-Embodied-0.5) releases MoT-2B, about 3.8B total parameters, under a license listed as `other`; the 32B model was not released in the material reviewed. [Hy-Embodied-VLM-1.0](https://huggingface.co/tencent/Hy-Embodied-VLM-1.0) is Apache-2.0, has 30.5B total parameters, and supports Transformers / vLLM. Its BF16 weights alone require roughly 61 GB, so running it on a 5090 would require quantization. The training data is not released.

**My take.** Three things are worth borrowing: introduce perception during pretraining, use a capability taxonomy to control the mixture, and keep refreshing RL data with partially solved prompts. Much of the pipeline is public data plus automatic annotation and stronger-model verification. Manual work is concentrated in social annotations and a small amount of CoT.

### RynnBrain: ground space in video and coordinates {#rynnbrain}

Alibaba DAMO Academy builds on Qwen-VL with spatiotemporal and physically grounded supervision. Boxes, points, trajectories and grasps are represented with integer coordinates, and frame indices tell the model *where in the video* a location belongs.

[RynnBrain 1.0](https://arxiv.org/abs/2602.14979), from February 2026, uses Qwen3-VL and comes in 2B, 8B and 30B-A3B sizes. [RynnBrain 1.1](https://arxiv.org/abs/2607.17977), from July, moves to Qwen3.5 with 2B, 9B and 122B-A10B variants.

**Model and training.**

- The architecture stays Qwen3-VL / Qwen3.5. The main changes are in data; an engineering addition is online load balancing by sequence length.
- Version 1.0 derives four variants after embodied pretraining: CoP (Chain-of-Point, interleaving reasoning with coordinates), Nav, Plan and VLA.
- Version 1.1 adds contact-point prediction and native 3D boxes; the latter are supported by the 2B / 9B models. The VLA uses a unified 81-dimensional action space with embodiment masks. Deployments include Unitree G1, Astribot S1, and **a Tianji arm with a Wuji dexterous hand**. The report explicitly acknowledges WUJI TECH's hardware support.

**How the data is made.**

Version 1.0 contains 19.89M pretraining examples: general data (4.8M), cognition (about 7.7M), localization (about 7.3M), and planning (0.16M). Datasets prefixed with “RynnBrain-” are built by the authors.

{{< robo-figure id="7" caption="RynnBrain 1.0, Table 2. The pretraining mixture." source="https://arxiv.org/abs/2602.14979" >}}

The custom pipeline combines collected first-person indoor video, public tools and human filtering:

- **Object understanding.** Qwen2.5-VL identifies objects → Grounding DINO 1.5 detects them in keyframes → SAM2 segments and tracks → Qwen2.5-VL generates QA → humans filter. This produces 712K examples, with at most two instances per object category in each video.
- **Spatial understanding.** MASt3R-SLAM reconstructs point clouds and camera poses from RGB video. Project instance masks into 3D, use RANSAC to find the floor and align gravity, then generate distance, height and orientation questions from templates. The result is 855K video and 272K image examples.
- **OCR.** GoMatching finds 85K text-containing clips in Ego4D / EPIC-KITCHENS. Humans annotate the first appearance, clearest frame, transcription and polygon; GPT-5.2 and templates expand these into about 1M QA pairs.
- **Regions, affordances and trajectories.** Each video subset has only about 6K examples: an LLM writes instructions and humans click points or draw paths. Corresponding image subsets contain roughly 450–510K each. Trajectories have at most ten points.
- **Grasping.** Convert Grasp-Anything's rotated rectangles to four corners. Mix object-oriented, scene-oriented and task-oriented instructions at 40%, 30% and 30%, respectively, for 1.3M examples.
- **Planning.** Following Hi Robot, split long tasks into atomic actions using an internal model and human verification. Annotate each subtask with a target box, placement-region points and affordance points. The plan itself is a mixture of text and coordinates.
- **CoP reasoning.** Qwen3-VL-235B writes a reasoning chain with entities in brackets. An internal model classifies each as a region or object. Humans choose the clearest frame and annotate points or boxes; the coordinates are then inserted back into the chain.
- **RL.** Score candidates with the SFT model and retain examples scoring 40–80, plus failures caused by selecting the wrong keyframe, for 30K examples. Rewards use Fréchet distance for trajectories, bidirectional Chamfer distance for affordances, and point-in-polygon tests for regions.

The 1.1 data update is shown in blue below.

{{< robo-figure id="8" caption="RynnBrain 1.1, Table 2. Blue marks new or substantially expanded sources relative to 1.0." source="https://arxiv.org/abs/2607.17977" >}}

- **3D boxes.** WildDet3D lifts COCO, LVIS and Objects365 boxes to 3D: 374K human-verified examples in the Essential subset and 888K in Synthetic. FoundationPose contributes 1,849K synthetic images with noise-free 3D boxes, regularizing metric scale. Qwen-VL generates referring expressions.
- **Contact points.** Replace four-corner grasp rectangles with a center point and gripper angle. Rectangle size depends on the gripper and annotation convention, and rectangle IoU need not correlate with execution success. Select observable camera views from grasp data such as GraspNet-1B, render and project the grasps, and obtain 2.6M examples.
- **Planning.** Add Galaxea-G0, using Gemini 3.1 Pro to infer high-level task descriptions from subtask labels.

**Results.** Version 1.0 reports substantial improvements over earlier embodied models across 20 embodied and eight general benchmarks. Its Nav variant reports state-of-the-art R2R / RxR results, and its VLA variant outperforms fine-tuned π0.5 on complex grasping. The authors also introduce RynnBrain-Bench.

Version 1.1's 122B-A10B model leads the evaluated open and closed models on VSI-Bench, MMSI and RefSpatial-Bench. One result I find particularly interesting is that, on spatiotemporal pointing, even the largest Qwen3.5 model falls below the smallest RynnBrain 1.1 model. For this task, the data seems to matter more than simply increasing model size.

**Weights and deployment.** Code, weights and benchmarks are released. The listed [RynnBrain-8B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-8B), [RynnBrain1.1-9B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-9B) and [122B-A10B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-122B-A10B) checkpoints use Apache-2.0. The 9B BF16 weights alone are about 18 GB, leaving runtime fit dependent on KV cache and visual inputs. The reports do not clearly establish whether all custom training data is released; that needs checking separately in the repository.

**My take.** This is a relatively plain recipe: retain the architecture, expand data coverage, and standardize spatial outputs. I would borrow the first-person video → SLAM → template QA pipeline and the grounded planning format. The Tianji-Wuji deployment also makes version 1.1 particularly relevant to dexterous manipulation.

### RoboBrain: 3D trajectories and task progress {#robobrain}

BAAI's [RoboBrain 2.5](https://arxiv.org/abs/2601.14352), from January 2026, adds two capabilities to the earlier [RoboBrain 2.0](https://arxiv.org/abs/2507.02029) family: metric 3D reasoning from monocular RGB, and dense task-progress estimation that can serve as a reward signal. Version 2.0 had 7B / 32B variants; the 2.5 report centers on Qwen3-VL-8B.

**Model and training.**

Training updates all parameters in two stages. Basic spatiotemporal learning uses 8.3M examples of general semantics, 2D grounding and open-loop planning. Specialized enhancement then uses 4.1M examples for metric 3D reasoning and dense value estimation. The same configuration is trained on both NVIDIA and Moore Threads GPUs.

The representation choices are the part I would pay most attention to:

- **3D trajectories use `(u, v, d)`.** Instead of predicting camera-frame `(x, y, z)` directly, predict image coordinates plus absolute depth. With camera intrinsics, these can be lifted into 3D. Remove `d` and the output is a 2D trajectory; retain only the endpoints and it becomes a pointing/grounding target. Existing 2D data remains useful in the same mixture.
- **Progress uses a normalized hop.** Predict progress between two frames relative to the remaining distance when moving forward, or to the distance already covered when moving backward. Accumulated progress stays in `[0, 1]`. At inference, combine incremental estimates with estimates anchored at the start and at the goal. Disagreement between the two anchors is treated as an out-of-distribution signal, prompting a conservative update to reduce reward exploitation.

**How the data is made.**

The roughly 12.4M training examples cover general, spatial and temporal domains. In the right-hand plot below, the sample axis is logarithmic. Dense value estimation, high-quality general data and 3D spatial reasoning are the largest components.

{{< robo-figure id="9" caption="RoboBrain 2.5, Fig. 2. Data hierarchy on the left (temporal in red, general in cyan, spatial in blue); subtask counts on a logarithmic scale on the right." source="https://arxiv.org/abs/2601.14352" >}}

| Data | Scale | Construction |
| --- | --- | --- |
| General | 2.83M | Honey-Data-1M with long CoT removed, retaining answers; LLaVA-OneVision-1.5 with text-only examples removed and balanced sampling; deduplicate afterward. |
| Grounding / pointing / affordance | 86K conversations; 190K; 561K + 320K | Convert LVIS boxes into multi-turn conversations. Remove Pixmo-Points examples with more than ten points; use GPT-4o to retain indoor objects and pair them with 28 templates. Use PACO-LVIS part boxes and GPT-4o functional questions, plus RoboPoint placement regions. |
| Spatial understanding | 826K | Follow RefSpatial: filter OpenImages from 1.7M to 466K → RAM tags and GroundingDINO boxes → Qwen2.5-VL descriptions of increasing specificity, such as “the third cup from the left” → UniDepth V2 depth and WildCamera intrinsics → SAM 2.1 masks → pseudo-3D scene graphs → template/QwQ questions. Add ScanQA, SQA3D and CA-1M, filtering the latter from 2M frames to 100K. |
| Spatial referring | 802K | Sample description–point pairs from scene graphs. For placement, generate top-down occupancy maps from 3D data and metric queries such as “10 cm to the right of the chair.” |
| Metric 3D reasoning, new | 1.74M examples / 8.08M QA | Follow TraceSpatial. Use CA-1M and ScanNet for metric object traces. Filter AgiBot-Beta from 167K to 59K and DROID from 116K to 24K; add RoboTwin 2.0 simulation. Qwen3-VL decomposes subgoals for multistep single-arm and bimanual 3D trajectories. |
| Planning | 1M + 9K + 50K + 44K | ShareRobot, manually annotated across 12 embodiments; AgiBot Planning; EgoPlan-IT; and DeepSeek-V3-generated multi-robot plans across 1,659 task types. |
| Closed-loop interaction | Not separately specified | Synthesize observation–thought–action trajectories in AI2-THOR; GPT-4o supplies the thought traces. |
| Dense value estimation, new | About 35M, downsampled to 3.5M | Follow Dopamine-Reward, starting from over 27M raw frames. Approximately 60% real robots (AgiBot-World, DROID, RoboBrain-X), 13% simulation (LIBERO, RoboCasa, RoboTwin), and 26% human video (EgoDex). Humans mark multi-view keyframes to split expert demonstrations into subtasks. Sample within segments, balance hop bins, and include zero-progress pairs. |

**Results.** The report compares 3D reasoning and value estimation against Gemini-3-Pro, GPT-5.2, Mimo-Embodied-7B and RoboBrain 2.0, and reports leading results on these evaluations. It includes TraceSpatial-Bench, RoboTwin 2.0 trajectory visualizations, and real-robot demonstrations where predicted progress falls after an external disturbance.

**Weights and deployment.** Hugging Face hosts [RoboBrain2.5-8B-NV](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV), the NVIDIA-trained checkpoint, alongside 8B-MT and 4B variants under Apache-2.0. Version 2.0 also has a 32B release. An 8B checkpoint needs about 16 GB for BF16 weights alone; KV cache and visual inputs add to that.

**My take.** The useful ideas are the two representations: `(u, v, d)` keeps 2D and 3D supervision compatible, and hop-based progress makes a VLM useful as a dense reward model. The latter is especially relevant to dexterous-hand RL. The data recipe relies heavily on public sources plus RefSpatial, TraceSpatial and Dopamine-Reward, with substantial filtering and relatively focused manual annotation.

### GroundingPI: get the location right first {#groundingpi}

[GroundingPI](https://arxiv.org/abs/2609.39601), released on September 30, 2026 by XPeng, Peking University, HKU, Tsinghua and collaborators, takes a narrower approach. Train a 4B grounding backbone across eleven perceptual capabilities—detection, referring, pointing, OCR, GUI and others—and test whether it improves manipulation and driving. Siqiao Huang from Tsinghua is among the authors.

It is also the direct Astra comparison in this set of papers: 73.68% versus 71.54%, averaged over 34 grounding benchmarks. These are **perception and localization results**, not a demonstration of stronger planning or general reasoning.

**Model and training.**

- MoonViT-V2, the Kimi K3 vision encoder, plus Qwen3-4B. A thousand quantized coordinate tokens, `<0>` through `<999>`, share the vocabulary with category names. If the requested category is absent, the output is `None`.
- The stages are projector alignment → general multimodal pretraining → video understanding → coordinate-alignment SFT across eight spatial tasks → GRPO. The reported training volumes are 500.41B + 221.17B tokens, on 128 GPUs.
- Detection reward combines **0.7 × set completeness + 0.3 × strict localization**. Completeness matches predictions to references by IoU; strict localization uses multi-threshold F1 to penalize duplicate and oversized boxes. OCR uses Hungarian matching to assess both transcription and position.

**How the data is made.**

The report does not provide a full data-mixture table. Its main contribution here is the data engine, shown in the lower half of the figure.

{{< robo-figure id="10" caption="GroundingPI, Figs. 2–3. Architecture and output sequences above; the data engine below." source="https://arxiv.org/abs/2609.39601" >}}

1. **Generate candidates.** Complementary teacher models annotate the same image and query. Category-level queries first discover categories, then localize instances of each category.
2. **Fuse by field.** Vote separately on category, box, mask and attributes. Each field is accepted, unresolved or rejected. Preserve part–whole and word–line containment rather than merging them as duplicates.
3. **Validate by task.** Accept an example only when all fields required by the query pass. “Find all instances” needs a coverage check. Absence must be verified too: an empty prediction is not evidence that an object is absent.
4. **Iterate with an expert.** Train a unified grounding expert on accepted annotations. Its predictions go through the same fusion and validation. Resolve uncertain fields with extra teachers or cropped views. Labels can be revised or withdrawn, with changes propagated to dependent tasks.
5. **Derive several tasks from one instance.** Category plus box gives detection; attributes and relations give referring; regions give pointing; correspondences give visual prompting; text regions and transcriptions give OCR.

For me, the most useful experiment is the data ablation. Split the perception data into six groups, remove each group in turn, and measure downstream manipulation and driving.

{{< robo-figure id="11" caption="GroundingPI, Fig. 10. A: basic detection; B: dense detection; C: referring; D: basic pointing; E: robotic pointing; F: OCR and related tasks. The right-hand plots show drops when groups are removed." source="https://arxiv.org/abs/2609.39601" >}}

- With only F, OCR/layout/GUI data, RoboCasa-GR1 OOD success is 7.28%. Adding A + D raises it to 29.28%. Basic localization and pointing account for much of this improvement.
- Removing dense detection, B, produces the largest drop in the reported ID/OOD manipulation comparison: −4.67 and −3.47 percentage points. The paper treats it as the spatial core.
- All six groups give the best combined result: 37.75% ID and 33.38% OOD. OCR alone is weak, but removing it from a richer mixture also causes a reported 5.33-percentage-point drop in the ablation. Its contribution depends on the rest of the data.

**Other results.** With the same Action DiT, action data and budget, GroundingPI leads all evaluated backbones on four RoboTwin 2.0 OOD settings, by up to **24.8% relative** to the strongest alternative. On RoboCasa-GR1, 50% of the demonstrations suffice to exceed other backbones trained on 75%. On nuScenes, average open-loop L2 error is 0.296 m, compared with 0.301 for Qwen3-VL-4B and 0.308 for RynnBrain.

Another result worth separating from the headline numbers: RynnBrain 1.1 underperforms 1.0 in all six manipulation settings in this evaluation. The authors suggest Qwen3.5's Gated DeltaNet may make spatial features less accessible to the action head. That is a proposed explanation, not an established causal result.

**Weights and deployment.** [GroundingPI/GroundingPI](https://huggingface.co/GroundingPI/GroundingPI) was uploaded on October 14, 2026. It has about 4.8B parameters, a license listed as `other`, and is a reasonable single-5090 candidate. GroundAnything repositories provide related work.

**My take.** Get “where” right, then let action data teach “how.” The field-wise fusion, task-wise validation and expert iteration are directly useful for automatic annotation. The ablations also give a practical starting point for the data mixture: protect dense detection and pointing before adding more specialized abilities.

### Embodied-R1.5: train for the failed attempt too {#embodied-r15}

[Embodied-R1.5](https://arxiv.org/abs/2606.11324), from Tianjin University and Tencent Hunyuan, extends the earlier pointing model into spatial reasoning, planning, correction and trajectories. The part I find most useful is **how it trains the model to deal with an unsuccessful execution**. A successful demonstration alone does not tell a planner what to do after a missed grasp.

**Model.** The [released model](https://huggingface.co/IffYuan/Embodied-R1.5) uses Qwen3-VL-8B-Instruct. Answers share a text interface: boxes and points use 0–1000 image coordinates; 3D traces add depth in meters. Structured answers sit inside `<answer>...</answer>` tags. A separate VLA adaptation attaches a flow-matching action expert. Grounded point predictions and continuous robot actions should therefore be distinguished when reading its results.

**How the data is made.** The full mixture combines 34 datasets and exceeds 15B tokens. Three custom pipelines fill particular gaps:

| Pipeline / custom subset | Construction |
| --- | --- |
| Tabletop spatial QA (About 20K) | Robot images → semantic labels, MoGe-2 geometry and instance masks → aligned 3D scene graph → geometric questions |
| Failure correction (About 800K) | Perturb plans; truncate or mismatch execution clips; inject simulated failures. Label detection, diagnosis and repair separately |
| Functional pointing and traces (About 400K) | Simulated object parts and existing affordances; project known end-effector poses or track grippers/objects; add articulated-object trajectories |

These counts describe the custom subsets, not the whole training mixture. The released [SFT dataset](https://huggingface.co/datasets/IffYuan/Embodied-R1.5-SFT-Dataset) and [RFT dataset](https://huggingface.co/datasets/IffYuan/Embodied-R1.5-RFT-Dataset) make the two stages separately inspectable.

**SFT → balanced RL.** SFT updates both the vision encoder and language model for one epoch. The RL pool retains roughly 200K prompts of intermediate difficulty; groups with identical rewards are masked. Each prompt gets eight rollouts. The advantage uses a group mean and a shared batch standard deviation:

{{< robo-math >}}
\widehat A_i=\frac{R_i-\mu_{\mathrm{group}(i)}}{\sigma_{\mathrm{batch}}+\epsilon}.
{{< /robo-math >}}

The subtraction compares answers to the same question. The denominator changes their scale relative to other groups in the batch. A common denominator alone does not guarantee equal influence for every task; task sampling and reward construction still matter.

Rewards follow the answer: exact match for determinate questions, IoU for boxes, distance for points, trajectory error for traces, and a learned scorer for open-ended plans or corrections. Distance-based scores use partial credit:

{{< robo-math >}}
\phi(d;\tau_p,\tau_z)=\operatorname{clip}\!\left(\frac{\tau_z-d}{\tau_z-\tau_p},0,1\right).
{{< /robo-math >}}

For point sets, the reported thresholds are 40 and 150 in normalized coordinates. For 3D traces, image-plane and depth scores are averaged; a one-point “trajectory” receives zero. The final reward mixes answer quality and format at 0.9 : 0.1. Here **RFT means reinforcement fine-tuning**, unlike HY's rejection-sampling use of the same abbreviation. [Training and reward definitions, §§4–5](https://arxiv.org/html/2606.11324v1#S4).

{{< robo-figure id="20" caption="Embodied-R1.5, Fig. 3. Planner, Grounder and Corrector share a model; observations and execution status feed the next decision." source="https://arxiv.org/abs/2606.11324" >}}

**How correction enters execution.** In the [PGC system](https://embodied-r.github.io/), one model serves three roles. The Planner chooses a subtask, the Grounder supplies targets for a low-level executor, and the Corrector checks whether execution succeeded, is still running, or failed. Recent observations and status labels stay in a FIFO memory. Failure feeds back into the next plan. This tests a different ability from answering a single grounding question correctly.

**What to take from it.** I would read the RL ablations and closed-loop examples together: better pointing can improve the first attempt, while correction data determines whether the system gets another useful attempt. The [official repository](https://github.com/pickxiguapi/Embodied-R1.5) links weights, both training datasets and EmbodiedEvalKit. That makes this a more concrete recipe to inspect than a benchmark score alone.


### ACE-Brain-0.5: keeping task interfaces from interfering {#ace-brain}

[ACE-Brain-0.5](https://arxiv.org/abs/2607.04426) addresses a problem that becomes visible once all these tasks share a model. Grounding emits coordinates, navigation emits actions, and progress estimation emits a temporal score. Improving one interface can damage another.

**Model.** Its 8B backbone builds on ACE-Brain-0 and Qwen3-VL. The architecture shares visual-language context across perception, planning and monitoring; manipulation uses a flow-matching Action Expert, with a fast DINOv3 visual path for fresh observations. The [model card](https://huggingface.co/ACERobotics/ACE-Brain-0.5-8B) documents the released checkpoint and supported interfaces.

{{< robo-figure id="21" caption="ACE-Brain-0.5, Fig. 2. Task data, shared backbone, action expert and the SSR+ stages." source="https://arxiv.org/abs/2607.04426" >}}

**Training: SSR+.** The four stages separate learning a task from making the task interfaces coexist:

1. **Scaffold:** retain the spatial knowledge supplied by ACE-Brain-0.
2. **Specialize:** train separate QA/planning, grounding, navigation and progress checkpoints.
3. **Reconcile:** merge their weight updates together with the spatial scaffold.
4. **Reactivate:** run a short mixed SFT stage to restore consistent output conventions.

The layer-level merging objective matches a merged layer to each expert on that expert's inputs. Writing its weights as `W`:

{{< robo-math >}}
W_l^*=\arg\min_{W_l}\sum_i\mathbb E_{x\sim\mathcal D_{i,l}}
\left\|W_{i,l}x-W_lx\right\|_2^2.
{{< /robo-math >}}

The report implements an approximate, data-free task-vector optimization through FusionBench, with 1,000 Adam iterations. Reactivation follows the merge. The authors interpret the fast recovery as retained task knowledge whose output formats need recalibration. It is useful to distinguish that interpretation from a guarantee that merging preserves every capability. [SSR+, §3.2](https://arxiv.org/html/2607.04426v1#S3.SS2).

**Data by interface.** Public data does much of the work:

| Interface | Supervision |
| --- | --- |
| Spatial reasoning / grounding | VSI-590K, VLM-3R, GPT4Scene, MindCube; RefSpatial, PixMo-Points and RoboPoint. Additional Spatial Harness annotations use geometric checks |
| Navigation | R2R, RxR, EnvDrop, ScaleVLN and SRDF-derived instructions, converted into observation/history → next-action examples |
| Manipulation | Robot trajectories standardized into state/action channels, with masks for unavailable channels |
| Progress | RBM-1M converted into 505K multi-image examples: eight-frame progress sequences and same-task trajectory preferences |

Progress values are represented as integers from 0 to 1000 during training, then rescaled for evaluation. This uses the same numerical range as pointing, but the quantity is task completion, not an image coordinate. The [official release](https://github.com/DAXIAORobotics/ACE-Brain-0.5) provides the model entry points; the data construction is described in [§5.1](https://arxiv.org/html/2607.04426v1#S5.SS1).

**What “self-improvement” means here.** The report describes two mechanisms. Deployment can update external task schemas, spatial memory and recovery cases without changing weights. Separately, its navigation experiment collects policy rollouts, lets an oracle recover from deviations, filters recovered trajectories, and retrains with those corrections. Memory updates and supervised policy improvement should not be counted as the same training step.

**The evaluation I would keep.** Reversing a successful trajectory preserves the objects and frames while changing whether the action makes progress. ACE-Brain evaluates this with refined progress benchmarks. For a reward model, that distinction is consequential: recognizing a nearly finished scene is insufficient if the robot is undoing the task. This is also a useful comparison with RoboBrain's hop-based progress representation.

### Molmo 2-ER: embodied pretraining before action learning {#molmo2-er}

Ai2 and collaborators build [Molmo 2-ER](https://arxiv.org/abs/2605.02881) on Molmo 2, then use it to initialize MolmoAct 2. The recipe is fairly direct: assemble public spatial and video datasets, train the VLM on embodied tasks, mix general data back in, and only then add action learning.

**Model and training.**

- Qwen3-4B language backbone plus SigLIP2 vision encoder. The ER checkpoint predicts text and locations; the later MolmoAct 2 checkpoints add robot actions.
- **Embodied specialization:** 20K steps on the roughly 3.3M-example ER corpus plus 8% Tulu-3 text. Sequence length 4,200, global batch size 64, on 16 H100s.
- **Joint refinement:** another 1.5K steps with Molmo 2's general multimodal data mixed back in. Sequence length rises to 16,384 for long videos and multi-image examples; per-device batch size drops from 4 to 1.
- Paper §2.2 keeps text at 8% and splits the remainder equally: **46% embodied / 46% general / 8% text**. The [model card](https://huggingface.co/allenai/Molmo2-ER#training) instead lists 50/42/8. Both describe the same two stages, but the exact mixture needs checking against the released configuration.

**How the data is made.** Most of the ER work is dataset selection and conversion. The sources cover different kinds of supervision: simulator ground truth, reconstructed real scenes, human-labeled videos and synthetic relational questions.

{{< robo-figure id="12" caption="MolmoAct 2, Table 1. The paper's reported multimodal corpus: Molmo 2 general data, Molmo 2-ER data and Tulu-3. The weights refer to the non-robotics mixture used in action pretraining." source="https://arxiv.org/abs/2605.02881" >}}

| Data | Construction and role |
| --- | --- |
| Single-image spatial QA | SAT supplies simulated motion, perspective and action-consequence questions. RefSpatial contributes 250K multiple-choice and 250K CoT examples. VST-P normalizes images to a virtual camera for metric depth, distance, direction and size; VSI-590K adds questions propagated from 3D labels. |
| Pointing and detection | 700K RoboPoint procedural examples specify objects or free space with normalized `(x, y)` targets. Add 80K RefSpatial pointing examples and 100K LVIS-derived detection examples. |
| Video QA | SIMS-VSI contributes 203K simulator-labeled questions; RoboVQA contributes a 200K subset covering planning, affordances and prediction; VSI-590K supplies 300K video examples. |
| Multi-image / ego–exo | A 500K SenseNova-SI subset and 200K cross-view VST-P examples train correspondence across cameras and first-/third-person views. |
| Abstract relations | 50K CLEVR and 100K GRiD-3D examples. The latter separates an object's own front/left from the camera's front/left. |

These are the paper's reported source subsets. Table 1's individual category counts do not reconcile exactly with its stated 3.3M subtotal, so I would not treat them as an independently verified partition.

**Connecting the backbone to actions.** MolmoAct 2 first learns discrete action tokens with a FAST-style tokenizer. Post-training then attaches a flow-matching action expert. At each layer, learned projections map the VLM's keys and values into the expert's cross-attention space. Discrete next-token supervision and continuous action supervision remain in the training objective.

{{< robo-figure id="13" caption="MolmoAct 2, Fig. 4. The action expert receives per-layer KV context from the pretrained VLA; discrete action prediction and flow matching are trained together." source="https://arxiv.org/abs/2605.02881" >}}

The robot mixture includes 720 hours of bimanual YAM demonstrations, filtered DROID and SO-100/101 data. DROID filtering removes idle segments and keeps successful episodes with language annotations. SO-100/101 filtering checks schema validity, removes evaluation-style data, checks eligibility, and applies a trajectory-quality gate. MolmoAct2-Think further predicts depth tokens, reusing unchanged scene regions between steps.

**Results and ablations.**

- ER averages **63.8% over 13 embodied-reasoning benchmarks**, 17 percentage points above Molmo 2 in the reported evaluation.
- With the same discrete action setup and 60K training steps, replacing Molmo 2 with ER raises **LIBERO Long from 77.6% to 83.6%**, before introducing a continuous action expert.
- The connection ablation gives **94.0%** for per-layer hidden states, **94.8%** for per-head KV and **95.9%** for projected per-layer KV, averaged over the four LIBERO suites. These runs start from MolmoAct2-Pretrain.
- The final benchmark-finetuned policies reach **97.2%** for MolmoAct 2 and **98.1%** for Think on LIBERO. These include later action training and should not be attributed to the ER backbone alone.

**Weights and deployment.** [Molmo 2-ER](https://huggingface.co/allenai/Molmo2-ER) is Apache-2.0. The [repository](https://github.com/allenai/molmoact2) links ER, discrete-action pretraining, continuous-action and Think checkpoints, training code and datasets. Select an action checkpoint adapted to the target embodiment for robot execution.

**My take.** Two parts are easy to reuse: organize public datasets by the spatial ability they supply, and test the specialized backbone under a fixed action-learning setup. The general-data stage is short enough to be a separate knob, rather than requiring the whole multimodal pretraining run to be repeated.

### Gemini Robotics-ER: grounding, monitoring and high-level planning {#gemini-robotics-er}

Google DeepMind's [Gemini Robotics 1.5 report](https://arxiv.org/abs/2510.03342) describes two models: Robotics-ER handles embodied reasoning and tool use; Robotics is the VLA that produces actions. Together they form a high-level planner and low-level executor. The report is especially concrete about the outputs and closed-loop evaluation.

**Model and training.**

- ER takes images or video with a request and returns grounded answers, points, trajectories, state estimates or tool calls. The VLA additionally consumes robot state and outputs continuous actions.
- ER can spend a variable number of thinking tokens before answering. In the 1.5 experiments, image/video QA benefits more from a larger thinking budget than pointing does.
- The report does not publish ER's full data counts, mixture or training schedule. Its robot-demonstration and motion-transfer sections describe the VLA. The later [ER 1.6 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-1-6/) names Gemini 3.0 Flash as the base, with additional embodied training, but still gives no reproducible data mixture.

{{< robo-figure id="14" caption="Gemini Robotics 1.5, Fig. 1. ER supplies grounded reasoning and tool calls; the VLA combines observations, instructions and proprioception to produce actions." source="https://arxiv.org/abs/2510.03342" >}}

**Task interfaces.** The following are demonstrated or evaluated capabilities, rather than a disclosed training-data inventory.

| Task | Input and output |
| --- | --- |
| Spatial pointing | Image + relational request → target points, including free-space locations such as the space left of a cup |
| Steerable pointing | Image + a requested correction → adjusted point, e.g. move the target slightly upward |
| Point-to-count | Identify instances with points, then return the count |
| Trajectories | A sequence of points expressing a path subject to spatial or physical constraints |
| Progress / success | Task description + single- or multi-view observations → completion estimate or success decision |
| Temporal ordering | Shuffled task frames → estimated completion percentages that recover their order |
| High-level planning | Task, observations and history → next instruction or tool call; check execution before continuing |

**Pointing results.** Figure 10 aggregates Point-Bench, RefSpatial, RoboSpatial, Where2Place and PixMo Count. A predicted point is correct when it falls inside the target mask; counting uses the final count.

{{< robo-figure id="15" caption="Gemini Robotics 1.5, Fig. 10. Pointing evaluation by task type. ER 1.5 scores 52.6% on the aggregate, 46.2% on spatial pointing, 67.8% on steerable pointing and 80.0% on point-to-count." source="https://arxiv.org/abs/2510.03342" >}}

The aggregate rises from 49.1% for the earlier ER to 52.6% for ER 1.5. On spatial pointing, ER 1.5 scores 46.2%, versus 26.3% for Gemini 2.5 Pro and 26.5% for GPT-5 in this evaluation. The separate RynnBrain 1.1 comparison cited earlier reports ER at AP@15 48.3 for **3D grounding**; that is a different task and metric.

**Success detection.** The report separates single-/multi-view input and offline/real-time evaluation. Real-time evaluation queries a recorded rollout at 5 Hz, accounts for inference latency and uses the latest returned prediction. Offline evaluation gives each prediction unlimited inference time.

{{< robo-figure id="16" caption="Gemini Robotics 1.5, Fig. 13. Success detection with and without latency, using one or multiple camera views. The model ranking changes with the evaluation setting." source="https://arxiv.org/abs/2510.03342" >}}

ER 1.5 scores 0.66 / 0.59 on real-time multi-/single-view detection, and 0.74 / 0.79 offline. GPT-5 reaches 0.80 on offline single-view detection but 0.46 in the real-time setting. This is a useful example of why inference latency belongs in a monitor's evaluation: a correct answer arriving after the robot has moved on can still be the wrong signal for control.

**Closed-loop execution.** The agent sends grounded subtasks to Gemini Robotics 1.5, observes the result and updates its plan. Eight tasks on ALOHA and bi-arm Franka compare the thinking VLA alone, Gemini 2.5 Flash + VLA, and ER 1.5 + the same VLA. Tasks include remembering and restoring an arrangement, web-assisted sorting and a nine-step drawer task. Franka receives additional post-training for long-horizon execution. Figure 17 shows task-dependent gains: for example, “Desk Organization” progress rises from 0.56 with Flash to 0.88 with ER; “Nut Allergy” remains 0.78 for both.

**Availability and later versions.** ER is a closed model rather than an open-weight training recipe. The [ER 2 release](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/) extends streaming interaction and tool use through Gemini Live API, and evaluates progress in five completion bins. The figures above are specifically ER 1.5; those results and the later interfaces should retain their version labels.

**My take.** The most useful part is the evaluation design: check whether the point is usable, whether the monitor responds in time, and whether replacing the high-level model helps while the executor stays fixed. These tests are also applicable to the open models above.

## Data construction {#data}

This section covers general multimodal data to mix into embodied training, followed by spatial questions from images, consistent geometry across views, and temporal questions from execution videos.

### LLaVA-OneVision: general multimodal data alongside embodied tasks {#llava-onevision-data}

LLaVA-OneVision's released training mixtures provide general VQA, OCR, document understanding and instruction-following examples. These give embodied training a source of broader visual-language supervision. The useful item here is the **data mixture**; using it does not require adopting the LLaVA model.

**Which data.** The [original OneVision release](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md) includes 3.2M single-image instruction samples and a later 1.6M mixture of single-image, multi-image and video examples. The [OneVision-1.5 report](https://arxiv.org/abs/2509.23661) releases a larger pair: **85M image–text examples for mid-training** and **22M instruction examples**. These are separate releases and stages; “LLaVA-OV data” alone does not identify which pool was used.

{{< robo-figure id="22" caption="LLaVA-OneVision-1.5, Fig. 3. Concept balancing on the left; mid-training sources and instruction-task distribution in the two pie charts." source="https://arxiv.org/abs/2509.23661" >}}

**How the 1.5 data is made.**

- **Select images by concept.** Combine sources such as COYO, Obelics, DataComp, ImageNet and SAM. MetaCLIP embeds images and a 500K-concept vocabulary; assign nearby concepts and sample with inverse-frequency weights, reducing the dominance of common concepts.
- **Generate captions.** A captioner writes English and Chinese descriptions; validity filtering removes duplicate and excessively long outputs. The reported 85M pool contains 65M English and 20M Chinese pairs.
- **Assemble instruction tasks.** Curate existing datasets into the 22M instruction pool, covering captions, general VQA, OCR, charts/tables, grounding/counting, science, code/math and domain-specific tasks.

**How embodied papers use it.** [OpenSpatial](https://arxiv.org/html/2604.07296v1#S4) mixes general OneVision data and spatial-reasoning data at **1:1** to retain general multimodal ability. [RoboBrain 2.5](https://arxiv.org/html/2601.14352v1#S3.SS1) uses **OneVision-1.5-Instruct**, removes text-only samples, balances visual subclasses, and deduplicates it with Honey-Data-1M by image and QA similarity. The resulting **2.83M** is the combined general-data subset, not the size of OneVision itself.

The reusable part is the task coverage and filtering recipe. The general-data ratio is still a training choice: OpenSpatial's 1:1 is a documented example, not a proportion shared by every model in this article.

### SpatialVLM: image geometry to spatial QA {#spatialvlm}

[SpatialVLM](https://arxiv.org/abs/2401.12168) builds **2B spatial QA pairs from 10M real-world images**, split equally between qualitative relations and quantitative measurements. The pipeline uses pretrained vision models and templates.

{{< robo-figure id="18" caption="SpatialVLM, Fig. 2. Filter images, extract object descriptions and masks, lift them to 3D, resolve ambiguous references, then generate spatial questions." source="https://arxiv.org/abs/2401.12168" >}}

**How the data is made.**

1. CLIP filters out unsuitable images, such as isolated product shots and screenshots.
2. Region proposals, FlexCap descriptions and segmentation produce object-level labels. Fine-grained descriptions distinguish “cake shaped like a house” from another cake.
3. Metric depth lifts pixels to 3D. Horizontal surfaces, such as a floor or tabletop, provide a reference for coordinate alignment.
4. Caption similarity identifies ambiguous references; these are refined or rejected. Object point clouds and boxes supply distances, sizes and relations.
5. Templates generate **38 question types**, with roughly 20 question and 10 answer templates per type. Answers include numbers and units with human-like rounding.

**Training and use.** The model follows PaLM-E's architecture with a PaLM 2-S language backbone. Spatial tasks take **5% of training tokens** in the mixed dataset. The work studies visual-encoder fine-tuning and uses predicted distances for reward generation. This is an early, concrete version of the pseudo-3D scene-graph recipe seen in later models; its metric labels inherit the errors of depth estimation.

### OpenSpatial: one scene representation across views {#openspatial}

[OpenSpatial](https://arxiv.org/abs/2604.07296) generates **3M examples across five task groups and 19 subtasks**. Its intermediate representation is an oriented 3D box in a shared, Z-up world frame: center, side lengths and rotation. The same object retains its identity as the camera moves.

{{< robo-figure id="17" caption="OpenSpatial, Fig. 2. Scene-level 3D boxes are projected into frames, checked for visibility and indexed; single-view and multi-view QA share these annotations." source="https://arxiv.org/abs/2604.07296" >}}

**How the data is made.**

- Obtain scene boxes through human annotation or automatic lifting. The automatic route uses Gemini object recognition and SAM masks, associates instances in 3D, then fits oriented boxes.
- Project boxes into each camera. Reject out-of-frustum objects; back-project depth pixels and check their occupancy inside the 3D box to filter occluded instances.
- Refine the surviving masks with SAM. Store masks, 2D/3D boxes, partial point clouds, object tags and a **metric-scale flag** in an object–frame index.
- Generate single-view questions from scene graphs and multi-view questions from overlapping views. Skip measurement questions when the metric flag is false.

**Coverage and release.** The five groups cover measurement, spatial relations, camera perception, multi-view consistency and scene reasoning. The [data engine](https://github.com/VINHYU/OpenSpatial) is open. Compared with generating each view's QA independently, this design gives cross-view questions a shared geometric reference and an explicit check on whether metric answers are valid.

### RoboVQA: execution videos to temporal QA {#robovqa}

[RoboVQA](https://arxiv.org/abs/2311.00899) collects **238 hours of video** across three office buildings, using a robot, a human arm and a human with a grasping tool. It contains 5,246 long-horizon episodes and 92,948 medium-horizon segments; the [release](https://github.com/google-deepmind/robovqa) lists 829,502 video–text pairs and 29,520 unique instructions.

{{< robo-figure id="19" caption="RoboVQA, Fig. 1. Collect a long task continuously, annotate its segments afterwards, and derive ten QA types from the segment labels and their temporal order." source="https://arxiv.org/abs/2311.00899" >}}

**How the data is made.** Users and operators propose long tasks, such as making coffee. Operators execute them continuously; annotators then mark segment boundaries and describe each subtask. The known order generates ten QA types: next-step planning, planning with history, remaining steps, future prediction, past description, success/failure, and generative or positive/negative discriminative affordance questions. For example, a segment boundary supplies before/after observations for “Is this task complete?” without separately annotating every question.

**Training and results.** RoboVQA-VideoCoCa fine-tunes a 383M video–text model on these pairs. The paper compares one-frame and 16-frame inputs and finds substantial gains from video; mixing human and robot data also helps robot-only evaluation. Continuous collection reduces resets and gives 2.2× the robot-collection throughput of the step-by-step baseline. Hindsight segmentation still costs human annotation time.

The evaluation separately covers recorded video, live teleoperation and autonomous execution. This distinction matters when interpreting reduced intervention: fewer corrections to planning can coexist with a human still controlling every physical action.

## The closed-model frontier and related work {#related}

Direct published comparisons with GPT-6 Astra remain limited in this set of notes. GroundingPI claims an advantage on grounding. Other papers use Gemini Robotics-ER or GPT-5.x as references, or evaluate Astra as one agent in a benchmark.

RobotWorld and PLAN-Bench are evaluation references. Vlaser remains a shorter pointer. The MolmoAct 2 row records a separate, externally reported execution test.

| Work | What it does | Data or evaluation detail |
| --- | --- | --- |
| [RobotWorld](https://arxiv.org/abs/2610.10409), October 2026 | Multimodal agents control different robots through tools. | Astra completes 16/84 tasks (19.0%), Opus 5.5 completes 13, and Kimi K3 completes two. Astra does better on spatial and constrained-contact tasks, but closed-loop execution remains difficult. |
| [PLAN-Bench](https://arxiv.org/abs/2610.18836), October 2026 | 699 problems on spatial cognition, temporal reasoning and multistep planning. | Of 28 VLMs, Qwen3.8-Max leads at 75.0%; Kimi K3 scores 70.2% and GPT-5.6-sol 65.8%. Astra is not evaluated in the paper. |
| [Vlaser](https://arxiv.org/abs/2510.11027), ICLR 2026 | A model bridging VLMs and VLAs. | An embodied-reasoning data engine with 6M examples. |
| MolmoAct2, Ai2 | An open action-reasoning model. | A secondary report by the-decoder gives Astra 7/100 completions and median progress 46 on StationeryBench bimanual tabletop tasks, versus 0/100 and median progress 12 for MolmoAct2. |

The Astra training explanation sometimes repeated in coverage—that its 3D gains may come from large quantities of 3D data, such as Blender scenes—is speculation attributed to researchers. OpenAI has not published the recipe in the sources used for these notes. See the [the-decoder report](https://the-decoder.com/gpt-6-astra-appears-to-show-a-step-change-in-spatial-reasoning-based-on-early-benchmarks/) for that discussion and the secondary StationeryBench comparison.

## Comparing the data recipes {#comparison}

The recurring pattern is public data converted to a shared format, custom automatic annotation, stronger-model verification, and selective human work. Each team allocates that effort differently.

| | HY-Embodied | RynnBrain | RoboBrain 2.5 | GroundingPI |
| --- | --- | --- | --- | --- |
| Backbone | Hunyuan / Hy3, MoT-2B and A3B | Qwen3-VL / Qwen3.5 | Qwen3-VL-8B | MoonViT-V2 + Qwen3-4B |
| Reported scale | About 625B pretraining tokens; 236B embodied/perceptual | 19.89M examples in 1.0 | 12.4M examples across 8.3M + 4.1M stages | About 500B + 221B tokens |
| Main emphasis | Early perception and iterative reasoning improvement | Video localization and grounded plans | Metric 3D traces and dense progress | Precise grounding as an interface for action |
| Coordinates | 0–1000 image coordinates | 0–1000 image coordinates plus frame indices | Normalized image coordinates; `(u, v, d)` for 3D | 1,000 dedicated coordinate tokens |
| Custom data | 62M detection; 36M depth; tracked trajectories; RGB-D spatial QA; manually labeled social interaction | First-person video → SLAM QA; OCR; human points and paths; grounded planning; simulated grasp projections | RefSpatial / TraceSpatial pseudo-3D scene graphs; 35M generated Dopamine-Reward progress examples | Multi-teacher engine; multiple tasks derived from each instance |
| Robot sources | Open-X, DROID and related videos | AgiBot, Open-X, Galaxea-G0 and collected video | AgiBot, DROID, RoboTwin, LIBERO, EgoDex | No robot data in pretraining |
| Human work | CoT collaboration and social interaction | Filtering, points, trajectories, planning checks, grounded CoP chains | Progress keyframes and ShareRobot planning | Little manual labeling in the described engine |
| RL examples | Dynamically selected partial successes; 50K per round | SFT scores of 40–80; 30K | No separate RL stage described here | GRPO with completeness and localization rewards |
| Training-data release | No | Not clearly specified | Partial; RefSpatial / TraceSpatial released separately | Not clearly specified |

The additional models answer a few more specific questions:

| Model | What its recipe adds to this comparison |
| --- | --- |
| Embodied-R1.5 | Failure correction data and balanced reinforcement fine-tuning |
| ACE-Brain-0.5 | Task specialization, parameter merging and a short mixed stage to restore output conventions |
| Molmo 2-ER | Embodied specialization followed by general-data rehearsal; a controlled action-transfer ablation |
| Gemini Robotics-ER | Orchestration and monitoring evaluations; insufficient public detail to reconstruct its training mixture |

OneVision supplies general multimodal supervision. SpatialVLM, OpenSpatial and RoboVQA add estimated image geometry, consistent scene geometry across views, and execution videos with temporal questions, respectively.

There are six steps I would extract from these pipelines. They are choices to combine, rather than a claim that every model uses every stage.

1. **Define the abilities.** Write down perception, space, affordance/trajectory, planning and recovery tasks before choosing mixtures and benchmarks. HY 1.0 and ACE-Brain organize their data this way.
2. **Standardize the outputs.** Share a coordinate interface for image boxes, points and paths. Add frame indices for video; retain depth or camera-frame box parameters where a task needs 3D.
3. **Start with public data.** General multimodal data, such as the OneVision mixtures above, helps retain general abilities. OpenSpatial uses a 1:1 general/spatial mixture; the proportion should be checked for each recipe. Convert embodied sources such as RoboPoint, RefSpatial, RoboAfford, ShareRobot and VSI-590K to the common format.
4. **Generate questions from geometry.** Posed RGB-D, point clouds, ScanNet, CA-1M and SLAM reconstructions give boxes and scene graphs from which templates can produce size, distance and orientation QA. Remove ambiguous cases.
5. **Mine robot and first-person videos.** Track end effectors for trajectories, split subtasks for plans, label keyframes for progress, and collect failures for recovery questions. Validate and filter the generated annotations.
6. **Add reasoning and RL where appropriate.** Generate and check CoT for cold-start SFT, then select verifiable prompts of intermediate difficulty. Match rewards to output structure: IoU, L2, DTW or Fréchet. HY further uses the RL policy to generate long CoT and restarts supervised training from a clean checkpoint.

## References {#references}

These notes were assembled on October 17, 2026 and expanded on October 18. Model availability and benchmark statements refer to the cited reports and the releases recorded then. Paper figures are reproduced from the corresponding PDFs; captions identify their figure or table numbers.

- **HY-Embodied:** [0.5 report](https://arxiv.org/abs/2604.07430) · [VLM-1.0 report](https://arxiv.org/abs/2607.12894) · [GitHub](https://github.com/Tencent-Hunyuan/HY-Embodied) · [0.5 weights](https://huggingface.co/tencent/HY-Embodied-0.5) · [1.0 weights](https://huggingface.co/tencent/Hy-Embodied-VLM-1.0).
- **RynnBrain:** [1.0 report](https://arxiv.org/abs/2602.14979) · [1.1 report](https://arxiv.org/abs/2607.17977) · [GitHub](https://github.com/alibaba-damo-academy/RynnBrain) · [1.1-9B weights](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-9B).
- **RoboBrain:** [2.5 report](https://arxiv.org/abs/2601.14352) · [2.0 report](https://arxiv.org/abs/2507.02029) · [2.5-8B weights](https://huggingface.co/BAAI/RoboBrain2.5-8B-NV).
- **GroundingPI:** [report](https://arxiv.org/abs/2609.39601) · [weights](https://huggingface.co/GroundingPI/GroundingPI).
- **Related work:** [ACE-Brain-0.5](https://arxiv.org/abs/2607.04426) · [OpenSpatial](https://arxiv.org/abs/2604.07296) · [RobotWorld](https://arxiv.org/abs/2610.10409) · [PLAN-Bench](https://arxiv.org/abs/2610.18836) · [Vlaser](https://arxiv.org/abs/2510.11027) · [Astra coverage, the-decoder](https://the-decoder.com/gpt-6-astra-appears-to-show-a-step-change-in-spatial-reasoning-based-on-early-benchmarks/).

- **Embodied-R1.5:** [report](https://arxiv.org/abs/2606.11324) · [code](https://github.com/pickxiguapi/Embodied-R1.5) · [weights](https://huggingface.co/IffYuan/Embodied-R1.5) · [SFT data](https://huggingface.co/datasets/IffYuan/Embodied-R1.5-SFT-Dataset) · [RFT data](https://huggingface.co/datasets/IffYuan/Embodied-R1.5-RFT-Dataset).
- **ACE-Brain-0.5:** [report](https://arxiv.org/abs/2607.04426) · [code](https://github.com/DAXIAORobotics/ACE-Brain-0.5) · [weights](https://huggingface.co/ACERobotics/ACE-Brain-0.5-8B).
- **OpenSpatial:** [report](https://arxiv.org/abs/2604.07296) · [data engine](https://github.com/VINHYU/OpenSpatial).

- **Molmo 2-ER / MolmoAct 2:** [report](https://arxiv.org/abs/2605.02881) · [ER model card](https://huggingface.co/allenai/Molmo2-ER) · [code and releases](https://github.com/allenai/molmoact2) · [Ai2 overview](https://allenai.org/blog/molmoact2).
- **Gemini Robotics-ER:** [1.5 report](https://arxiv.org/abs/2510.03342) · [1.6 model card](https://deepmind.google/models/model-cards/gemini-robotics-er-1-6/) · [ER 2 release](https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/).
- **SpatialVLM:** [paper](https://arxiv.org/abs/2401.12168) · [project](https://spatial-vlm.github.io/).
- **RoboVQA:** [paper](https://arxiv.org/abs/2311.00899) · [project](https://robovqa.github.io/) · [data and code](https://github.com/google-deepmind/robovqa).

- **LLaVA-OneVision data:** [original report](https://arxiv.org/abs/2408.03326) · [original training data](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md) · [1.5 report](https://arxiv.org/abs/2509.23661).
