---
title: "Titans: Learning to Memorize at Test Time"
date: 2025-02-03
draft: false
ShowToc: true
tags: ["machine-learning", "computer-science", "deep-learning", "transformer", "nlp", "aritificial-intelligence", "time-series", "paper-reading", "memory", "test-time-scaling"]
summary: "This article introduces Titans, a novel architecture that as a meta in-context learner, learns to memorize at test time. Through designing a long-term memory module, and proposing three variants of Titans (MAC, MAG, MAL), the model achieves superior performance compared to Transformers and other baselines, especially in long-context tasks."
---
![](../img/titans/Pasted%20image%2020250203185130.png#center)
Original Paper: **Titans: Learning to Memorize at Test Time**[^1].
## 1. Motivation
**Key question: How to make the model memorize the context?**
- Recurrent models: compress the data into a fixed-size memory (called hidden state)
- Attention: attending to the entire context window, capturing the direct dependencies of all tokens.
However, attention comes at a **quadratic cost**, therefore in pratice, the model is **limited to a fixed context length.**
### 1.1 What is Transformer (attention) good at?
The short answer: short but accurate memory.

Transformer Blocks function as associative memory blocks, where they learn to store key-value associations and retrieve them by computing pairwise similarity between queries (i.e., search signals) and keys (i.e., contexts). Accordingly, by design, the output of a Transformer is exclusively conditioned on the direct dependencies of tokens in the current context window. 

**Attention due to its limited context but accurate dependency modeling performs as a short-term memory.**
### 1.2 The Dilemma of Linear Attention Schemes
Despite efficiency and the ability to scale to longer context, linear Transformers **do not show competitive performance compared to Transformers** as the kernel trick makes the model a linear recurrent network, in which the data is compressed into a matrix-valued states.

So there emerges a dilemma:
- We introduce Linear Attention Schemes to scale Transformers to long context
- Long context cannot be properly compressed in a small vector-valued or matrix-valued states
### 1.3 Understanding Memory of NNs

> “Most existing architectures consider memory as a neural update caused by an input, and define learning as a process for acquiring effective and useful memory, given an objective.” (Behrouz et al., 2024, p. 2)

#### 1.3.1 RNNs
- Memory Module: $M \equiv$ Hidden State, **Vector-valued, Fixed-Size**
- At each timestep $t$:
	1. Update Memory: $M _  {t}\leftarrow f(M _  {t-1},x _  t)$
	2. Retrieve Output: $y _  t =g(M _  t,x _  t)$   
#### 1.3.2 Transformer
We consider the case of Causal Attention.
- Memory Module: Growing Memory of size $2\times \mathbb{R}^{N\times d}$
- At each token $t$:
	1. Update Memory: $\texttt{Append}(k _  t,v _  t)$
	2. Retrieve Output: $Y _  t\in \mathbb{R}^{N\times d}$, where the $r$-th row is:
	$$y _  {r}=\sum _  {i=1}^{r}\frac{\exp(q _  r^Tk _  i/\sqrt{d})}{\sum _  {j=1}^{r} \exp(q _  r^Tk _  j/\sqrt{d})} v _  i$$
#### 1.3.3 Linear Transformers
- Memory Module:  $M$  **Matrix-valued, Fixed-Size**
If we substitute $\exp(q^T k/\sqrt{d})$ in Transformers as a kernel function $\phi(\cdot,\cdot)$ , s.t. $\phi(x,y)=\phi(x)\cdot \phi(y)$, we have:
$$y _  {r}=\sum _  {i=1}^{r}\frac{\phi(q _  r, k _  i)}{\sum _  {j=1}^{r} \phi(q _  r, k _  j)} v _  i=\sum _  {i=1}^{r}\frac{\phi(q _  r)^T\phi(k _  i)}{\sum _  {j=1}^{r} \phi(q _  r)^T \phi(k _  j)} v _  i=\frac{\phi(q _  r)^T \sum _  {i=1}^r \phi(k _  i)v _  i}{\phi(q _  r)^T \sum _  {i=1}^r \phi(k _  i)}$$
Then because $\sum _  {i=1}^r \phi(k _  i)v _  i, \sum _  {i=1}^r \phi(k _  i)$ can be re-used in each step, the computation efficiency is much better. An concrete example: $\phi \equiv Id$:
$$M _  t=M _  {t-1}+k _  t^Tv _  t$$
$$y _  t=q _  t M _  t$$
In this perspective, linear Transformers's memory update is equivalent to additively compress and write keys and values,$(k _  t, v _  t)$, into a matrix-valued memory unit $M _  t$.

So, the key questions are:

- (Q1) What constitute a good structure for the memory? 
- (Q2) What is a proper memory update mechanism? 
- (Q3) What is a good memory retrieval process?
## 2. Long Term Memory
We need an online meta-model that **learns how to memorize/forget the data at test time.** In this setup, the model is learning a function that is capable of memorization, but it is not overfitting to the training data, resulting in a better generalization at test time.
### 2.1 Learning Process and Surprise Metric
Intuitively, we treat the training as a online learning problem:
$$M _  t\leftarrow M _  {t-1}-\theta _  t \underbrace{\nabla l(M _  {t-1};x _  t)} _  {\text{suprise}}$$
However, if one is surprised at some moment, that person is less likely to be surprised for the upcoming moments. This, reflected in the model is that the gradient can be extremely small after some suprising steps, and get stuck at local minima. 
> From the human memory perspective, an event might not consistently surprise us through a long-period of time although it is memorable. The reason is that the initial moment is surprising enough to get our attention through a long time frame, leading to memorizing the entire time frame.

Building upon this idea, we break the surprise metric to: 1) past suprise  and 2) momentary suprise.
$$M _  t\leftarrow M _  {t-1}+S _  t$$
$$S _  t\leftarrow \eta _  t \underbrace{S _  {t-1}} _  {\text{Past Suprise}}-\theta _  t \underbrace{\nabla l(M _  {t-1};x _  t)} _  {\text{Momentary Suprise}}$$
This formulation is similar to gradient descent with momentum. $S _  t$, the momentum, can be viewed as a memory of surprise across the sequence.
### 2.2 Objective
What is this $l(\cdot,\cdot)$?

Here, we focus on *associative memory*, and the loss function is defined by:
$$l(M _  {t-1};x _  t)=||M _  {t-1}(k _  t)-v _  t|| _  2^2$$
Note that this is a meta-learning process, where in the inner loop, $M$ is updated fixing other parameters (e.g. $W _  k,W _  v$), and in the outer loop, other parameters are learnt. In this way, the model learns how to memorize the mapping between keys and values at test time.

### 2.3 Forgetting Mechanism. 
When dealing with very large sequences (e.g., millions of tokens), it is crucial to manage which past information should be forgotten–even with a deep or a very large matrix-valued memory. Therefore, for the updating rules, we add a gating mechanism:
$$M _  t\leftarrow (1-\alpha _  t)M _  {t-1}+S _  t\quad \alpha _  t \in[0,1]$$
$$S _  t\leftarrow\eta _  t \underbrace{S _  {t-1}} _  {\text{Past Suprise}}-\theta _  t \underbrace{\nabla l(M _  {t-1};x _  t)} _  {\text{Momentary Suprise}}$$
### 2.4 Model Architecture
In this work, the long term memory module $M$ is implemented using MLPs with $L _  M$ layers.
### 2.5 Retrieval
We simply do $$y _  t=M^* (q _  t)$$
(You can try to convince yourself that this makes sense.)
### 2.6 How to Parallelize the Long-term Memory Training
Without parallelizing, the training of long-term memory module requires $O(N)$ FLOPs. This is similar to TTT's method. If you are interested, check out Section 3.2 of the original paper.
![](../img/titans/Pasted%20image%2020250203215455.png#center)
![](../img/titans/Pasted%20image%2020250203213346.png#center)
### 2.7 Persistent Memory
The Long Term Memory can be seen as **Contextual Memory**, as it is solely dependent on context. Therefore, in addition to the long-term memory, we also use a set of **learnable but input-independent parameters to act as task-related memory** (also referred to as meta memory in literature). Specifically, the input $x$ is modified to be :
$$x _  {\text{new}}=[p _  1,...,p _  {N _  p}]||x$$
where $||$ is concatenation, the $p$ s are learnable parameters.
## 3. Titans
After designing the long-term neural memory, an important remaining question is how to effectively and efficiently incorporate memory into a deep learning architecture. 

The essential parts of Titans include:
1. **Core**: this module consists of the short-term memory, and is responsible for the main flow of processing the data (we use attention with limited window size); 
2. **Long-term Memory**: this branch is our neural long-term memory module that is responsible to store/remember long past; 
3. **Persistent Memory**: this is a set of learnable but date-independent parameters that encodes the knowledge about a task. 

As a proof of concept, we present three variants of Titans, in which we incorporate memory as: (i) a context, (ii) a layer, and (iii) a gated branch.
### 3.1. MAC: Memory as a Context
![](../img/titans/Pasted%20image%2020250203220324.png#center)
First, we segment the input $x$ into fixed-size segments: $S^{(i)}$ for $i=1,2,...,N/C$. Then we retrieve the long-term memory:
$$h _  t=M _  {t-1}^* (q _  t)$$
where $q _  t=S^{(t)} W _  Q$ . Then we concat the input to:
$$\tilde{S}^{(t)}=[p _  1,...,p _  N]\ ||\ h _  t\ ||\ S^{(t)}$$
and apply attention:
$$y _  t=\texttt{Attn}(\tilde{S}^{(t)})$$
Then we update the long-term memory and get the final output:
$$M _  t=M _  {t-1}(y _  t)$$
$$o _  t=y _  t \otimes M _  t^*(y _  t)$$
### 3.2 MAG: Memory as Gating
This variant uses a sliding window attention (SWA).
![](../img/titans/Pasted%20image%2020250203223217.png#center)
$$\tilde{x}=[p _  1,...,p _  {N _  p}] || x$$
$$y=\texttt{SW-Attn}^\*(\tilde{x})$$
$$o=y\otimes M(\tilde{x})$$
Here $\texttt{SW-Attn}^*$ denotes sliding window attention with prefix, and $\otimes$ is a non-linear gating. In practice, it is set as normalizing the outputs $y$ and $M(\tilde{x})$ using learnable vector-valued weights, followed by a non-linearity $\sigma(\cdot)$.

The attention masks for the two architectures are shown below:
![](../img/titans/image.png#center)
### 3.3 MAL: Memory As a Layer
The last variant uses the neural Memory As a Layer (MAL) of a deep neural network.
![](../img/titans/Pasted%20image%2020250203224403.png#center)
$$\tilde{x}=[p _  1,...,p _  {N _  p}]\ ||\ x$$
$$y=M(\tilde{x})$$
$$o=\texttt{SW-Attn}(y)$$

Lastly, an interesting theorem:
>Theorem: Contrary to Transformers, diagonal linear recurrent models, and DeltaNet, all of which are limited to $\texttt{TC}^0$ , Titans are capable of solving problems beyond $\texttt{TC}^0$, meaning that Titans are theoretically more expressive than Transformers and most modern linear recurrent models in state tracking tasks.

## 4. Experiments

### 4.1 Common-Sense Reasoning Benchmark
![](../img/titans/Pasted%20image%2020250203225837.png#center)
### 4.2 Needle in a Haystack
The needle-in-a-haystack (NIAH) task is designed to measure the actual effective context length of models. In this task, we evaluate the model on retrieving a piece of information (i.e., the “needle”) from long distractor texts (i.e.,the “haystack”). In this part, we use Single NIAH (S-NIAH) task from RULER benchmark (Hsieh et al. 2024) and evaluate Titans and baselines on sequences with length 2K, 4K, 8K, and 16K.
![](../img/titans/Pasted%20image%2020250203225148.png#center)
### 4.3 BABILong Benchmark
In this benchmark, the model needs to reason across facts distributed in extremely long documents.
![](../img/titans/Pasted%20image%2020250203225621.png#center)
### 4.4 The Effect of Deep Memory
![](../img/titans/Pasted%20image%2020250203230059.png#center)
![](../img/titans/Pasted%20image%2020250203230153.png#center)
### 4.5 Time Series Forecasting
Simba framework for time series forecasting, and replace its Mamba module with our neural memory.
![](../img/titans/Pasted%20image%2020250203230321.png#center)
### 4.6 DNA Modeling
![](../img/titans/Pasted%20image%2020250203230354.png#center)
### 4.7 Efficiency
![](../img/titans/Pasted%20image%2020250203230445.png#center)
### 4.8 Ablation
We consider our neural memory module as a base model and then changing one component at a time: (1) replacing deep memory with linear memory, removing (2) convolution, (3) momentum in the surprise measure, (4) weight decay (or forgot mechanism), and (5) persistent memory. The results are reported in Table 5. All components of neural memory design are positively contributing to its performance, where the greatest contribution comes from weight decay, momentum, convolution, and persistent memory, respectively.
![](../img/titans/Pasted%20image%2020250203230530.png#center)

[^1]: Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. "Titans: Learning to Memorize at Test Time." _arXiv preprint arXiv: 2501.00663_, 2024.


