---
title: "Machine Learning Series: 4.Robust Machine Learning"
date: 2024-12-29
draft: false
ShowToc: true
tags: ["machine-learning", "computer-science", "optimization", "math", "artificial-intelligence", "algorithm"]
summary: "This is the fourth article in the Machine Learning Series. It covers classic approaches to Robust Machine Learning, including Adversial Attacks, Adversial Training, Robust Features, Obfuscated Gradients and Provable Robust Certificates."
---
## 动机
鲁棒的机器学习的动机来自于对于模型而言，在输入中加入一点点微小的修改，就可以让模型的输出大不相同，这样的现象可以被别有用心者利用，比如让摄像头识别不出来人，或者引诱自动驾驶汽车认为前面有障碍物，从而干扰后续的decision making.
![](../img/ml4/image.png#center)
## Adversial Attacks
所以为了获得一个对于扰动鲁棒的模型，我们讲原模型的loss function重新定义：
$$\mathbb{E}_ {x,y}[Loss_ \theta (f_ \theta (x),y)]\Rightarrow \mathbb{E}_ {x,y}[\max_ {\delta \in \Delta}Loss_ \theta (f_ \theta (x+\delta),y)] $$
面对这样的优化目标，我们可以采取以下两种优化算法来获得最佳的$\delta$：
### Projected gradient descent
这个算法本质上就是做正常的梯度递降，然后投影到$\Delta$上
$$\delta_ {t+1}=P_ \Delta[\delta_ t+\eta\nabla_ x Loss(f(x+\delta_ t),y)]$$

![](../img/ml4/image2.png#center)

其中
$$ \Delta=\{\delta: ||\delta||_ \infty < \epsilon \}$$
$$P_ \delta=Clip(\delta,[-\epsilon,\epsilon])$$
### Fast Gradient Sign Method (FGSM)
考虑$\eta\rightarrow \infty$的情况，对于一个convex set, 我们一定会走到convex set的vertex上，所以我们只选取梯度的每个方向上的符号，就足够了。

![](../img/ml4/image3.png#center)
所以
$$\delta=\eta \cdot sign(\nabla_ x Loss(f(x),y)).$$

所以，也可以把PGD理解成多次进行FGSM，这样会更慢但是找到的local optimal更好。
### Adversial Training
鲁棒机器学习带来的一个问题就是说我们知道怎么优化$\delta$了，那怎么优化$\theta$呢？
这个本质上是一个min-max的优化问题：
$$\min_ \theta \sum_ {(x,y)\in S}\max_ {\delta \in \Delta}Loss_ \theta (f_ \theta (x+\delta),y)$$
这个max函数是没有gradient的，那么怎么解决呢？

这里我们要引入一个数学定理:

> *Danskin's Theorem:*
> $$\frac{\partial}{\partial \theta}\max_ {\delta \in \Delta}Loss_ \theta (f_ \theta (x+\delta),y)=\frac{\partial}{\partial \theta}Loss_ \theta (f_ \theta (x+\delta^\*),y)$$
> 其中$\delta^\*=\arg\max_ {\delta \in \Delta}Loss_ \theta (f_ \theta (x+\delta),y)$

这个看似显然的定理的证明其实不是很容易。
于是乎我们有如下算法：

> Repeat the following:
> 1. Select minibatch **B**
> 2. For each $(x, y) \in B$, compute adversarial example $\delta^*(x)$
> 3. Update parameters:$\theta_ {t+1} = \theta_ t - \frac{\eta}{|B|} \sum_ {(x, y) \in B} \frac{\partial}{\partial \theta_ t} \text{Loss}(f_ {\theta_ t}(x + \delta^*(x)), y)$

在测试模型的时候，可以随机选取数据点，进行PGD的adversial attack，来测试模型预测的准确性。

值得注意的是，鲁棒的模型并不具备通用的鲁棒性，也就是对于L2-norm鲁棒的模型不一定对于L1或者L$\infty$的鲁棒性。

这样的鲁棒模型也可能带来意想不到的好处，比如这样的adversial training模式可能可以迫使模型学习到一些在语义上有价值的信息：

![](../img/ml4/image4.png#center)

## Robust Features

下面介绍什么是鲁棒的特征，什么是非鲁棒(Non-Robust)的特征。

鲁棒的特征就是在扰动下不变的特征，而非鲁棒的特征是在扰动下改变的特征。

### 非鲁棒特征足以进行分类
这是一个有趣的研究。对于$(x,y)$，我们进行PGD攻击得到的样本为$(x',y')$,这样的$x'$中蕴含和$x$相同的鲁棒特征，但不蕴含相同的非鲁棒特征。然后我们完全从$(x',y')$构成的数据集中训练模型，发现得到的模型在正常的数据点上的效果也不错。

![](../img/ml4/image5.png#center)

### 如何生成鲁棒的数据集
对于一个鲁棒的模型$M$，将他的特征提取函数记为$g$,对于每一个训练的输入$x$，我们通过随机初始化$x_ r$并用梯度递降实现$g(x)=g(x_ r)$, 这样产生的数据集的鲁棒特征是一致的，但是非鲁棒特征是完全不同的。对于这样的$\{x_ r\}$构建的数据集上正常训练的模型的效果也很好。

也就是说,(Default) Dataset+Robust Machine Learning与Robust Feature Dataset + (Default) Machine Learning，都是可行的获得鲁棒模型的方案。

但是你仔细想一下不对劲，你在造Robust Feature Dataset的时候已经有一个鲁棒模型$M$了，所以这个研究的真正意义在于在小数据集上通过这种方式获得一个鲁棒的数据集，然后用正常的训练，得到一个鲁棒的模型。也就是小数据集上，plain machine learning在$\{x\}$上不能学到一个鲁棒的模型但在$\{x_ r\}$上可以。直觉上这个很对，因为$\{x_ r\}$上没有任何不是噪音的非鲁棒信息，所以这迫使模型学习到鲁棒特征。

![](../img/ml4/image6.png#center)

## Obfuscated Gradients
前面的PGD，需要知道导数信息，那自然而然想到的**防守**方法，就是把导数信息给藏起来。

- Shattered Gradient
  - Non-differentiable
  - Numeric instability
  - Gradient nonexistent/incorrect
  
我们的模型可以不可导，或者求出来是非法值,这样PGD就没法攻击了。
- Exploding and vanishing gradients
  - Multiple iterations of neural networks
  - Very deep network, long chain rule, so gradients explode/vanish

导数太难求了，这样也没法PGD攻击了。
- Stochastic Gradient
随机化的分类器PGD也没法攻击。

但是呢，对应这样的防守策略，也会有对应的进攻策略来crack这些防守。
- Backward pass differentiable approximation：

比如说守方用$f(g(x))$来保护模型，其中$g(x)$比如JPEG压缩是一个不可导的操作，进攻方可以通过:

$$\nabla_ x f(g(x))|_ {x=x_ 0}=\nabla_ x f(x)|_ {x=g(x_ 0)}$$

来计算梯度。

- Attack randomized classifier

进攻方可以通过多次采样，然后对于梯度求期望来获得估计值。

## Provable Robust Certificates
我们从上面的argument能够发现，这样的工作变成了一种“矛盾游戏”，攻守双方不断的crack对方的方法。没有对于模型鲁棒性的理论保障。

首先，我们认为在高维空间中很容易找到一个微小扰动就能产生分类器结果改变的点(adversial point)。所以为了fix这样的非鲁棒性，我们引入一个核函数(smooth kernel)和一个点周围的单位球，用这样得到的分布在单位球上不同类的占比来决定输出(取占比最大的)。

![](../img/ml4/image7.png#center)

这样的话，微小的扰动可以反应在直方图(histogram)的概率变化上，就没有那么容易的发生改变了。

## Greedy Filling Algorithm
那么为了分析最坏情况下一个微扰($x\rightarrow x+\delta$)能够发生的直方图的变化，我们先把之前的定义数学的写出来：

$f$: base的分类器的函数，是不鲁棒的

$g$: smoothed的分类器的函数，我们认为它更鲁棒

$$g(x)=\int_ {v \in B_ r(0)}f(v)\Pr(v)dv$$
其中$\Pr(v)$来自于核函数(smooth kernel)的分布。我们想找到一个最大的$||\delta||$使得$g(x)$与$g(x+\delta)$一定是相同色的。

二分类的情况下，也就是$g(x)$对应的输出类在$g(x+\delta)>1/2$。

![](../img/ml4/image8.png#center)

那么这个Greedy Filling究竟是怎么染色的呢：

首先定义likelihood: $$LL(y)=\frac{\Pr(y-x)}{\Pr(y-x-\delta)}$$
然后按照$LL(y)$从大到小依次排序，比如说我们知道$x$点的直方图是$\Pr(blue)=0.6,\Pr(red)=0.4$,我们就先把$LL(y)$最大的区域填蓝色，直到蓝色的部分填够了，剩下的部分就填红色。最后我们check一下是不是$g(x+\delta)>1/2$了。

值得注意的是，对于Gaussian核函数，等likelihood线是一个直线。
![](../img/ml4/image9.png#center)
证明也很简单，如下：

假设高斯分布为：

$$
p(z; \mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{\|z - \mu\|^2}{2\sigma^2}\right)
$$

因此：

$$
\frac{p(z; x, \sigma)}{p(z; x + \delta, \sigma)} = \exp\left(-\frac{\|z - x\|^2}{2\sigma^2} + \frac{\|z - x - \delta\|^2}{2\sigma^2}\right)
$$

对于具有相同likelihood的两个 $z$ ，我们有：

$$
\|z - x - \delta\|^2 - \|z - x\|^2 = \langle \delta, 2z - 2x - \delta \rangle = C
$$

因此，对于 $z_ 1, z_ 2$，我们有：

$$
\langle \delta, z_ 1 - z_ 2 \rangle = 0
$$

因此，所有具有相同likelihood的点都落在同一条线上。 ■
