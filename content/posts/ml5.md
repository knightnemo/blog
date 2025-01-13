---
title: "Machine Learning Series: 5.Hyperparameter Selection"
date: 2025-01-01
draft: false
ShowToc: true
tags: ["machine-learning", "computer-science", "optimization", "math", "artificial-intelligence", "algorithm", "random-process", "multi-arm-bandit"]
summary: "This is the fifth article in the Machine Learning Series. It covers classic approaches to Hyperparameter Selection, including Bayesian Optimization, Gradient Optimization, Random Search, Multi-Arm Bandits and Neural Architecture Search."
---
## Motivation

调超参当然是一个痛苦的事情了，那么有没有什么办法来让我们少调一调呢？答案是有的，下面将介绍一些古人的智慧。

但在开始之前，还是让我们用数学的语言表达一下超参选择是个什么样的问题吧：

我们想要找到$f(x_ 1,⋯, x_ d )$的最小值点，其中$x_ i$代表超参可能是连续或者离散的：$x_ i\in[a,b]$或者$x_ i \in \{0,1,2,...\}$，我们只能通过查询单点处的函数值，没法获得1阶和2阶的梯度信息。

而且，这个$f$很可能是一个没有太多好的性质的函数，比如不一定有convex的性质。我们希望找最小值的过程尽可能的sample-efficient,因为每一次实验都可能需要好几小时甚至天。

## Bayesian Optimization

这个算法的high-level idea是这样的：
>   - **Step 1**: Assume a prior distribution for the loss function $f$. 
>   - **Step 2**: Select new sample(s) that balances exploration and exploitation. Either the new sample(s) gives better result.Or gives more information about $f$.
>   - **Step 3**: Update prior with the new sample(s) using Bayes' rule. Go to Step 2.

在这里，我们需要两个模型：
- **代理模型（surrogate model）**: 用于对目标函数进行建模。
- **采集函数（acquisition function）**：用来平衡探索和利用，指导选择下一个采样点。 

### Gaussian Process
这里一般对于代理模型的建模，我们将高斯函数拓展到无穷维空间上去，其中每一个输入$x$都对应一个维度。通常假设每个维度的均值为 0（即$\mathbb{E}[f(x_ i)]=0$），这样能简化模型。如果有先验知识，也可以设为非零的均值函数。两个输入点 $x_ i$和$x_ j$的相关性由核函数$\mathbb{E}[f(x_ i)f(x_ j)]=K(x_ i,x_ j)$定义。这个核函数表达了我们对函数光滑性、相似性等性质的假设。

经过一些数学，我们有如下结论：

- 当给定 $m$ 个已有的观测点 $(x_ 1, y_ 1), \cdots, (x_ m, y_ m)$ 时，我们有：
$$
f(x) \mid ((x_ 1, y_ 1), \cdots, (x_ m, y_ m)) \sim \mathcal{N}\left(k_ *^T \Sigma^{-1} y, K(x, x) - k_ *^T \Sigma^{-1} k_ *\right)$$

- 其中：
  - $k_ * = [K(x_ 1, x), \cdots, K(x_ m, x)]$
  - $y = (y_ 1, \cdots, y_ m)$
  - $\Sigma$ 是 $m \times m$ 的协方差矩阵，由 $K(x_ i, x_ j)$ 组成

由此我们可以获得我们当前模型在特定输入下的输出的均值和方差，之后我们的采集函数可以通过对于计算例如期望改进（Expected Improvement, EI）在当前最优值的基础上寻找改进期望值。

这个具体的过程鼠鼠我也不是很懂，放几个链接大家有兴趣看看去吧：

- [链接1](https://zhuanlan.zhihu.com/p/349600542)
- [链接2](https://zhuanlan.zhihu.com/p/358606341)

看点visualization:

![](../img/ml5/image1.png#center)

![](../img/ml5/image2.png#center)

![](../img/ml5/image3.png#center)

![](../img/ml5/image4.png#center)

![](../img/ml5/image5.png#center)

![](../img/ml5/image6.png#center)

![](../img/ml5/image7.png#center)


## Gradient Optimization

对于连续的超参，我们可以通过梯度递降的方法来优化：

![](../img/ml5/image8.png#center)

我们考虑一个简单的例子，在线性回归中寻找学习率的最佳值。我们知道这个损失函数是：$$L(w)=\frac{1}{2}\sum_ {i=1}^n(w^T x-y)^2$$
对于其参数$w$求梯度：
$$\nabla_ w L(w)=\sum_ {i=1}^n(w^T x-y)x$$
然后梯度递降：
$$w_ 1=w_ 0-\eta \nabla_ w L(w_ 0)$$
$$w_ 2=w_ 1-\eta \nabla_ w L(w_ 1)$$
那么我们想要求$\nabla_ \eta f(w_ 0,\eta):=L(w_ 2)$，其实只用使用下链式法则：
$$\nabla_ \eta f(w_ 0,\eta)=\nabla_ w L(w_ 2)\cdot \nabla_ \eta w_ 2=\sum_ {i=1}^n(w_ 2^T x-y)x \cdot \nabla_ \eta w_ 2$$
对于$\nabla_ \eta w_ 2$,我们继续求导：
$$\nabla_ \eta w_ 2=\nabla_ \eta w_ 1-\nabla_ w L(w_ 1)-\eta \nabla_ \eta(\nabla_ w L(w_ 1))$$
然后接着顺着往下求。

### Memory Problem
刚才是naive的反向传播梯度的方法，但是这个会带来一个很显著的问题，就是对于计算$\eta \nabla_ \eta \nabla_ w L(w_ i),i=1,2,...,T$的梯度，我们假如将$w_ 1,...,w_ T$全部存入内存的话，内存是会爆炸的，因为太大了。那么，有什么办法解决吗？让我们对SGD with momentum的优化器进行分析：

![](../img/ml5/image9.png#center)
$v_ t$如何理解呢？$v_ t$可以理解为一个历史梯度状态的压缩(等比平均?)，因为当前的梯度的方差可能太大，所以有这样一种soft的更新方法有利于让优化过程更鲁棒的。而且收敛率也更快，$O(\frac{1}{T^2})$快于SGD的$O(\frac{1}{\sqrt{T}})$.

这里的核心出装在于：
$$
v_ {t+1} = \gamma v_ t - (1 - \gamma) \nabla_ w L(w_ t)
$$

$$
w_ {t+1} = w_ t + \eta v_ {t+1}
$$
也就是说因为我只需要$w_ t$和$v_ t$,我们就可以左脚踩右脚，算出之前的$w_ i$和$v_ i$了，所以我们只需要存一对当前时刻的$w_ t$和$v_ t$即可了。

![](../img/ml5/image10.png#center)

听起来挺好的，但是因为这是计算机科学不是数学，我们存的数是会有精度损失的，也就是说因为$v_ t$的精度有限，所以其实还是会丢失一部分历史信息。而且这个问题不能忽略，因为误差累计是指数上涨的。那么怎么解决呢？

我们可以用整数表达一切，在除什么的时候，将余数放入一个Buffer中，然后再乘回来的时候把这个余数加回来。

Comments: 这种方法只适用于连续的超参优化，而且优化过程也比较容易卡在local minima。

## Random Search

顾名思义，就是对于可选的参数区间随机的取样。在实际中效果很好，比Grid Search（枚举所有可能）要样本利用率高很多。

## Multi-Arm Bandits

### Best Arm Identification

这里的多臂老虎机的目标和强化学习中比如UCB算法是不同的，对于UCB类的算法，他的目标是获得最高的累积回报，而在这里的setup是去找到最佳的老虎机。

有$n$个臂，每次拉动一个臂时都会得到一个奖励，该奖励是一个具有期望值$v_ i$的有界随机变量。
每次选择一个臂并拉动时，会得到其奖励的一个独立样本。
在固定预算的情况下，我们如何找到期望值$v_ i$最大的臂？

### Successive Halving(SH) Algorithm

![](../img/ml5/image11.png#center)
也就是说每一轮我们把预算平均分配给还存活的机器，然后计算获得的回报的均值，然后去掉回报小的那一半机器，再进入下一轮。下图为一示例：

![](../img/ml5/image12.png#center)

WLOG, 我们假设$v_ 1>v_ 2\geq...\geq v_ n$,定义$\Delta_ i=v_ 1-v_ i$.

> **Thm.** With Probability $1-\delta$, the algorithm finds the best arm with
> $$B = \Theta\left(H_ 2 \log n \log\left(\frac{\log n}{\delta}\right)\right)$$
> arm pulls. $H_ 2=max_ {i>1}\frac{i}{\Delta_ i^2}$.

证明如下：

如果第一个arm在第$r$轮之前没有被淘汰，那么对于任意不是arm 1的$i \in S_ r$, 对于每一个arm有$\frac{B}{|S_ r|log(n)}$的采样率，所以由Hoeffding Inequality:

$$Pr[\hat{v}_ 1^r<\hat{v}_ i^r]\leq \exp(-\frac{1}{2}\frac{B\Delta^2_ i}{|S_ r|\log(n)})$$

令$n_ r=\frac{n}{2^{r+2}}$,也就是说我们在round r把这些还存活的arm进行4等分。接下来我们把这个arm对应的真实值小的后3/4记为$S_ r'$,那么如果我们用$N_ r$记录$S_ r'$中在这一轮中的平均值大于arm1的arm的数量，有：
$$\mathbb{E}[N_ r]\leq\sum_ {i \in S_ r'} \exp(-\frac{1}{2}\frac{B\Delta^2_ i}{|S_ r|\log(n)})\leq |S_ r'|\exp(-\frac{1}{8}\frac{B\Delta^2_ {n_ r}}{n_ r \log(n)}) $$
接着用Markov Inequality:
$$Pr[N_ r>\frac{1}{3}|S_ r'|]\leq 3 \exp(-\frac{1}{8}\frac{B\Delta^2_ {n_ r}}{n_ r \log(n)})$$
也就是说，有很高概率并没有那么多不那么好的机器的empirical mean比最好的机器的empirical mean大。

最后，因为只有在后3/4中有至少1/3比arm 1大的时候，arm 1才有可能被淘汰，所以说arm 1在任意一轮被淘汰的概率最多是：
$$3 \sum_ {r=1}^{\log n} \exp(-\frac{1}{8}\frac{B\Delta^2_ {n_ r}}{n_ r \log(n)})\leq 3 \log(n) \exp(-\frac{B}{8 H_ 2 \log(n)})$$
这等价于
$$B = \Omega\left(H_ 2 \log n \log\left(\frac{\log n}{\delta}\right)\right)$$
### Application to HyperParameter Tuning
在超参选择上，每一个超参的set都是一个arm，在初始阶段，我们随机选择许多配置。

在setting上不太一样的点是：

- **假设**：可以观察到中间结果，能够在训练中途终止一些配置。

- **操作**：在训练过程中移除较不具前景的超参对应的实验。

另一不一样的点是，我们并不是直接从随机变量中抽取样本，而是可以通过付出一定的代价来获得更加准确的观测值，这个代价就是更久的观察时间。最后观测到的值作为返回值。

![](../img/ml5/image13.png#center)

也就是说对于所有 $i \in [n], k \geq 1$，令 $\ell_ {i,k} \in \mathbb{R}$ 为臂 $i$ 的一个序列，假设：
    $$
    v_ i = \lim_ {\tau \to \infty} \ell_ {i,\tau} \quad \text{存在}
    $$
那么对应的投入更多的budget就是对于运行更多的epoch数。

![](../img/ml5/image14.png#center)
一个实际运行的例子：
![](../img/ml5/image15.png#center)

那么在这样的setting下有没有理论的保证呢？

我们首先引入一些记号：

- $\gamma_ i (t)$: 关于$t$单调不增，它给出了每个 $t$ 对应的最小值，使得：
  $$|\ell_ {i,t} - v_ i| \leq \gamma_ i(t)$$
  也就是说它是曲线的“包络线”，表示当前观测值距离极限$v_ i$的接近程度。
- $\gamma_ i^{-1}(\alpha) = \min\{t \in \mathbb{N}: \gamma_ i(t) \leq \alpha\}$
  表示首次进入与 $v_ i$ 的 $\alpha$-邻域的时间点,值得注意的是，这里我们假设一旦我们进入，我们就再也不会出去了。
  
如果 $ k_ i \geq \gamma_ i^{-1}\left(\frac{v_ i - v_ 1}{2}\right) $ 且 $ k_ 1 \geq \gamma_ 1^{-1}\left(\frac{v_ i - v_ 1}{2}\right) $，则臂 $ i $ 和臂 $ 1 $ 可以被分开（即区分出优劣)。

> **Theorem**：\
>  令 $\bar{\gamma}(t) = \max_ i \gamma_ i(t)$，则有：$$B \geq 2 \log_ 2(n) \left( n + \sum_ {i=2,\dots,n} \bar{\gamma}^{-1}\left(\frac{v_ i - v_ 1}{2}\right) \right)$$
> 在以上条件下，SH算法能够返回最佳臂。

证明如下：

注意到：
  $$
  B' = 2 \left( n + \sum_ {i=2,\dots,n} \bar{\gamma}^{-1} \left( \frac{v_ i - v_ 1}{2} \right) \right)
  $$

每个臂被拉的次数为：$\frac{B'}{|S_ r|}$,其中：
  $$
  \frac{B'}{|S_ r|} > \bar{\gamma}^{-1} \left( \frac{v_ {\lfloor\frac{|S_ r|}{2}\rfloor+1} - v_ 1}{2} \right)
  $$
这个结论是初等数学结论，读者不难自证。

如果：
  $$
  k_ i \geq \gamma_ i^{-1} \left( \frac{v_ i - v_ 1}{2} \right), \quad k_ 1 \geq \gamma_ 1^{-1} \left( \frac{v_ i - v_ 1}{2} \right)
  $$
  那么臂 $i$ 和臂 $1$ 可以被区分开。
  
  因此，在第 $k$ 轮中，我们知道臂 $\lfloor |S_ r| / 2 \rfloor + 1$ 和臂 $1$ 已经被区分开。
  
  所以我们在$S_ {\log_ 2(n)}$轮中就能够辨认最佳臂1了。
## Neural Architecture Search

![](../img/ml5/image16.png#center)

这里的任务是Given a specific task. Find the best network structure for this task.

一些成功的工作包括：
- 强化学习

![](../img/ml5/image17.png#center)

![](../img/ml5/image18.png#center)

- 随机搜索

![](../img/ml5/image19.png#center)

- **传统NAS算法**需要大量的GPU计算资源。
- 通常只能用于一些代理任务（小规模/辅助任务）：
  - 在小型数据集上训练。
  - 使用少量的神经网络模块（blocks），仅训练几个epoch。
  - 计算代价较低，但扩展到大规模任务时效果有限。
### ProxyLess NAS

我们希望找到一个算法，使其能够适用于更大的任务。因此引入ProxylessNAS。

![](../img/ml5/image20.png#center)

**方法**：
  - 对于每一层（或边），考虑所有可能的结构
  - 定义 $N$ 个组件 $o_ i$（例如不同的卷积滤波器大小、Identity层、池化层等）
  - **联合训练结构（layer被选择的概率）与权重（内部的权重和偏置）**

对于模型输出的类型，有如下三种方法：
1. **One-shot方法**（Bender et al., 2018）：
   - 输出为所有组件的加权和：
     $$
     \sum_ {i=1}^N o_ i(x)
     $$
   - **性能不足**。

2. **DARTS方法**（Liu et al., 2018）：
   - 使用权重 $ \alpha_ i $ 定义输出：
     $$
     \sum_ {i=1}^N p_ i o_ i(x), \quad p_ i = \frac{e^{\alpha_ i}}{\sum_ {j=1}^N e^{\alpha_ j}}
     $$
   - **缺点**：内存效率低，因为需要存储所有$N$条路径。最终模型只包含一条路径。

3. **ProxylessNAS方法**：
   - 二值化路径，定义布尔变量 $g$（一个one-hot向量）：
     $$
     g =
     \begin{cases}
       [1, 0, \dots, 0], & \text{概率为 } p_ 1 \\
       \vdots \\
       [0, 0, \dots, 1], & \text{概率为 } p_ N
     \end{cases}
     $$
   - 输出依赖于单一路径：
     $$
     \sum_ {i=1}^N g_ i o_ i(x) =
     \begin{cases}
       o_ 1(x), & \text{概率为 } p_ 1 \\
       \vdots \\
       o_ N(x), & \text{概率为 } p_ N
     \end{cases}
     $$
    注：这里的$p_ {[1:N]}$是根据$\alpha_ {[1:n]}$通过softmax采样得到的。
   - **优势**：大幅节省内存。只需存储单一路径，而不存储所有 \(N\) 条路径。
   
最后看一下训练过程：
  
1. 交替训练网络结构和权重：
   - 在训练权重时，冻结 $ \alpha_ i $，并采样结构。
   - 在训练 $ \alpha_ i $ 时，冻结权重。

2. 如何学习 $\alpha_ i$：
   - 链式法则近似计算 $ \frac{\partial L}{\partial \alpha_ i} $：
     $$
     \frac{\partial L}{\partial \alpha_ i} = \sum_ {j=1}^N \frac{\partial L}{\partial g_ j} \frac{\partial g_ j}{\partial \alpha_ i} \approx \sum_ {j=1}^N \frac{\partial L}{\partial g_ j} \frac{\partial p_ j}{\partial \alpha_ i}
     $$
   - 其中：
     $$
     \frac{\partial p_ j}{\partial \alpha_ i} = \sum_ {j=1}^N \delta_ {ij} p_ j (1 - p_ i) - p_ i p_ j
     $$
     $ \delta_ {ij} = 1 $ 如果 $ i = j $，否则为0。
更多的细节可以看一下[原论文](https://arxiv.org/pdf/1812.00332)。

## Misc
2025年了，祝大家新年快乐！

至于认识我的朋友，解释下为什么今年没有发朋友圈，因为实在是没有太多值得说的东西，有很多under-construction的事情，所以，2025对我、也希望对大家，会是很让人兴奋的一年。
![](../img/ml5/image21.png#center)
调超参是一个听起来很有趣但实际上大家都在做Graduate Student Search的领域，也希望在做AI相关领域科研的朋友能够在2025年有“金手指”，调参手到擒来！


