---
title: "Machine Learning Series: 3.Unsupervised Learning(II)"
date: 2025-01-02
draft: false
ShowToc: true
tags: ["machine-learning", "computer-science", "algebra", "math", "artificial-intelligence", "algorithm"]
summary: "This is the third article in the Machine Learning Series. It covers the second part of unsupervised learning, including topics like Clustering, Spectral Graph Clustering, SimCLR, SNE and t-SNE."
---
## Clustering
这里是正统的无监督学习了，我们想要把数据点分组，希望在同一组的数据点有一些共同的性质。
### K-Means
形式化一下，我们被给予：
- n个数据点$(x_ 1,...,x_ n), x_ i \in \mathbb{R}^d$

想要把他们partition成：
- k个cluster: $\{S_ 1,...,S_ k\}$

我们的目标是最小化在同一个cluster的点距离cluster中心的距离的平方之和。
$$\arg\min_ S \sum_ {i=1}^k \sum_ {x\in S_ i} ||x-\mu_ i||^2$$
其中$\mu_ i$是$S_ i$的中心。

这个问题是NP-Hard的，但是有一些Heuristic的算法，下面介绍Lloyd Algorithm:

![](../img/ml3/image.png#center)

![](../img/ml3/image2.png#center)

![](../img/ml3/image3.png#center)


![](../img/ml3/image4.png#center)


![](../img/ml3/image5.png#center)
> **Lloyd's Method:**
> 1. Decide $k$
> 2. Randomly pick $k$ centers
> 3. Decide membership of all points by assigning them to the nearest center
> 4. Re-estimate k centers by average of cluster members
> 5. Repeat 3&4 until convergence


很符合直觉，但是在最坏情况下，这个算法会找到arbitrarily-worse的solution,而且即使是对于seperated gaussian clusters都没有分开的保障。

Lloyd算法是保证终止的，因为每一次都会发生聚类的变化，然后聚类一共只有有限种，所以最终会终止。

### Spectral Graph Clustering

对于有的图，p-范数并不能够做到很好的聚类，比如下面这张:

![](../img/ml3/image7.png#center)

这个时候，我们就引入一种思想，对于两个点之间引入一种**相似性**的衡量，假如说两个点之间的相似性超过一个threshold,那么我们用边把他们连接起来，边权$$w_ {i,j}=\text{Similarity}(i,j)$$

理想情况下，我们希望同组之间的边权大，不同组之间的边权小，如果没有边，我们定义$w_ {i,j}=0$.

举一些例子：
1. **$\epsilon$-neighborhood graph (unweighted)**:
   - 定义边权重：
     $$
     w_ {i,j} = 1 \text{ iff } x_ i, x_ j \text{ are } \epsilon\text{-close}.
     $$
   - 即如果 $x_ i$ 和 $x_ j$ 的距离小于 $\epsilon$，则两点之间有一条边。


2. **$k$-nearest neighbor graph**:
   - 如果 $x_ i$ 是 $x_ j$ 的 $k-nn$，或者 $x_ j$ 是 $x_ i$ 的 $k-nn$，则两点之间有一条边。
   - 注意：nearest neighbor 关系不对称。

3. **Fully connected graph**:
   - 定义一个相似性函数（similarity function）来衡量 $x_ i$ 和 $x_ j$ 之间的关系。

下面引入Laplacian Matrix:

> **Graph Laplacian**\
> 下面引入$L=D-A$，其中$D$是$diag\{deg(v_ i)\}$, $A$是邻接矩阵。

这个Laplacian Matrix有很多好性质，比如：

> **Theorem**\
> Given: $G$ 是一个无向图，具有非负权重。
> 1. **零特征值的数量**：
>   - $L$的零特征值的数量等于图 $G$ 中的连通支的数量。
>   - 记连通分量为 $A_ 1, A_ 2, \dots, A_ k$。
> 2. **零特征值的性质**：
>   - 零特征值对应的特征向量由连通分量的指示向量 $I_ {A_ 1}, I_ {A_ 2}, \dots, I_ {A_ k}$ 张成。

证明如下:

1. $L$是半正定的

对于任意的$v\in \mathbb{R}^n$
$$v^T L v=v^T D v-v^T A v$$
$$=\sum_ {i=1}^n d_ i v_ i^2-\sum_ {i,j}v_ i v_ j w_ {i,j}$$
$$=\frac{1}{2}(\sum_ {i=1}^n d_ i v_ i^2-2\sum_ {i,j}v_ i v_ j w_ {i,j}+\sum_ {j=1}^n d_ j v_ j^2)$$
$$=\frac{1}{2}\sum_ {ij} w_ {i j} (v_ i-v_ j)^2\geq 0$$

记$L$的特征值: $0=\lambda_ 1\leq \lambda_ 2 \leq ...\leq \lambda_ n$

2. $\lambda_ 1=0$

这个原因是$$\sum_ j w_ {i j}=d_ i$$
所以我们可以让$v=(1,...,1)^T \in \mathbb{R}^n$,就有$v^T L v=0$

3. 连通图只有一个零特征值

利用反证法，考虑$\sum_ {ij} w_ {i j} (v_ i-v_ j)^2$，除了$v$为全1之外，假设还有一个$v$对应的特征值也是0。这个特征向量一定存在两个分量:$v_ i\neq v_ j$, 而且之间有路径连接: $w_ 1,...,w_ k>0$, 这条路径就会对$\sum_ {ij} w_ {i j} (v_ i-v_ j)^2$贡献非负值，而我们知道这个取0意味着所有的$w_ {i j} (v_ i-v_ j)^2$都取0，所以矛盾。

4. k联通支对应k个0特征向量

我们可以对于顶点进行适当的排序，得到
$$
L=\begin{pmatrix}L_ 1 & 0 & 0 & 0 \\
0 & L_ 2 & 0 & 0 \\
0 & 0 & \ddots & 0 \\
0 & 0 & 0 & L_ n \\
\end{pmatrix}$$
于是我们有$I_ {A_ i}$这一系列的0特征值对应的特征向量，共k个。$\blacksquare$

接下来就该讲讲这个Laplacian Matrix怎么在Clustering里用的了:

#### 使用 Laplacian 找到 $k$ 个聚类的方法：

1. **计算拉普拉斯矩阵 $L$ 的前 $k$ 个特征向量**：
   - 特征向量：$\mu_ 1, \mu_ 2, \dots, \mu_ k$。
   - 对应的特征值接近 0（不一定等于 0）。
2. **构造矩阵 $U$:**
   - $U \in \mathbb{R}^{n \times k}$，以 $\mu_ 1, \mu_ 2, \dots, \mu_ k$ 作为列。


3. **构造特征向量的行向量**：
   - 对于 $i = 1, 2, \dots, n$，令：
     $$
     y_ i \in \mathbb{R}^k \text{ 是 } U \text{ 的第 } i \text{ 行向量}.
     $$
   - 构造点集 $\{y_ i\}_ {i=1, \dots, n}$。


4. **运行 $k$-means 聚类**：
   - 在点集 $\{y_ i\}_ {i=1, \dots, n}$ 上运行 $k$-means 算法，得到 $k$ 个聚类 $C_ 1, C_ 2, \dots, C_ k$


5. **输出最终的聚类结果**：
   - 定义每个聚类的集合：
     $$
     A_ i = \{j | y_ j \in C_ i\}, \quad i = 1, 2, \dots, k.
     $$
   - 输出聚类结果 $A_ 1, A_ 2, \dots, A_ k$。
   
看一个理想情况的例子:

![](../img/ml3/image6.png#center)
这就很漂亮，但是现实情况下一般图是联通的，所以没那么好的事～

最后解决一个technical problem,我们怎么去找最小的特征值、特征向量对呢？其实很简单，比如说$A$是半正定的，我们取 
 $$B = A - \lambda _ {\max} I $$
对$B$做power-method，然后加$\lambda_ {\max}$就ok了。

#### Why This Makes Sense

考虑RatioCut问题:
$$ratiocut(A_ 1,...,A_ k)=\sum_ {i=1}^k \frac{W(A_ i, \overline{A_ i})}{|A_ i|}$$
其中$W(A_ i, \overline{A_ i})$代表这$A_ i, \overline{A_ i}$之间所有边权重之和。

这个问题是NP-Hard的。

**下面先考虑$k=2$的情况:**

假设 $G$ 是连通图,那么全 1 向量是最小的特征向量。

对于满足下面这种形式的$v^A\in \mathbb{R}^n$：
  $$
  v_ i^A = 
  \begin{cases} 
  \sqrt{\frac{|\overline{A}|}{|A|}}, & \text{if } i \in A \\ 
  -\sqrt{\frac{|A|}{|\overline{A}|}}, & \text{o/w}.
  \end{cases}
  $$

我们有

$$
(v^A)^T L v^A =\frac{1}{2}\sum_ {i,j} w_ {i,j} (v^A_ i-v^A_ j)^2 
$$
$$=\frac{1}{2}\sum_ {i\in A, j\in \overline{A}}w_ {i,j}\left(\sqrt{\frac{|\overline{A}|}{|A|}}-\sqrt{\frac{|A|}{|\overline{A}|}}\right)^2+\frac{1}{2}\sum_ {i\in \overline{A}, j\in A}w_ {i,j}\left(\sqrt{\frac{|\overline{A}|}{|A|}}-\sqrt{\frac{|A|}{|\overline{A}|}}\right)^2$$
$$=cut(A,\overline{A})(\frac{|A|}{|\overline{A}|}+\frac{|\overline{A}|}{|A|}+2)$$
$$=cut(A,\overline{A})(\frac{|A|+|\overline{A}|}{|\overline{A}|}+\frac{|\overline{A}|+|A|}{|A|})$$
$$=|V|\cdot ratiocut(A,\overline{A})$$

也就是说对于这种形式的$v^A$，我们minimize RatioCut等价于找到一个好的$A \subset V$:
$$\min_ A ratiocut(A,\overline{A})=\min_ {A \subset V} (v^A)^T L v^A$$
这个当然还是NP-Hard的，但是我们可以relax一下：
$$\min_ v v^T L v \quad \text{s.t. } \langle v, I_ V \rangle=0, ||v||=\sqrt{n}$$
这不就是求第二小的特征向量吗？

然后通过这个形式有差异的$v$中还原$A$:
- naive way: $i \in A \text{ iff } v_ i>\alpha$，其中$\alpha$是某个threshold。
- Run 2-means on $\{v_ i\}$:

诶，我们仔细一看，**这种方式Run 2-mean是和在$(1,v_ i)$上跑2-means是完全一样的，再一想，这不就是我们 Spectral Graph Clustering的算法吗？**至此，豁然开朗。

但是我们还是没完呢，这个只是$k=2$的情况，**对于$k>2$呢？**

定义 $h_ {ij}$：
  $$
  h_ {ij} = 
  \begin{cases} 
  \frac{1}{\sqrt{|A_ j|}}, & \text{if } v_ i \in A_ j \\ 
  0, & \text{o/w}.
  \end{cases}
  $$

给定 $H \in \mathbb{R}^{n \times k}$，使得 $H^T H = I$。


我们可以看到(设 $h_ i$ 是 $(h_ {1i}, \cdots, h_ {ni}) \in \mathbb{R}^n$ 的向量)：

回忆：$$v^T L v = \frac{1}{2} \sum_ {ij} w_ {ij}(v_ i - v_ j)^2$$
  - $$h_ i^T L h_ i = \frac{\text{cut}(A_ i, A_ i^c)}{|A_ i|} = (H^T L H)_ {ii}$$
所以：
    $$
    \text{ratiocut}(A_ 1, \cdots, A_ k) = \sum_ {i=1}^k h_ i^T L h_ i = \sum_ {i=1}^k (H^T L H)_ {ii} = \text{Tr}(H^T L H)
    $$

也就是说对于满足上述$h$的条件的矩阵$H$,我们的**优化目标**是：
  $$
  \text{min } \text{Tr}(H^T L H) \quad \text{s.t. } H^T H = I, h_ {ij} \text{ see above}
  $$

与$k=2$情况类似，我们relax对于$H$的限制，得到:
$$
  \text{min } \text{Tr}(H^T L H) \quad \text{s.t. } H^T H = I
  $$
  
这里有个结论，就是说这个$H$对应的就是$L$最小的$k$个特征向量(受限篇幅与作者的能力（小声），聪明的读者应该不难自证)，然后对$H$的行向量跑k-means就可以还原出聚类$\{A_ {[1:k]}\}$了。所以我们找到Graph Spectral Clustering算法的理论依据了，也就是说我们在解决一个relaxed version的最小化聚类的RatioCut。

## SimCLR

这里讲个Clustering+ Metric Learning的应用，也是袁洋老师ICLR 2024年的崭新工作，证明了SimCLR这种对比学习方法其实就是在similarity graph上折腾了一些操作的spectral clustering。

### What is SimCLR

首先讲讲什么是SimCLR。这个是对比学习的一个算法，比如说给一个被查询的样本$q$，同时还有一个正样本$p_ 1$,对应$N-1$个负样本$\{p_ i\}_ {i=2}^N$。

![](../img/ml3/image8.png#center)
一个现实的例子是我们对于所有输入的图片生成两个augmented图片，那么以其中一个图片作为查询样本，另一个就是正样本，别的图片augment之后的结果就是负样本。

![](../img/ml3/image9.png#center)

这里的$q,p_ 1$可以在pixel space上差距极大，但是我们希望他们在我们学到的semantic space上距离接近。我们的优化目标是**InfoNCELoss**:

$$
L(p,q,\{p_ i\}_ {i=2}^N)=-\log \frac{\exp(-||f(q)-f(p_ 1)||^2)/2\tau}{\sum_ {i=1}^N \exp(-||f(q)-f(p_ i)||^2)/2\tau}
$$

### What is the Similarity Graph Here
这里的Similarity Graph的定义对于所有augmented image构成的$\{X_ i\}$集合，$(X_ i,X_ j)$的边权是
$$\pi_ {i,j}=\Pr[X_ i,X_ j\text{ are sampled together}]$$
有没有一个理想的空间，其中semantic similarity是被自然的捕捉到的呢？答案是有的，这里引入**Reproducing Kernel Hilbert Space**.

### Reproducing Kernel Hilbert Space (RKHS)
给定两个在$Z$空间的物品: $Z_ 1,Z_ 2\in Z$, 考虑$\phi: Z\rightarrow H$, 这里$H$的维度远大于$Z$.

$$k(Z_ i,Z_ j)=\langle \phi(Z_ i),\phi(Z_ j)\rangle_ H$$

这里的$H$就是Hilbert space,而$k$Rreproducing Kernel。

![](../img/ml3/image10.png#center)

因为我们关心的只是样本之间的相似性，所以我们不用真的知道或者能算$\phi(Z_ i)$,我们只需要算样本之间的$k$就ok了。

### Markov Random Fields (MRF)
原论文里是这么说的
> Due to the large size of $\pi$ and in practice $\pi$ is usually formed by using positive samples sampling and hard to explicitly construct, directly comparing $K_ Z$ and $\pi$ can be difficult, so we treat them as MRFs and compare the induced probability distributions on subgraphs instead.

motivation很清晰，那么我们具体怎么采样呢？我们想要sample出的unweighted子图我们设为$W$(也就是说$W_ {ij}\in\{0,1\}$)

$$P(W;\pi)\propto \Omega(W)\cdot \prod_ {\{i,j\}\in [n]^2}\pi_ {i,j}^{W_ {i,j}}  $$
怎么解读呢？
$$s(W,\pi)=\prod_ {\{i,j\}\in [n]^2}\pi_ {i,j}^{W_ {i,j}}$$
对于可能的$W$，我们根据他的score function来采样，然后我们看看采样出来的$W$是不是正确的形状，比如

$$\Omega(w)=\prod_ i \mathbb{1}[\sum_ jW_ {ij}=1]$$
这个被称为Unitary out-degree filter，也就是说每一个顶点要求有且仅有一条出边。

接下来考虑$P(W;K_ z)$:
$$P(W;K_ z)\propto \Omega(W)\cdot \prod_ {\{i,j\}\in [n]^2}k(Z_ i,Z_ j)^{W_ {i,j}}  $$


![](../img/ml3/image11.png#center)

### InfoNCE and Spectral Clustering

好了，概念搭的差不多了，落地我们通过比较$W_ X,W_ Z$的差异来反应我们理想中$\pi,K_ Z$的差值。这一差值可以通过Cross Entropy衡量：
$$H_ \pi^k(Z)=-\mathbb{E}_ {W_ X\sim P[\cdot;\pi]}\log P[W_ Z=W_ X;K_ z]$$
接下来如果能证明

1. InfoNCE和$H_ \pi^k(Z)$等价
2. $H_ \pi^k(Z)$和Spectral Clustering等价

我们的任务就完成了。

#### InfoNCE和$H_ \pi^k(Z)$等价
先从直觉上理解下这件事情:
$$H_ \pi^k(Z)=-\mathbb{E}_ {W_ X\sim P[\cdot;\pi]}\log P[W_ Z=W_ X;K_ z]$$
- $W_ X\sim P[\cdot;\pi]$: 我们从$\pi$上取样，代表着Data Augmentation Step

接下来用一个原论文中的重要结论: 对于Unitary out-deg $\Omega(w)$, 
$$W_ i\sim M(1, \frac{\pi_ i}{\sum_ j \pi_ {i,j}})$$
也就是我们按照$\frac{\pi_ i}{\sum_ j \pi_ {i,j}}$取样一个one-hot vector。
![](../img/ml3/image12.png#center)
因为每一行是独立的，所以我们有:
$$H_ \pi^k(Z)=-\sum_ i\mathbb{E}_ {W_ {X,i}}\log P[W_ {Z,i}=W_ {X,i};K_ z]$$
这里$W_ {X,i}$代表$W_ X$的第$i$行(one-hot 向量)。

接下来想一下InfoNCE在说什么事情：

$$InfoNCE=-\sum_ {i=1}^n\log \frac{\exp(-||f(X_ i)-f(X_ i')||^2)/2\tau}{\sum_ {j=1}^N \exp(-||f(X_ i)-f(X_ j))||^2)/2\tau}$$

如果我们定义
$$Q_ i=\frac{K_ {Z,i}}{||K_ {Z,i}||_ 1} $$
作为$P(\cdot;K_ Z)$的分布

那么
$$\text{InfoNCE}=-\sum_ {i=1}^n \sum_ {i'=1}^n \Pr[W_ {X,i,i'}=1] \log Q_ {i,i'}=-\sum_ i\mathbb{E}_ {W_ {X,i}}\log P[W_ {Z,i}=W_ {X,i};K_ z]=H_ \pi^k(Z)$$

#### $H_ \pi^k(Z)$和Spectral Clustering等价

$$H_ \pi^k(Z)=-\mathbb{E}_ {W_ X\sim P[\cdot;\pi]}\log P[W_ Z=W_ X;K_ z]$$
其中
$$P[W_ Z=W_ X;K_ z] \propto \Omega(W_ X) \prod_ {\{i,j\}\in [n]^2} K_ {Z_ {i,j}}^{W_ {X_ {i,j}}}$$

然后我们就可以显式的表达概率
$$R(Z)=\sum_ {W}\Omega(W)\cdot \prod_ {\{i,j\}\in [n]^2}K_ {Z_ {i,j}}^{W_ {i,j}}$$

所以说
$$\log P[W_ Z=W_ X;K_ z]=\sum_ {i,j}W_ {X_ {i,j}}\log K_ {Z_ {i,j}}+\log \Omega(W_ X)-\log R(Z)$$

这里对于fixed $W_ X$，$\log \Omega(W_ X)$是常数，所以:
  $$
  \arg\min_ Z H_ \pi^k(Z) = \arg\min_ Z -\mathbb{E}_ {W_ X \sim P(\cdot; \pi)} \Bigg[\sum_ {(i,j) \in [n]^2} W_ {X,ij} \log k(Z_ i, Z_ j) - \log R(Z) \Bigg]
  $$
  $$
  = \arg\min_ Z -\mathbb{E}_ {W_ X \sim P(\cdot; \pi)} \sum_ {(i,j) \in [n]^2} W_ {X,ij} \log k(Z_ i, Z_ j) + \log R(Z)
  $$
回忆下，$k$ 是 Gaussian 分布来的：

  $$\log k(Z_ i, Z_ j) = -\frac{||Z_ i - Z_ j\||^2}{2\tau}
    $$

所以原式
$$
  = \arg\min_ Z \mathbb{E}_ {W_ X \sim P(\cdot; \pi)} \frac{1}{2\tau} \sum_ {(i,j) \in [n]^2} W_ {X,ij} \|Z_ i - Z_ j\|^2 + \log R(Z)
  $$

  $$
  = \arg\min_ Z \mathbb{E}_ {W_ X \sim P(\cdot; \pi)} \frac{1}{\tau} \operatorname{tr}(Z^T L(W_ X) Z) + \log R(Z)
  $$

  $$
  = \arg\min_ Z \frac{1}{\tau} \operatorname{tr}(Z^T L^* Z) + \log R(Z)\quad\blacksquare
  $$
  真长啊。
## t-SNE
最后讲一个Data Visualization/ Dimension Reduction的算法，首先回忆一下NCA:

原空间上两个点$(x_ i,x_ j)$的similarity是这么定义的:
  $$
  p_ {j|i} = \frac{\exp\left(-\frac{\|x_ i - x_ j\|_ 2^2}{2\sigma_ i^2}\right)}{\sum_ {k \neq i} \exp\left(-\frac{\|x_ i - x_ k\|_ 2^2}{2\sigma_ i^2}\right)}
  $$

我们做映射后$ x_ i \to y_ i = f(x_ i) $:
  $$
  q_ {j|i} = \frac{\exp\left(-\|y_ i - y_ j\|_ 2^2\right)}{\sum_ {k \neq i} \exp\left(-\|y_ i - y_ k\|_ 2^2\right)}
  $$

我们希望$p,q$能对应的上。

在训练中，用KL-Divergence做损失函数：
$$
  L = \sum_ i \text{KL}(P_ i \| Q_ i) = \sum_ i \sum_ j p_ {j|i} \log \frac{p_ {j|i}}{q_ {j|i}}
$$

细心的读者发现和无监督学习(I)里讲的NCA的优化目标好像略有区别(差个log)，但无伤大雅，我们不去管他。**怎么去选$\sigma_ i$呢?**

### SNE

用户选择一个**Perplexity**，这个直觉的理解是对于有效邻居数量的
smooth measure。
  $$
  \text{Perp}(P_ i) = 2^{H(P_ i)}
  $$
  $$
  H(P_ i) = -\sum_ j p_ {j|i} \log p_ {j|i}
  $$
接下来我们对$\sigma_ i$做二分查找找到合适的$\sigma_ i$值，这就叫**SNE algorithm**。

### t-SNE

SNE很经典，但是有如下两个缺点的
1. 要优化很多个loss，你看loss其实是对i和j求了两遍和，相当于每个pair贡献了一个loss，或者说有$n$个分布要去学

2. **拥挤问题(crowding problem)**

![](../img/ml3/image13.png#center)
比如说考虑一个$\{0,1\}^d$的grid,对于$r=10$的情况有$2^{10}$个可以放的位置，但是投影到$r=2$上就只有$2^2$个了，全挤到一块分不开了。

**为了解决第一个问题**，t-SNE的做法是，把所有的距离放一起做运算，捏成一个概率分布，优化一个
single概率分布的loss:
$$p_ {i,j}=\frac{p_ {j|i}+p_ {i|j}}{2n}$$
这样设置是为了
$$\sum_ {i j} p_ {i j}=1$$

为什么要变成一个distribution呢，因为计算梯度更容易、更快。

**为了解决第二个问题**，t-SNE的想法是换一个更heavy-tail的distribution，这样保持相对距离，绝对距离改变，就还能分得开。

也就是:
- 高维中近距离的点，在低维中距离要变得更小
- 高维中远距离的点，在低维中距离要变得更大

这里用student t-distribution就很合适,因为更heavy-tail:

距离由$\frac{1}{1+||y_ i-y_ j||^2}$刻画
$$p_ {i j}=\frac{(1+||y_ i-y_ j||^2)^{-1}}{\sum_ {k\neq l}(1+||y_ l-y_ k||^2)^{-1}}$$

没了，感觉这一部分大量参考了[这篇文献](https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)。

### A side note:
If you are interested in clustering, you can also check out [this website](https://leo1oel.github.io/clustering/), which contains a survey of clustering algorithms(the pdf file link is in the website). It's a project done by me and [Yiming Liu](https://leo1oel.github.io/)