---
theme: seriph
background: black
class: text-center
highlighter: shiki
lineNumbers: false
download: true
exportFilename: optimization-in-deep-learning
export:
  format: pdf
  timeout: 30000
  dark: true
  withClicks: false
  withToc: false
info: ''
drawings:
  persist: false
title: Optimization in Deep Learning
---

# Optimization in Deep Learning

~~从入门到放弃~~

Bingjie YAN

bj.yan.pa@qq.com

ICT, CAS

<div
  v-motion
  :initial="{ x:0, y: 40, opacity: 0}"
  :enter="{ y: 0, opacity: 1, transition: { delay: 100 } }">

[Powered by @slidev](https://sli.dev/)

</div>

<!--
大家好，我是闫冰洁，这次我的知识点分享主要是李沐老师《动手学深度学习》第 11 章优化部分的 11.1-11.5 节，这个 slide 是通过 slidev所以大家也可以在自己的设备上输入地址打开这个 slide
-->

---

## TOC

1. Optimization Problem & Deep Learning
2. Basic Definition (Convexity, Lipschitz Continuity, Smoothness, etc.)
3. Gradient Decent & Mini-batch Stochastic Gradient Decent
4. Convergence Analysis
5. (Optional) Convergence Analysis for Distributed Machine Learning
6. (Optional) Convergence Analysis for Federated Learning

<!--
11.1-11.5 主要有几个部分，我重新组织了一下，一个部分就是优化问题的定义，还有优化中一些常见的定义例如凸性、拉普利斯连续、光滑等等，后面就是梯度下降，随机梯度下降，小批量随机梯度下降，后面会有一下收敛性的分析
-->

---

## Note:


1. 会对李沐老师的《动手学深度学习》中的优化部分进行一些优化和补充，有侧重点，对于深度学习中不常用的优化内容可能会省略。

2. 优化和收敛性分析在很多领域都越来越重要，尤其是近年顶会中的 FL，几乎都会有 Convergence Analysis 的部分，我会尽可能讲一些对大家科研可能有帮助的实用的优化技巧和收敛性分析方法，希望对大家有用 :)

3. 本次分享的内容也是我在学习优化和收敛性分析时的一些经验，可能会有一些错误，欢迎大家交流和指正！

<!--
这里我稍微修改了重组了一下李沐老师的大纲，我们组大多都是做深度学习的，所以对深度学习中不常用的优化内容会进行取舍，比如牛顿法的分析，这是一个利用二阶梯度进行优化的，但是深度学习中我们并不常用

其次就是优化的收敛性分析在...
-->

---
layout: center
class: text-center
---

## Optimization Problem & Deep Learning

优化问题、深度学习中的优化目标、深度学习中优化的挑战

<!--
OK，那我们第一部分就是优化与深度学习
-->

---

## Optimization Problem Definition

<br/>

一个优化问题通常定义为：

$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_i(x) = 0, \quad i = 1, \ldots, p
\end{aligned}
$$

其中 $f(x)$ 称为目标函数，$x$ 称为优化变量，$\text{s.t.}$ 表示满足的约束条件

<!--
这是一个优化问题的基本定义，首先是我们有目标函数 f(x)，其中 x 是我们要优化的变量， s.t. 是 subject to，表示约束条件，g_i, f_i 是优化变量需要满足的一些不等式和等式约束
-->

---

## Optimization Problem & Deep Learning

深度学习中优化函数：损失函数 -> 最小化的目标函数

由于优化算法的目标函数通常是基于训练数据集的损失函数，因此优化的目标是减少**训练误差**，但其实我们希望的是减少**泛化误差**，也就是在验证集或测试集上误差减小。

因此在深度学习中，我们不仅需要考虑优化算法的收敛性，还需要考虑优化算法的泛化性，防止**过拟合**。

然而训练数据集的最小经验风险可能与最小风险（泛化误差）不同 <twemoji-smiling-face-with-tear />

<!-- center -->
<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_optimization-intro_70d214_33_1.svg" class="h-60 rounded" />
</div>

<!--
在深度学习中呢，我们要优化的其实就是损失函数，损失函数一般包含了很多我们想要最小化内容的组合
-->

---

## Challenges in Deep Learning Optimization

<br/>

### 1. 局部最小值(local minimum)

<br/>

对于任何目标函数 $f(x)$，如果在 $x$ 处对应的值 $f(x)$ 小于在 $x$ 附近任意其他点的 $f(x)$ 值，那么 $f(x)$ 可能是局部最小值。如果在 $f(x)$ 处的 $x$ 值是整个域中目标函数的最小值，那么 $f(x)$ 是全局最小值。

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_optimization-intro_70d214_48_0.svg" class="h-60 rounded" />
</div>


---

## Challenges in Deep Learning Optimization

<br/>

### 2. 鞍点(saddle point)

<br/>

鞍点是指函数的所有梯度都消失但既不是全局最小值也不是局部最小值的任何位置。这时优化可能会停止，尽管它不是最小值。

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_optimization-intro_70d214_63_0.svg" class="h-60 rounded" />
  <img src="/assets/output_optimization-intro_70d214_78_0.svg" class="h-60 rounded" />
</div>


---

## Challenges in Deep Learning Optimization

<br/>

### 3. 梯度消失

在引入激活函数前，神经网络的每一层都是线性的，因此多层神经网络的输出也是线性的，这样就无法解决非线性问题。因此我们引入了激活函数，使得神经网络可以解决非线性问题。但是，激活函数的导数可能会很小，这样就会导致梯度消失，使得优化算法无法继续优化。

举个栗子<twemoji-chestnut />：

使用 $f(x)=\tanh(x)$ 激活函数，恰好从 $x=4$ 处开始，然而 $f'(x)=1-\tanh^2(x)$，因此 $f'(4)=0.0013$，这意味着在 $x=4$ 处的梯度很小，可能优化将会停滞很长一段时间。

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_optimization-intro_70d214_93_0.svg" class="h-50 rounded" />
</div>


---

## Challenges in Deep Learning Optimization

<br/>

可以看出 **局部最小值** 和 **鞍点** 挑战都是由于目标函数的非凸性导致的。

而 **梯度消失** 可能会导致优化停滞，重参数化通常会有所帮助。对参数进行良好的初始化也可能是有益的。

但尽管深度学习是非凸的，但它们也经常在局部极小值附近表现出一些凸性。

凸性在优化算法的设计中起到至关重要的作用， 因为在这种情况下对算法进行分析和测试要容易。

很多深度学习的收敛性分析也是基于凸性的。


---
layout: center
class: text-center
---

## Basic Definition

凸集、凸函数、强凸、Lipschitz 连续、光滑性


---

## Convex Set

一个集合 $\mathcal{X}$ 是凸集，当且仅当对于任意 $x, y \in \mathcal{X}$ 和 $\alpha \in [0, 1]$，都有：

$$
\alpha x + (1-\alpha)y \in \mathcal{X}
$$


<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/pacman.svg" class="h-50 rounded" />
</div>


---

## Convex Set

假设两个凸集 $\mathcal{X}$ 和 $\mathcal{Y}$，那么它们的交集 $\mathcal{X} \cap \mathcal{Y}$ 也是凸集。

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/convex-intersect.svg" class="h-50 rounded" />
</div>


---

## Convex Set

考虑两个不相交的凸集 $\mathcal{X}$ 和 $\mathcal{Y}$，那么它们的并集 $\mathcal{X} \cup \mathcal{Y}$ 不一定是凸集。


<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/nonconvex.svg" class="h-50 rounded" />
</div>


---

## Convex Function

<br/>

给定一个凸集 $\mathcal{X}$，如果对于任意 $x, y \in \mathcal{X}$ 和 $\alpha \in [0, 1]$，都有：

$$
f(\alpha x + (1-\alpha)y) \leq \alpha f(x) + (1-\alpha)f(y)
$$

则函数 $f$ 是凸函数。


<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_convexity_94e148_18_1.svg" class="h-50 rounded" />
</div>

<!--
这三个函数，左右都是凸函数，中间的不是凸函数
-->

---

## Properties of Convex Function

<br/>

### 1. 局部极小值是全局极小值

<br/>

如果函数 $f$ 是凸函数，假设 $x^{*}\in \mathcal{X}$ 是 $f$ 的局部极小值，那么 $x^{*}$ 也是 $f$ 的全局极小值。

我们可以通过反证法证明：

假设 $x^{*}$ 是 $f$ 的局部极小值，则存在一个很小的正值 $p$，使得当 $x\in \mathcal{X}$ 且 $0 < | x - x^{*} | < p$ 时，都有 $f(x) \geq f(x^{*})$。

现在假设 $x^{*}$ 不是 $f$ 的全局极小值，那么存在 $x \in \mathcal{X}$ 使得 $f(x) < f(x^{*})$。当 $\alpha = 1-\frac{p}{|x - x^{*}|}$ 时，我们有：

$$
\begin{aligned}
f(\alpha x + (1-\alpha)x^{*}) &\leq \alpha f(x) + (1-\alpha)f(x^{*}) \\
&< \alpha f(x^{*}) + (1-\alpha)f(x^{*}) \\
&= f(x^{*})
\end{aligned}
$$

这与 $x^{*}$ 是 $f$ 的局部极小值相矛盾。故 $x^{*}$ 是 $f$ 的全局极小值。

<!--
这里是凸函数的一些性质，第一点就是
-->

---

## Properties of Convex Function

<br/>

### 2. 一阶导数是单调递增的

<br/>

### 3. 二阶导数是非负的

<br/>

...

<!--
凸函数还有一些其他性质，但我们主要还是用前面第一点的性质
-->

---

## Strong Convex

<br/>

通常我们会遇到凸函数，但是在实际中，我们更希望函数更加凸，即函数的曲率更大，这样可以加快函数的收敛速度。因此我们引入了 $\mu$-强凸的概念，$\mu$-强凸的定义如下：


考虑实值函数 $f: R^{d}\to R$，和模 $||\cdot||$，如果对任意自变量 $x,y\in R^d$，都有下面不等式成立：


$$
f(y) \geq f(x)+\nabla f(x)^{\top}(y-x)+\frac{\mu}{2}||y-x||^2, \quad \forall x, y \in \mathcal{D}
$$


则称函数 $f$ 关于模 $||\cdot||$ 是 $\mu$-强凸的。


<div class="flex flex-wrap justify-center gap-4">
  <img src="/strong-convex.png" class="h-50 rounded" />
</div>

<!--
强凸
-->

---

## Strong Convex

可以看到，$\mu$-强凸的定义是在凸函数的基础上，加上了一个模，这个模的系数为 $\mu$，也就是让 $\mu$-强凸的函数的曲率更大。

如果令 $y = x^{*}$，则 $\nabla f(y) = \nabla f(x^{*}) = 0$，则有：


$$
\langle \nabla f(x), x-x^{*}\rangle \geq \frac{\mu}{2}||x-x^{*}||^2
$$


同时，也不难验证，函数 $f$ 是 $\mu$-强凸的当且仅当 $f-\frac{\mu}{2}||\cdot||^2$ 是凸的。

<!--
After all；这里是一个推论
-->

---

## Lipschitz Continuity

$L$-Lipschitz 连续，要求函数图像的曲线上任意两点连线的斜率一致有界，就是任意的斜率都小于同一个常数，这个常数就是 Lipschitz 常数。


考虑实值函数 $f: R^{d}\to R$，和模 $||\cdot||$，如果存在常数 $L>0$，对任意自变量 $x,y\in R^d$，都有下面不等式成立：

$$
|f(x)-f(y)| \leq L||x-y||
$$

则称函数 $f$ 关于模 $||\cdot||$ 是 $L$-Lipschitz 连续的。

<div class="flex flex-wrap justify-center gap-4">
  <img src="/lipschitz.png" class="h-50 rounded" />
</div>

<!--
拉普利斯连续，让梯度有界
-->

---

## Smoothness

对于可导函数，光滑性质依赖于函数的导数，定义如下：

考虑实值函数 $f: R^{d}\to R$，和模 $||\cdot||$，如果存在常数 $L>0$，对任意自变量 $x,y\in R^d$，都有下面不等式成立：

$$
f(x)-f(y) \leq \nabla f(y)^{\top}(x-y)+\frac{L}{2}||x-y||^2
$$

则称函数 $f$ 关于模 $||\cdot||$ 是 $L$-光滑的。

<div class="flex flex-wrap justify-center gap-4">
  <img src="/smoothness.png" class="h-50 rounded" />
</div>

<!--
光滑，让曲率更光滑
-->

---

## Smoothness

另一种形式是：

$$
|\nabla f(x)-\nabla f(y)| \leq L||x-y||
$$

即，凸函数$f$是$L$-光滑的充分必要条件是其导数$\nabla f$是$L$-Lipschitz 连续的。

<!--
拉普利斯连续则是针对 f(x)
-->

---
layout: center
class: text-center
---

## Gradient Decent & Mini-batch Stochastic Gradient Decent

梯度下降、随机梯度下降、小批量随机梯度下降


---

## Gradient Decent

梯度下降公式为：

$$
w_{t+1}=w_{t}-\eta_{t} \nabla f(w_{t})
$$


其中，$\eta_t$ 为学习率，$w_t$ 为参数，$f(w_t)$ 为目标函数，且 $\eta_t \leq \eta_{t-1} \leq \cdots \leq \eta_1$ 为递减序列。

有下面任何情况下都成立的等式：

$$
\begin{align}
\| w_{t+1} - w^* \|^2 &= \| w_{t} - \eta_t \nabla f(w_t) - w^* \|^2 \\
&= \| w_{t} - w^* \|^2 - 2\eta_t \langle \nabla f(w_t), w_t - w^* \rangle + \eta_t^2 \| \nabla f(w_t) \|^2 \\
\end{align}
$$

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_gd_79c039_33_1.svg" class="h-50 rounded" />
</div>

<!--
学习率递减
w^\ast 是最优解，有时会省略二阶小
-->

---

## Gradient Decent

为什么梯度下降可以找到最优解？

考虑 $f(x)$ 在 $x$ 处的泰勒展开（一阶近似）：

$$
f(x+\epsilon)=f(x)+\epsilon f^{\prime}(x)+o(\epsilon^{2})
$$

如果我们选择步长 $\eta > 0$，取 $\epsilon = -\eta f'(x)$，则有：

$$
f(x-\eta f^{\prime}(x))=f(x)-\eta f^{\prime}(x)^{2}+o(\eta^{2} f^{\prime}(x)^{2})
$$

如果 $f^{\prime}(x) \neq 0$，那么我们总可以让 $\eta$ 足够小，使得 $o(\eta^{2} f^{\prime}(x)^{2})$ 可以忽略，那么有：

$$
f(x-\eta f^{\prime}(x))<f(x)
$$

<!--
这里其实就是前面的等式；那么就是梯度下降一直让目标函数严格递减
-->

---

## Gradient Decent (Learning Rate)

学习率的选择：

1. 如果学习率太小，那么梯度下降的收敛速度会很慢，需要更多的迭代次数才能达到最优解。

<br/><br/><br/>

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_gd_79c039_48_1.svg" class="h-60 rounded" />
</div>

<!--
这是一个学习率较小的例子，需要迭代很多次
-->

---

## Gradient Decent (Learning Rate)

<br/>

2. 如果学习率太大，那么梯度下降可能会发散，甚至不收敛。

<br/><br/>

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_gd_79c039_63_1.svg" class="h-60 rounded" />
</div>

<!--
这是一个较大学习率的情况，梯度下降甚至会发散
-->

---

## Gradient Decent (Learning Rate)

<br/>

3. 较高的学习率可能导致较差的局部最优解，也可能获得较好的局部最优解。

~~（所以你可以在跑实验时调一下随机数种子<twemoji-exploding-head />）~~

<br/><br/>

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_gd_79c039_78_1.svg" class="h-60 rounded" />
</div>

<!--
这是一个炼丹的过程 hhh
-->

---

## Stochastic Gradient Decent

由于梯度下降在数据量大的时候，计算复杂度也相对较大，因此有了随机梯度下降，每次只用一个 sample $(\mathbf{x_i}, y_i)$的梯度来估计总体梯度，随机梯度下降公式为：


$$
w_{t+1}=w_{t}-\eta_{t} \nabla f(w_{t}; \xi_{t})
$$


随机梯度经常用 $g(w_t)=\nabla f(w_t; \xi_t)$ 来表示。

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_sgd_baca77_48_1.svg" class="h-60 rounded" />
</div>

<!--
由于是随机选取的一个样本，因此是随机梯度下降
-->

---

## Mini-batch Stochastic Gradient Decent

Mini-batch Stochastic Gradient Descent 是随机梯度下降的一种改进，每次使用 $b$ 个样本来估计总体梯度，即，$g(w_t)=\frac{1}{b}\sum_{i=1}^b \nabla f(w_t; \xi_{t,i})$。Mini-batch Stochastic Gradient Descent 公式为：


$$
w_{t+1}=w_{t}-\frac{\eta_{t}}{b} \sum_{i=1}^{b} \nabla f(w_{t}; \xi_{t,i})
$$

<div class="flex flex-wrap justify-center gap-4">
  <img src="/assets/output_minibatch-sgd_f4d60f_183_0.svg" class="h-60 rounded" />
</div>

<!--
小批量随机梯度下降，同时对一个 batch 计算总体梯度，因此梯度会相对稳定，并且可以并行，也是我们训练时常用的方式
-->

---

## Mini-batch Stochastic Gradient Decent

如果使用小批量随机梯度下降，那么往往会有以下假设：

### Unbiased Estimate

mini-batch SGD 的梯度估计是无偏的，即：

$$
\mathbb{E}_{\xi} [g(w; \xi)] = \nabla f(w)
$$

### Gradient Bounded Variance Assumption

梯度方差有界假设，设一次 mini-batch sample 出来的梯度为 $\nabla f(w; \xi)$，则有下式：

$$
\mathrm {Var}(\nabla f(w; \xi)) \leq \sigma^2
$$

<!--
期望g是利用完整样本梯度的无偏
-->

---

## Mini-batch Stochastic Gradient Decent


由以上两个假设（无偏估计和梯度方差有界），我们有：


$$
\mathrm{Var}(g(w; \xi)) = \mathbb{E}_{\xi}[||g(w; \xi)||^2]- ||\mathbb{E}_{\xi}[g(w; \xi)]||^{2} \leq \frac{\sigma^2}{b}
$$

即，

$$
\mathbb{E}_{\xi}[||g(w; \xi)||^2] \leq ||\nabla f(w)||^{2} + \frac{\sigma^2}{b}
$$

<!--
可以得到一个不等式，方差等于平方的期望-期望的平方，可以得到随机梯度关于梯度的一个不等式
-->

---

### Newton's Method

回顾一下泰勒展开：

$$
f(x+\epsilon)=f(x)+\epsilon \nabla f(x)+\frac{1}{2} \epsilon^{2} \nabla^{2} f(x)+o(\epsilon^{3})
$$

一般 $\mathbf{H} = \nabla^{2} f(x)$ 称为 Hessian 矩阵，它是一个 $n \times n$ 的矩阵，其中第 $i$ 行第 $j$ 列的元素为 $\frac{\partial^{2} f}{\partial x_{i} \partial x_{j}}$。

在最小值往往 $\nabla f(x) = \frac{f(x+\epsilon)-f(x)}{\epsilon}$ 为 0，再对 $\epsilon$ 求导，我们可以直接计算得到：

$$
\nabla f(x) + \mathbf{H} \epsilon = 0 \Rightarrow \epsilon = -\mathbf{H}^{-1} \nabla f(x)
$$

<div class="flex flex-wrap justify-center gap-4">
  <img src="https://zh.d2l.ai/_images/output_gd_79c039_123_1.svg" class="h-50 rounded" />
</div>

<!--
牛顿法，利用二阶梯度，也就是 Hessian 矩阵
-->

---

### Newton's Method

然而 Hessian 矩阵的存储和计算、求逆都较为复杂，因此牛顿法在深度学习中并不常用。

选读：[拟牛顿法](https://zh.wikipedia.org/zh-hans/%E6%93%AC%E7%89%9B%E9%A0%93%E6%B3%95)

<!--
梯度已经是现有梯度的平方级别了，显存消耗和计算复杂度都很高，同时需要求逆，这个又是一个复杂的运算，因此在深度学习中并不常用；但是还有一种拟牛顿法，并不这么复杂，是对二阶梯度的一种估计，有些深度学习方法中可能会用到，这个是选读内容，感兴趣的同学可以进一步了解
-->

---
layout: center
class: text-center
---

## Convergence Analysis

收敛性定义、收敛性分析（凸函数 & GD、mini-batch SGD；非凸函数 & mini-batch SGD）

<!--
下面就是收敛性分析了
-->

---

## Convergence Definition

一个梯度方法是收敛的可以从三个方面来定义：

1. 目标函数值收敛到最优值：$\mathbb{E} f(w_T) - f^{*} \leq \epsilon(T)$，其中 $f^{*}$ 是目标函数的最优值

2. 迭代序列收敛到最优解：$\mathbb{E} \||w_T - w^{*}||^2 \leq \epsilon(T)$，其中 $w^{*}$ 是参数的最优解

如果，$\epsilon(T) \to 0$，那么我们称这个梯度方法是**收敛的**。对于收敛优化算法，它们的收敛速率可能并不相同，通常，用 $\log \epsilon(T)$ 的衰减速率来定义优化算法的收敛速率。

- 1） 如果 $\log \epsilon(T)$ 与 $-T$ 同阶，那么我们称这个算法具有**线性收敛速率**。
- 2） 如果 $\log \epsilon(T)$ 比 $-T$ 衰减速度慢，那么我们称这个算法具有**次线性收敛速率**。
- 3） 如果 $\log \epsilon(T)$ 比 $-T$ 衰减速度快，那么我们称这个算法具有**超线性收敛速率**，进一步地，如果 $\log \log \epsilon(T)$ 与 $-T$ 同阶，那么我们称这个算法具有**二阶收敛速率**。

而在非凸优化中，由于可能存在多个局部极小点，不容易找到全局最优，因此考虑算法能否收敛到梯度为 0 的临界点。

3. 利用梯度的遍历距离作为度量：$\min_{t=1,\cdots,T} \mathbb{E} ||\nabla f(w_T)||^2 \leq \epsilon(T)$ 或者 $\frac{1}{T}\sum_{t=1}^{T}\mathbb{E}||\nabla f(w_{t})||^2$ 趋于 0

<!--
首先我们要知道什么是收敛
-->

---

## Convergence Analysis ($\mu$-strongly convex and $L$-smooth & GD)

假设目标函数 $f$ 是 $R^{d}$ 上的$\mu$-强凸函数，并且 $L$-光滑，当步长 $\eta  \leq \frac{1}{L}$ 且起始点为 $w_0$ 时，经过 $t$ 步迭代，$f(w_T)$ 被 bounded 如下：

$$
f\left(w_{t}\right)-f^{*} \leq (1-\eta \mu)^{t}\left(f\left(w_{0}\right)-f^{*}\right)
$$

_Proof_:

先看单步迭代：

$$
\begin{aligned}
f\left(w_{t+1}\right)-f\left(w_{t}\right) &=f\left(w_{t}-\eta  \nabla f\left(w_{t}\right)\right)-f\left(w_{t}\right) \\
& \leq \nabla f\left(w_{t}\right)^{\top}\left(w_{t}-\eta  \nabla f\left(w_{t}\right)-w_{t}\right)+\frac{L}{2}\left\|w_{t}-\eta  \nabla f\left(w_{t}\right)-w_{t}\right\|^{2} \\
&=-\eta \left\|\nabla f\left(w_{t}\right)\right\|^{2}+\frac{L}{2} \eta ^{2}\left\|\nabla f\left(w_{t}\right)\right\|^{2} \\
&=\left(\frac{L}{2} \eta ^{2}-\eta \right)\left\|\nabla f\left(w_{t}\right)\right\|^{2}
\end{aligned}
$$

<!--
我们这里讲一个最强假设的收敛性（FL 较早的收敛性分析中也有很多利用了强凸和光滑假设）；首先我们先给出结论；也是第一种收敛性的定义；这里我们直接代入梯度下降公式，代入L-光滑的定义的公式，稍微整理下就可以得到下面的这个式子，可以观察到这里eta是开口向上的二次函数
-->

---

这时候我们通过对两项都添加负号，然后利用二次函数的顶点来确定 $\eta$ 的取值，同时希望用上强凸函数导出的性质，即


$$
2\mu (f(x)-f^{*}) \leq \|\nabla f(x)\|^2
$$

于是有

$$
\begin{aligned}
f\left(w_{t+1}\right)-f\left(w_{t}\right) &\leq \eta \left(1-\frac{L}{2} \eta \right)\left(-\left\|\nabla f\left(w_{t}\right)\right\|^{2}\right) \\
& \leq \eta \left(1-\frac{L}{2} \eta \right)\left(2\mu \left(f\left(w_{t}\right)-f^{*}\right)\right)
\end{aligned}
$$

这时，我们令 $\eta \leq \frac{1}{L}$，则有 $(1-\frac{L}{2} \eta ) \geq \frac{1}{2}$，于是有

$$
f\left(w_{t+1}\right)-f\left(w_{t}\right) \leq -\eta \mu (f\left(w_{t}\right)-f^{*})
$$

注意到，左右都是关于 $f(x)$ 的式子，在左边 $+-f^*$，得到

$$
f\left(w_{t+1}\right)-f^{*}+f^{*}-f\left(w_{t}\right) \leq -\eta \mu (f\left(w_{t}\right)-f^{*}) \\
\Rightarrow f\left(w_{t+1}\right)-f^{*} \leq (1-\eta \mu) (f\left(w_{t}\right)-f^{*})
$$

<!--
这里我们先利用PL（Polyak-Lojasiewicz）不等式推出的一个结论；直接代入即可，一个向下开口的二次函数，很容易看出在对称轴取得最小值，因为我们能改变的只有eta，然后进行一个小学二年级学的等比数列的配方就可以了
-->

---

后面就归纳一下：

$$
\begin{aligned}
f\left(w_{t+1}\right)-f^{*} & \leq(1-\eta  \mu)\left(f\left(w_{t}\right)-f^{*}\right) \\
& \leq(1-\eta  \mu)\left(1-\eta \mu\right)\left(f\left(w_{t-1}\right)-f^{*}\right) \\
& \leq \cdots \\
& \leq(1-\eta  \mu) \cdots(1-\eta \mu)\left(f\left(w_{1}\right)-f^{*}\right) \\
& \leq(1-\eta  \mu)^{t+1}\left(f\left(w_{0}\right)-f^{*}\right)

\end{aligned}
$$

现在，我们得到了 boundary，来计算一下复杂度。

根据复杂度的定义，令 $f(w_t)-f^* \leq \epsilon$，则有

$$
(1-\eta \mu)^{t}(f(w_0)-f^*) \leq \epsilon \\
\Rightarrow  t\log (1-\eta \mu) + \log (f(w_0)-f^*) \leq \log \epsilon \\
\Rightarrow  t\log \frac{1}{1-\eta \mu} - \log (f(w_0)-f^*) \geq \log \frac{1}{\epsilon} \\
\Rightarrow  t=O(\log \frac{1}{\epsilon})
$$

因此在这些假设下的 GD 有线性的收敛速率。

<!--
这里求完以后可以将t+1替换成t
-->

---

## Convergence Analysis ($\mu$-strongly convex and $L$-smooth & mini-batch SGD)

假设目标函数 $f$ 是 $R^{d}$ 上的$\mu$-强凸函数，并且 $L$-光滑，同时满足梯度无偏估计和梯度方差有界假设，当步长 $\eta  \leq \frac{1}{L}$ 且起始点为 $w_0$ 时，经过 $t$ 步 mini-batch SGD 迭代，$\mathbb{E}[f(w_{t})]$ 被 bounded 如下：

$$
\mathbb{E}[f(w_{t})] - f(w^{*}) - \frac{\eta L \sigma^{2}}{2\mu b} \leq (1- \eta \mu)^{t}\left (\mathbb{E}[f(w_{0})] - f(w^{*}) - \frac{\eta L \sigma^{2}}{2\mu b} \right )
$$

_Proof_:

由光滑性：

$$
\begin{align}
f(w_{t+1}) - f(w_{t})&\leq \nabla f(w_{t})^{\top}(w_{t+1} - w_{t}) + \frac{L}{2}||w_{t+1} - w_{t}||^{2} \\
&\leq -\eta \nabla f(w_{t})^{\top}g(w_{t}; \xi_{t}) + \frac{L}{2}||-\eta g(w_{t}; \xi_{t})||^{2} \\
&\leq -\eta \nabla f(w_{t})^{\top}g(w_{t}; \xi_{t}) + \frac{L\eta^{2}}{2} ||g(w_{t}; \xi_{t})||^{2} \\
\end{align}
$$

<!--
我们这里还有一个小批量梯度下降的证明，差不多的思路，只不过中间会对两边求一下期望，得到相似的结论，由于时间原因这里就不展开了，感兴趣的同学了阅读后面的参考内容了解一下
-->

---

对左右求期望得：

$$
\begin{align}
\mathbb{E}[f(w_{t+1}) - f(w_{t})] &\leq -\eta \mathbb{E}[\nabla f(w_{t})^{\top}g(w_{t}; \xi_{t})] + \frac{L\eta^{2}}{2} \mathbb{E}[||g(w_{t}; \xi_{t})||^{2}] \\
\mathbb{E}[f(w_{t+1})] - f(w_{t}) &\leq -\eta ||\nabla f(w_{t})||^{2} + \frac{L\eta^{2}}{2}||\nabla f(w_{t})||^{2} + \frac{L\eta^{2} \sigma^{2}}{2b} \\
&\leq (\eta-\frac{L\eta^{2}}{2}) (-||\nabla f(w_{t})||^{2})  + \frac{L\eta^{2} \sigma^{2}}{2b}
\end{align}
$$

这里和 GD 就比较像了，取 $\eta<\frac{1}{L}$，在利用强凸性则有：

$$
\begin{align}
\mathbb{E}[f(w_{t+1})] - f(w_{t}) &\leq \frac{\eta}{2}(-||\nabla f(w_{t})||^{2}) + \frac{L\eta^{2} \sigma^{2}}{2b} \\
&\leq \frac{\eta}{2}(-2\mu(f(w_{t})-f^{*})) + \frac{L\eta^{2} \sigma^{2}}{2b}  \\
& \leq -\eta\mu(\mathbb{E}[f(w_{t})]-f^{*}) + \frac{L\eta^{2} \sigma^{2}}{2b}  \\
\end{align}
$$


---

左右都 $+f^{*}-f^{*}$，得到

$$
\mathbb{E}[f(w_{t+1})] - f^{*} \leq (1-\eta\mu)(\mathbb{E}[f(w_{t})]-f^{*}) + \frac{L\eta^{2} \sigma^{2}}{2b} 
$$

接下来就需要配方了，我们还是假设一个未知数 $x$，使得

$$
\mathbb{E}[f(w_{t+1})] - f^{*} + x \leq (1-\eta\mu)(\mathbb{E}[f(w_{t})]-f^{*} + x)
$$

也就是

$$
(1-\eta\mu)x - x = \frac{L\eta^{2} \sigma^{2}}{2b} \\
x = - \frac{L\eta \sigma^{2}}{2\mu b}
$$

即

$$
\mathbb{E}[f(w_{t+1})] - f^{*} - \frac{L\eta \sigma^{2}}{2\mu b} \leq (1-\eta\mu)(\mathbb{E}[f(w_{t})]-f^{*} - \frac{L\eta \sigma^{2}}{2\mu b})
$$


---

得到

$$
\mathbb{E}[f(w_{t})] - f(w^{*}) - \frac{\eta L \sigma^{2}}{2\mu b} \leq (1- \eta \mu)^{t}\left (\mathbb{E}[f(w_{0})] - f(w^{*}) - \frac{\eta L \sigma^{2}}{2\mu b} \right )
$$

证毕。

可以看出随机梯度下降的收敛速率同梯度下降的收敛速率相同，都是线性的，但是需要更多的迭代次数。


---

## Convergence Analysis (non-convex & mini-batch SGD)

假设目标函数 $f$ 是 $R^{d}$ 上的 $L$-光滑函数，并且满足梯度无偏估计和梯度方差有界假设，当步长 $\eta  \leq \frac{1}{L}$ 且起始点为 $w_0$ 时，经过 $t$ 步 mini-batch SGD 迭代，$f$ 梯度平方的平均期望被 bounded 如下：

$$
\mathbb{E}[\frac{1}{t}\sum_{i=1}^{t} ||\nabla f(w_{i})||^{2}] \leq \frac{L \sigma^{2}}{b} + \frac{2(f(w_{0})-f(w_{\inf}))}{t\eta}
$$

_Proof_:

直接从上面的 $L$-光滑性得到的结论，和 $\eta < \frac{1}{L}$ 开始

$$
\mathbb{E}[f(w_{t+1})] - f(w_{t}) \leq -\eta ||\nabla f(w_{t})||^{2} + \frac{L\eta^{2} \sigma^{2}}{2b}
$$

<!--
这里是进一步放缩条件，放成非凸的条件进行小批量梯度下降的证明，非凸的时候前两种收敛条件都已经不适用了，我们只能证明累积的梯度是被bound住的来证明收敛性
-->

---

对两边都求和，在除以 $t$，得到

$$
\begin{align}
\frac{1}{t}\sum_{i=1}^{t} \mathbb{E}[f(w_{i+1})] - f(w_{i}) &\leq -\frac{\eta }{2t}\sum_{i=1}^{t} ||\nabla f(w_{i})||^{2} + \frac{L\eta^{2} \sigma^{2}}{2b} \\
\frac{1}{t} \sum_{i=1}^{t} ||\nabla f(w_{i})||^{2} &\leq -\frac{2\mathbb{E}[f(w_{t+1})-f(w_{0})]}{\eta t} + \frac{L\eta \sigma^{2}}{2b} \\
&\leq \frac{2(f(w_{0})-f(w_{\inf}))}{\eta t} + \frac{L\eta \sigma^{2}}{2b}
\end{align}
$$

证毕。

复杂度 $t=O(\frac{1}{\epsilon})$，这个收敛速率是次线性的。

<!--
这里就已经得不到次线性的收敛条件了
-->

---
layout: center
class: text-center
---

## Other Convergence Analysis


---

## Other Convergence Analysis

| Condition | GD | SGD |
| --- | --- | --- |
| Convex | $O(\frac{1}{\sqrt{T}})$ | $O(\frac{1}{\sqrt{T}})$ |
| + Lipschitz | $O(\frac{1}{T})$ | $O(\frac{1}{\sqrt{T}})$ |
| + Strongly Convex | $O(c^{T})$ | $O(\frac{1}{T})$ |

> 10-725/36-725: Convex Optimization(Fall 2018), Lecture 24: November 26, Ryan Tibshirani

### Distributed Optimization

- Distributed Synchrounous SGD
- Distributed Asynchrounous SGD
- Federated Learning


---

## Reference

1. 《Convex Optimization》, Stephen Boyd
2. 《Optimization Algorithm for Distributed Machine Learning》, Gauri Joshi
3. 《Dive into Deep Learning》, D2L.ai
4. 《Optimization Methods for Large-Scale Machine Learning》, Léon Bottou

<br>

Learning more about optimization and convergence analysis:

  [https://blog.bj-yan.top/tags/convergence-analysis/](https://blog.bj-yan.top/tags/convergence-analysis/)
