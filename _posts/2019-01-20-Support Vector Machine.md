---
title: '机器学习| 支持向量机详解 (Python 语言描述)'
layout: post
tags:
  - Machine Learning
  - Support Vector Machine
  - Python
category: 
  - Machine Learning
  - Support Vector Machine
  - Python
---
在前面的文章中，我们对线性分布和非线性分布的数据处理方法进行了简单的介绍和实际的实验操作，当前还有一种机器学习方法，它在解决小样本、非线性及高维模式识别中都表现出了许多独特的优势，在样本量较小的情况，其实际运用效果甚至超过了神经网络，并且其不仅可以应用于线性分布数据，还可以用于非线性分布数据，相比于其他基本机器学习分类算法如逻辑回归、KNN、朴素贝叶斯等，其最终效果的表现一般都会优于这些方法。

## 线性分类支持向量机

在逻辑回归中，我们尝试通过一条直线针对线性可分数据完成分类。同时，通过最小化对数损失函数来找到最优分割边界，也就是下图中的紫色直线。

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/1.png)

逻辑回归是一种简单高效的线性分类方法。而在这里中，我们将接触到另一种针对线性可分数据进行分类的思路，并把这种方法称之为支持向量机（英语：Support vector machine，简称：SVM）。

如果你第一次接触支持向量机这个名字，可能会感觉读起来比较拗口。至少我当时初次接触支持向量机时，完全不知道为什么会有这样一个怪异的名字。假如你和当时的我一样，那么当你看完下面这段介绍内容后，就应该会对支持向量机这个名词有更深刻的认识了。

<!--more-->

### 支持向量机分类特点

假设给定一个训练数据集 $T=\lbrace(x_1,y_1),(x_2,y_2),\cdots ,(x_n,y_n)\rbrace$ 。同时，假定已经找到样本空间中的分割平面，其划分公式可以通过以下线性方程来描述：
$$
wx+b=0\tag{1}
$$
使用一条直线对线性可分数据集进行分类的过程中，我们已经知道这样的直线可能有很多条：

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/4.png)

问题来了！**哪一条直线是最优的划分方法呢？**

在逻辑回归中，我们引入了 S 形曲线和对数损失函数进行优化求解。如今，支持向量机给了一种从几何学上更加直观的方法进行求解，如下图所示：

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/2.png)

上图展示了支持向量机分类的过程。图中 $wx-b=0$ 为分割直线，我们通过这条直线将数据点分开。与此同时，分割时会在直线的两边再设立两个互相平行的虚线，这两条虚线与分割直线的距离一致。这里的距离往往也被我们称之为「间隔」，而支持向量机的分割特点在于，要使得**分割直线和虚线之间的间隔最大化**。同时也就是两虚线之间的间隔最大化。

对于线性可分的正负样本点而言，位于 $wx-b=1$ 虚线外的点就是正样本点，而位于 $wx-b=-1$ 虚线外的点就是负样本点。另外，正好位于两条虚线上方的样本点就被我们称为支持向量，这也就是支持向量机的名字来源。

### 支持向量机分类演示

下面，我们使用 Python 代码来演示支持向量机的分类过程。

首先，我们介绍一种新的示例数据生成方法。即通过 scikit-learn 提供的 `samples_generator()` 类完成。通过 `samples_generator()` 类下面提供的不同方法，可以产生不同分布状态的示例数据。首先要用到 `make_blobs` 方法，该方法可以生成团状数据。

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
from sklearn.datasets import samples_generator

x, y = samples_generator.make_blobs(n_samples=60, centers=2, random_state=30, cluster_std=0.8) # 生成示例数据

plt.figure(figsize=(10, 8)) # 绘图
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_00.png)

接下来，我们在示例数据中绘制任意 3 条分割线把示例数据分开。

```python
plt.figure(figsize=(10, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')

# 绘制 3 条不同的分割线
x_temp = np.linspace(0, 6)
for m, b in [(1, -8), (0.5, -6.5), (-0.2, -4.25)]:
    y_temp = m * x_temp + b
    plt.plot(x_temp, y_temp, '-k')
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_01.png)

然后，可以使用 `fill_between` 方法手动绘制出分类硬间隔。

```python
plt.figure(figsize=(10, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')

# 绘制 3 条不同的分割线
x_temp = np.linspace(0, 6)
for m, b, d in [(1, -8, 0.2), (0.5, -6.5, 0.55), (-0.2, -4.25, 0.75)]:
    y_temp = m * x_temp + b
    plt.plot(x_temp, y_temp, '-k')
    plt.fill_between(x_temp, y_temp - d, y_temp + d, color='#f3e17d', alpha=0.5)
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_02.png)

<div style="color: #999;font-size: 12px;font-style: italic;"><strong>上图为了呈现出分类间隔的效果，手动指定了参数。</strong></div>

可以看出，不同的分割线所对应的间隔大小是不一致的，而支持向量机的目标是找到最大的分类硬间隔所对应的分割线。

### 硬间隔表示及求解

我们已经知道支持向量机是根据最大间隔来划分，下面考虑如何求得一个几何间隔最大的分割线。

对于线性可分数据而言，几何间隔最大的分离超平面是唯一的，这里的间隔也被我们称之为「硬间隔」，而间隔最大化也就称为硬间隔最大化。上图实际上就是硬间隔的典型例子。

最大间隔分离超平面，我们希望最大化超平面 $(w,b)$ 关于训练数据集的几何间隔 $\gamma$，满足以下约束条件：每个训练样本点到超平面 $(w,b)$ 的几何间隔至少都是 $\gamma$ ，因此可以转化为以下的约束最优化问题： 

$$
\max\limits_{w,b}\gamma =\frac{2}{\left \|w\right \|} \tag{2a}
$$

$$
\begin{equation}
\textrm s.t. y_i(\frac{w}{\left \|w\right \|}x_i+\frac{b}{\left \|w\right \|})\geq \frac{\gamma}{2} \tag{2b}
\end{equation}
$$

实际上，$\gamma$ 的取值并不会影响最优化问题的解，同时，我们根据数学对偶性原则，可以得到面向硬间隔的线性可分数据的支持向量机的最优化问题：
$$
\min\limits_{w,b}\frac{1}{2}\left \|w\right \|^2 \tag{3a}
$$

$$
\begin{equation}
\textrm s.t. y_i(wx_i+b)-1\geq 0\tag{3b}
\end{equation}
$$

我们通常使用拉格朗日乘子法来求解最优化问题，将原始问题转化为对偶问题，通过解对偶问题得到原始问题的解。对公式（3）使用拉格朗日乘子法可得到其「对偶问题」。具体来说，对每条约束添加拉格朗日乘子 $\alpha_i \geq 0$，则该问题的拉格朗日函数可写为：
$$
L(w,b,\alpha)=\frac{1}{2}\left \| w\right \|^2+\sum\limits_{i=1}^{m}\alpha_i(1-y_i(wx_i+b)) \tag{4}
$$
我们通过将公式（4）分别对 $w$ 和 $b$ 求偏导为 `0` 并代入原式中，可以将 $w$ 和 $b$ 消去，得到公式（3）的对偶问题：
$$
\max\limits_{\alpha} \sum\limits_{i=1}^{N}\alpha_i-\frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N}\alpha_i \alpha_j y_i y_j x_i x_j \tag{5a}
$$

$$
s.t. \sum\limits_{i=1}^{N}\alpha_i y_i=0,\tag{5b}
$$

$$
\alpha_i \geq 0,i=1,2,\cdots,N  \tag{5c}
$$

解出最优解 $\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)$ 后，基于此我们可以求得最优解 $w^*$, $b^*$，由此得到分离超平面：
$$
w^*x+b^*=0  \tag{6}
$$
使用符号函数求得正负类之间的分类决策函数为：
$$
f(x)=sign(w^*x+b^*) \tag{7}
$$

### 软间隔表示及求解

上面，我们介绍了线性可分条件下的最大硬间隔的推导求解方法。在很多时候，我们还会遇到下面这种情况。你可以发现，在实心点和空心点中各混入了零星的不同类别的数据点。对于这种情况，数据集就变成了严格意义上的线性不可分。但是，造成这种线性不可分的原因往往是因为包含「噪声」数据，它同样可以被看作是不严格条件下的线性可分。

当我们使用支持向量机求解这类问题时，就会把最大间隔称之为最大「软间隔」，而软间隔就意味着可以容许零星噪声数据被误分类。

<img width='300px' style="border:2px dashed #000000;" src="https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/3.png"></img>

当出现上图所示的样本点不是严格线性可分的情况时，某些样本点 $(x_i,y_i)$ 就不能满足函数间隔 $\geqslant 1$ 的约束条件，即公式（3b）中的约束条件。为了解决这个问题，可以对每个样本点 $(x_i,y_i)$ 引入一个松弛变量 $\xi_i \geq 0$，使得函数间隔加上松弛变量 $\geqslant 1$，即约束条件转化为：
$$
y_i(wx_i+b) \geq 1-\xi_i \tag{8}
$$
同时，对每个松弛变量 $\xi_i$ 支付一个代价 $\xi_i$，目标函数由原来的 $\frac{1}{2}||w||^2$ 变成：
$$
\frac{1}{2}\left \| w \right \|^2+C\sum\limits_{j=1}^{N}\xi_i \tag{9}
$$
这里，$C>0$ 称为惩罚参数，一般根据实际情况确定。$C$ 值越大对误分类的惩罚增大，最优化问题即为：
$$
\min\limits_{w,b,\xi} \frac{1}{2}\left \| w \right \|^2+C\sum\limits_{i=1}^{N}\xi_i \tag{10a}
$$

$$
s.t. y_i(wx_i+b) \geq 1-\xi_i,i=1,2,...,N \tag{10b}
$$

$$
\xi_i\geq 0,i=1,2,...,N \tag{10c}
$$

这就是软间隔支持向量机的表示过程。同理，我们可以使用拉格朗日乘子法将其转换为对偶问题求解：
$$
\max\limits_{\alpha}  \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N}\alpha_i\alpha_jy_iy_j(x_i*x_j)-\sum\limits_{i=1}^{N}\alpha_i \tag{11a}
$$

$$
s.t. \sum\limits_{i=1}^{N}\alpha_iy_i=0 \tag{11b}
$$

$$
0 \leq \alpha_i \leq C ,i=1,2,...,N\tag{11c}
$$

解出最优解 $\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)$ 后，基于此我们可以求得最优解 $w^*$, $b^*$，由此得到分离超平面：
$$
w^*x+b^*=0  \tag{12}
$$
使用符号函数求得正负类之间的分类决策函数为：
$$
f(x)=sign(w^*x+b^*) \tag{13}
$$

### 线性支持向量机分类实现

上面，我们对硬间隔和软间隔支持向量机的求解过程进行了推演，推导过程比较复杂不需要完全掌握，但至少要知道硬间隔和软间隔区别。接下来，我们就使用 Python 对支持向量机找寻最大间隔的过程进行实战。由于支持向量机纯 Python 实现太过复杂，所以本次直接使用 scikit-learn 完成。

scikit-learn 中的支持向量机分类器对应的类及参数为：

```python
sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
```

主要的参数如下：

- `C`: 软间隔支持向量机对应的惩罚参数，详见公式（9）.
- `kernel`: 核函数，linear, poly, rbf, sigmoid, precomputed 可选，下文详细介绍。
- `degree`: poly 多项式核函数的指数。
- `tol`: 收敛停止的容许值。

这里，我们还是使用上面生成的示例数据训练支持向量机模型。由于是线性可分数据，`kernel` 参数指定为 `linear` 即可。

首先，训练支持向量机线性分类模型：

```python
from sklearn.svm import SVC

linear_svc = SVC(kernel='linear')
linear_svc.fit(x, y)
```

```python
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
```

对于训练完成的模型，我们可以通过 `support_vectors_` 属性输出它对应的支持向量：

```python
linear_svc.support_vectors_
```

```python
array([[ 2.57325754, -3.92687452],
       [ 2.49156506, -5.96321164],
       [ 4.62473719, -6.02504452]])
```

可以看到，一共有 `3` 个支持向量。如果你输出 `x, y` 的坐标值，就能看到这 `3` 个支持向量所对应的数据。

接下来，我们可以使用 Matplotlib 绘制出训练完成的支持向量机对于的分割线和间隔。为了方便后文重复使用，这里将绘图操作写入到 `svc_plot()` 函数中：

```python
def svc_plot(model):
    
    # 获取到当前 Axes 子图数据，并为绘制分割线做准备
    ax = plt.gca()
    x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
    y = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # 使用轮廓线方法绘制分割线
    ax.contour(X, Y, P, colors='green', levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    
    # 标记出支持向量的位置
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], c='green', s=100)
```

```python
# 绘制最大间隔支持向量图
plt.figure(figsize=(10, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')
svc_plot(linear_svc)
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_03.png)

如上图所示，绿色实线代表最终找到的分割线，绿色虚线之间的间隔也就是最大间隔。同时，绿色实心点即代表 `3` 个支持向量的位置。

上面的数据点可以被线性可分，所以得到的也就是硬间隔支持向量机的分类结果。那么，如果我们加入噪声使得数据集变成不完美线性可分，结果会怎么样呢？

接下来，我们就来还原软间隔支持向量机的分类过程：

```python
# 向原数据集中加入噪声点
x = np.concatenate((x, np.array([[3, -4], [4, -3.8], [2.5, -6.3], [3.3, -5.8]])))
y = np.concatenate((y, np.array([1, 1, 0, 0])))

plt.figure(figsize=(10, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_04.png)

可以看到，此时的红蓝数据团中各混入了两个噪声点。

训练软间隔支持向量机模型并绘制成分割线和最大间隔：

```python
linear_svc.fit(x, y) # 训练

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')
svc_plot(linear_svc)
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_05.png)

由于噪声点的混入，此时支持向量的数量由原来的 `3` 个变成了 `13` 个。

前面的实验中，我们提到了惩罚系数 $C$，下面可以通过更改 $C$ 的取值来观察支持向量的变化过程。与此同时，我们要引入一个可以在 Notebook 中实现交互操作的模块。你可以通过选择不同的 $C$ 查看最终绘图的效果。

```python
from ipywidgets import interact
import ipywidgets as widgets

def change_c(c):
    linear_svc.C = c
    linear_svc.fit(x, y)
    plt.figure(figsize=(10, 8))
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap='bwr')
    svc_plot(linear_svc)
    
interact(change_c, c=[1, 10000, 1000000])
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_06.png)

## 非线性分类支持向量机

上面的内容中，我们假设样本是线性可分或不严格线性可分，然后通过支持向量机建立最大硬间隔或软间隔实现样本分类。然而，线性可分的样本往往只是理想情况，现实中的原始样本大多数情况下是线性不可分。此时，还能用支持向量机吗？

其实，对于线性不可分的数据集，我们也可以通过支持向量机去完成分类。但是，这里需要增加一个技巧把线性不可分数据转换为线性可分数据之后，再完成分类。

与此同时，**我们把这种数据转换的技巧称作「核技巧」，实现数据转换的函数称之为「核函数」**。

### 核技巧与核函数

根据上面的介绍，我们提到一个思路就是核技巧，即先把线性不可分数据转换为线性可分数据，然后再使用支持向量机去完成分类。那么，具体是怎样操作呢？

<div style="text-align:center;color:blue;"><i>核技巧的关键在于空间映射，即将低维数据映射到高维空间中，使得数据集在高维空间能被线性可分。</i></div>

<div style="color: #999;font-size: 12px;font-style: italic;">* 核技巧是一种数学方法，仅针对于其在支持向量机中的应用场景进行讲解。</div>

<img width='500px' style="border:2px dashed #000000;" src="https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/5.png"></img>

如上图所示，假设我们在二维空间中有蓝色和红色代表的两类数据点，很明显无法使用一条直线把这两类数据分开。此时，如果我们使用核技巧将其映射到三维空间中，就变成了可以被平面线性可分的状态。

对于「映射」过程，我们还可以这样理解：分布在二维桌面上的红蓝小球无法被线性分开，此时将手掌拍向桌面（好疼），小球在力的作用下跳跃到三维空间中，这也就是一个直观的映射过程。

同时，「映射」的过程也就是通过核函数转换的过程。这里需要补充说明一点，那就是将数据点从低维度空间转换到高维度空间的方法有很多，但往往涉及到庞大的计算量，而数学家们从中发现了几种特殊的函数，这类函数能大大降低计算的复杂度，于是被命名为「核函数」。也就是说，核技巧是一种特殊的「映射」技巧，而核函数是核技巧的实现方法。

下面，我们就认识几种常见的核函数：

#### 线性核函数

$$
k\left ( x_i, x_j \right )=x_i*x_j \tag{14}
$$

#### 多项式核函数

$$
k\left ( x_i, x_j \right )=\left ( x_i*x_j \right )^d, d \geq 1 \tag{15}
$$

#### 高斯径向基核函数

$$
k\left ( x_i, x_j \right ) = \exp \left(-{\frac  {\left \|{\mathbf  {x_i}}-{\mathbf  {x_j}}\right \|_{2}^{2}}{2\sigma ^{2}}}\right)=exp\left ( -\gamma * \left \| x_i-x_j \right \|_{2} ^2 \right ), \gamma>0 \tag{16}
$$

#### Sigmoid 核函数

$$
k\left ( x_i, x_j \right )=tanh\left ( \beta * x_ix_j+\theta \right ), \beta > 0 , \theta < 0 \tag{17}
$$

这 `4` 个核函数也就分别对应着上文介绍 `sklearn` 中 `SVC` 方法中 `kernel` 参数的 `linear, poly, rbf, sigmoid` 等 `4` 种不同取值。

此外，核函数还可以通过函数组合得到，例如：

若 $k_1$ 和 $k_2$ 是核函数，那么对于任意正数 $\lambda_1,\lambda_2$，其线性组合：
$$
\lambda_1 k_1+\lambda_2 k_2 \tag{18}
$$

### 引入核函数的间隔表示及求解

我们通过直接引入核函数 $k(x_i,x_j)$，而不需要显式的定义高维特征空间和映射函数，就可以利用解线性分类问题的方法来求解非线性分类问题的支持向量机。引入核函数以后，对偶问题就变为：
$$
\max\limits_{\alpha}  \frac{1}{2}\sum\limits_{i=1}^{N}\sum\limits_{j=1}^{N}\alpha_i\alpha_jy_iy_jk(x_i*x_j)-\sum\limits_{i=1}^{N}\alpha_i \tag{19a}
$$

$$
s.t. \sum\limits_{i=1}^{N}\alpha_iy_i=0 \tag{19b}
$$

$$
0 \leq \alpha_i \leq C ,i=1,2,...,N \tag{19c}
$$

同样，解出最优解 $\alpha^*=(\alpha_1^*,\alpha_2^*,...,\alpha_N^*)$ 后，基于此我们可以求得最优解 $w^*$, $b^*$，由此得到分离超平面：
$$
w^*x+b^*=0  \tag{20}
$$
使用符号函数求得正负类之间的分类决策函数为：
$$
f(x)=sign(w^*x+b^*) \tag{21}
$$

### 非线性支持向量机分类实现

同样，我们使用 scikit-learn 中提供的 SVC 类来构建非线性支持向量机模型，并绘制决策边界。

首先，实验需要生成一组示例数据。上面我们使用了 `make_blobs` 生成一组线性可分数据，这里使用 `make_circles` 生成一组线性不可分数据。

```python
x2, y2 = samples_generator.make_circles(150, factor=.5, noise=.1, random_state=30) # 生成示例数据

plt.figure(figsize=(8, 8)) # 绘图
plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=40, cmap='bwr')
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_07.png)

上图明显是一组线性不可分数据，当我们训练支持向量机模型时就需要引入核技巧。例如，我们这里使用下式做一个简单的非线性映射：
$$
k\left ( x_i, x_j \right )=x_i^2 + x_j^2 \tag{22}
$$

```python
def kernel_function(xi, xj):
    poly = xi**2 + xj**2
    return poly
```

```python
from mpl_toolkits import mplot3d
from ipywidgets import interact, fixed

r = kernel_function(x2[:,0], x2[:,1])
plt.figure(figsize=(10, 8))
ax = plt.subplot(projection='3d')
ax.scatter3D(x2[:, 0], x2[:, 1], r, c=y2, s=40, cmap='bwr')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_08.png)

上面展示了二维空间点映射到效果维空间的效果。接下来，我们使用 sklearn 中 SVC 方法提供的 RBF 高斯径向基核函数完成实验。

```python
rbf_svc = SVC(kernel='rbf')
rbf_svc.fit(x2, y2)
```

```python
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
  kernel='rbf', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False)
```

```python
plt.figure(figsize=(8, 8))
plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=40, cmap='bwr')

svc_plot(rbf_svc)
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_09.png)

同样，我们可以挑战不同的惩罚系数 $C$，看一看决策边界和支持向量的变化情况：

```python
def change_c(c):
    rbf_svc.C = c
    rbf_svc.fit(x2, y2)
    plt.figure(figsize=(8, 8))
    plt.scatter(x2[:, 0], x2[:, 1], c=y2, s=40, cmap='bwr')
    svc_plot(rbf_svc)
    
interact(change_c, c=[1, 100, 10000])
```

![](https://laugh12321-1258080753.cos.ap-chengdu.myqcloud.com/laugh's%20blog/images/Support%20vector%20machine/output_10.png)

## 多分类支持向量机

支持向量机最初是为二分类问题设计的，当我们面对多分类问题时，其实同样可以使用支持向量机解决。而解决的方法就是通过组合多个二分类器来实现多分类器的构造。根据构造的方式又分为 2 种方法：

- **一对多法**：即训练时依次把某个类别的样本归为一类，剩余的样本归为另一类，这样 $k$ 个类别的样本就构造出了 $k$ 个支持向量机。
- **一对一法**：即在任意两类样本之间构造一个支持向量机，因此 $k$ 个类别的样本就需要设计 $k(k-1) \div 2$ 个支持向量机。

而在 scikit-learn，实现多分类支持向量机通过设定参数 `decision_function_shape` 来确定，其中：

- `decision_function_shape='ovo'`：代表一对一法。
- `decision_function_shape='ovr'`：代表一对多法。

由于这里只需要修改参数，所以就不再赘述了。

---

**拓展阅读：**

- [支持向量机 - 维基百科](https://zh.wikipedia.org/zh-hans/支持向量机)

- [知乎上关于支持向量机的问题讨论](https://www.zhihu.com/question/21094489

- [机器学习实战教程（八）：支持向量机原理篇之手撕线性SVM](https://cuijiahua.com/blog/2017/11/ml_8_svm_1.html)

