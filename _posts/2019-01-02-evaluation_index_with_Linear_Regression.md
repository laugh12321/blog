---
title: '机器学习|线性回归三大评价指标实现『MAE, MSE, MAPE』（Python语言描述）'
layout: post
tags:
  - Machine Learning
  - Linear Regression
  - Python
  - MAE
  - MSE
  - MAPE
category: 
  - Machine Learning
  - Linear Regression
  - Python

---

对于回归预测结果，通常会有平均绝对误差、平均绝对百分比误差、均方误差等多个指标进行评价。这里，我们先介绍最常用的3个：

**平均绝对误差（MAE）**
就是绝对误差的平均值，它的计算公式如下：
$$
MAE(y,\hat{y}) = \frac{1}{n}(\sum_{i = 1}^{n}\left | y - \hat{y} \right |)
$$
其中，$y_{i}$ 表示真实值，$\hat y_{i}$ 表示预测值，$n$ 则表示值的个数。MAE 的值越小，说明预测模型拥有更好的精确度。<!--more-->我们可以尝试使用 Python 实现 MAE 计算函数：

```python
import numpy as np

def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    
    返回:
    mae -- MAE 评价指标
    """
    
    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred))/n
    return mae
```
**均方误差（MSE）**
它表示误差的平方的期望值，它的计算公式如下：
$$
{MSE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y})^{2}
$$

其中，$y_{i}$ 表示真实值，$\hat y_{i}$ 表示预测值，$n$ 则表示值的个数。MSE 的值越小，说明预测模型拥有更好的精确度。同样，我们可以尝试使用 Python 实现 MSE 计算函数：

```python
import numpy as np

def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    
    返回:
    mse -- MSE 评价指标
    """
    
    n = len(y_true)
    mse = sum(np.square(y_true - y_pred))/n
    return mse
```
**平均绝对百分比误差 $MAPE$**。

$MAPE$ 是 $MAD$ 的变形，它是一个百分比值，因此比其他统计量更容易理解。例如，如果 $MAPE$ 为 $5$，则表示预测结果较真实结果平均偏离 $5%$。$MAPE$ 的计算公式如下：
$$
{MAPE}(y, \hat{y} ) = \frac{\sum_{i=1}^{n}{|\frac{y_{i}-\hat y_{i}}{y_{i}}|}}{n} \times 100
$$

其中，$y_{i}$ 表示真实值，$\hat y_{i}$ 表示预测值，$n$ 则表示值的个数。$MAPE$ 的值越小，说明预测模型拥有更好的精确度。使用 Python 实现 MSE 计算函数：

```python
import numpy as np

def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值
    
    返回:
    mape -- MAPE 评价指标
    """
    
    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred)/y_true))/n*100
    return mape
```

---
**参考**：

- [方差（variance）、标准差（Standard Deviation）、均方差、均方根值（RMS）、均方误差（MSE）、均方根误差（RMSE）](https://blog.csdn.net/cqfdcw/article/details/78173839)
- [Mean squared error-Wikipedia](https://en.wikipedia.org/wiki/Mean_squared_error)

