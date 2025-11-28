# OpenBMI 运动想象脑电数据二分类问题

> **报告基本要求**
>
> (1)分析 OpenBMI 运动想象脑电数据集；
>
> (2)利用 CSP、FBCSP 等方法提取运动想象信号的分类特征；
>
> (3) 实现至少两种分类器设计(FLDA 分类器，朴素贝叶斯分类器， SVM 分类器，深度神经
> 网络等），对不少于10个被试的运动想象脑电数据进行跨被试或不跨被试的分析；
>
> **加分点**
>
> (1)考虑如何改进系统和算法；
>
> (2) 实现多个分类方法的对比分析；
>
> (3) 对该数据集所有 54个被试数据进行分析；
>
> (4) 同时进行跨被试和不跨被试分析。

## 数据集介绍

![img](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251124194434003.png)

> 引用博主网图

（关于数据集如何详细的说明，就不过多赘述）

## 数据预处理

#### 基础数据预处理流程

raw MI data -> resample to 250 -> choose 20 special channels -> filter (8-30Hz) -> split -> csp -> model input data

1. 原始运动想象脑电数据       一个被试的 'train' 部分数据处理过程中的维度变化        (1418040, 62)

2. 时间域上进行降采样，从1000Hz采样至250Hz

3. 选择特定的通道 `['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']`，为了只保留与运动想象（MI）相关的脑区，去掉无关通道

4. 频域滤波，只保留 8-30Hz 段的波形

5. 分割，将原始数据拼接成一整块的MI EEG数据，按照触发点切分成一段一段的 trial ，用于分类

   一个 trial 就是一次实验        (100, 20, 626)

6. CSP 方法进行特征提取，只提取了**2对**特征(4个，即n_components=4)        (100, 8)

7. 得到可以输入模型训练的数据

（感觉数据维度变化可以简单说明一下原因，并结合着解释这些操作的原因）

上述处理流程为算法改进之前对以下三个实验采取的通用处理流程，作为 **Benchmark**

## 不跨试分析

within-subject analysis 主要采用了两种方式：

1. 只采用会话1的数据，对于每一个被试，将其内部的 `'train'` 标签对应字段(向量) `'test'` 标签对应字段，做一个拼接(concat)组成新的特征向量。然后在每一个被试内部进行5折交叉验证(`Kfold`)并取平均值，如此得到一个被试的一个模型的表现 accuracy 。

   传统机器学习模型在每一个被试上的表现结果

   `RandomForest`

   ![image-20251124203306153](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251124203306271.png)

   `朴素贝叶斯`

   ![image-20251124203606241](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251124203606310.png)

   `KNN`

   ![image-20251124204212495](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251124204212597.png)

   `Logistic Regression`

   ![image-20251124204535089](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251124204535143.png)

   `SVM`

   ![image-20251124204852117](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251124204852179.png)

   (由可视化的结果直观能够体现被试在不同模型上面的表现具有分布相似性，说明被试数据间存在简单/复杂之分)

   各模型平均表现

   | accuracy\model | SVM  | Logistic Regression | KNN  | 朴素贝叶斯 | RandomForest |
   | :------------: | :--: | :-----------------: | :--: | :--------: | :----------: |
   |      mean      | 0.68 |        0.68         | 0.67 |    0.63    |     0.68     |
   |      var       | 0.17 |        0.17         | 0.16 |    0.15    |     0.16     |

2. 对于每一个被试，采用会话1(2)的数据训练，会话2(1)的数据进行测试，同样在每一个被试内部拼接`'train'` 标签对应字段(向量)和 `'test'` 标签对应字段；同样的取两次的测试的平均值，如此得到一个被试跨时间维度在一个模型的表现 accuracy 。

   同1的可视化展示在文件夹templates/within_subject_cross_session_accuracy_comparison

   各模型平均表现

   | accuracy\model | SVM  | Logistic Regression | KNN  | 朴素贝叶斯 | RandomForest |
   | :------------: | :--: | :-----------------: | :--: | :--------: | :----------: |
   |      mean      | 0.63 |        0.63         | 0.62 |    0.60    |     0.62     |
   |      var       | 0.16 |        0.16         | 0.16 |    0.15    |     0.15     |

## 跨被试分析

cross-subject analysis 主要采用一种方式：

1. 只采用会话1的数据，同上，在被试内部做拼接；使用留一法(`LeaveOneOut`)进行训练和测试；得到每一个被试在作为测试样本时在不同模型上的表现 accuracy 。(此处可以分析被试数据的复杂度)

   同上可视化展示在文件夹 templates/cross_subject_model_accuracy_comparison

   各模型平均表现

   | accuracy\model | SVM  | Logistic Regression | KNN  | 朴素贝叶斯 | RandomForest |
   | :------------: | :--: | :-----------------: | :--: | :--------: | :----------: |
   |      mean      | 0.52 |        0.56         | 0.53 |    0.51    |     0.55     |
   |      var       | 0.06 |        0.08         | 0.06 |    0.04    |     0.08     |

   但是整体效果极差，精度大都接近0.5，几乎和猜差不多

> 至此，以满足除了加分点1的所有任务点

## 算法改进

算法改进主要围绕数据和模型两个方面来进行，数据方面就是数据的预处理，模型方面就是模型结构的选择

#### 数据处理

1. 首先是数据预处理中使用csp方法进行特征提取，经过实验探索（适当调整CSP提取的空间滤波器数量，以找到最佳的特征维度）最终选择了输出 4对特征（8个即n_components=8）时性能最佳（之前是2对）。

   对于**不跨被试**的两个实验都有略微的提升：

   - 不跨session

   | accuracy\model |    SVM     | Logistic Regression |    KNN     | 朴素贝叶斯 | RandomForest |
   | :------------: | :--------: | :-----------------: | :--------: | :--------: | :----------: |
   |      mean      | 0.70&uarr; |     0.70&uarr;      |    0.67    |    0.62    |  0.70&uarr;  |
   |      var       | 0.15&darr; |     0.15&darr;      | 0.14&darr; | 0.14&darr; |  0.15&darr;  |

   - 跨session

   | accuracy\model |    SVM     | Logistic Regression |    KNN     | 朴素贝叶斯 | RandomForest |
   | :------------: | :--------: | :-----------------: | :--------: | :--------: | :----------: |
   |      mean      | 0.64&uarr; |     0.65&uarr;      |    0.62    |    0.59    |  0.64&uarr;  |
   |      var       |    0.16    |     0.15&darr;      | 0.15&darr; | 0.14&darr; |     0.15     |

2. 探索数据自身特点

   **时间波形图**（做了垂直偏移变换，为了防止可视化时通道重叠）实际上大部分电位分布在±20之间，见下图

   <img src="https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251128190901159.png" alt="1" style="zoom: 25%;" />

   **EEG 幅度直方图**

   <img src="https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251128182400149.png" alt="屏幕截图 2025-11-28 174039" style="zoom:33%;" />

   ​        由此可以看出原始脑的电位数据分布还是比较符合正态分布的，数据质量比较好

   **CSP 特征方差分布图**  (CSP 特征提取以后两个类别对比)

   <img src="https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251128185213114.png" alt="image-20251128185213069" style="zoom:33%;" />

   ​        大部分由CSP方法提取得到的特征分布方差图都与上图类似，可以看出使用**CSP方法并不能完全的将两个类别分开**，甚至有的特征维度还出现了大量的重叠区域

   > 首先我们尝试了做 **归一化** ，但是结果是性能提升几乎为0。分析原因就是使用 csp 做特征提取之后得到的特征向量自身数量级已经很小，几乎都在0-1之间了，所以做归一化的变化其实就不大了。

   

   

#### 模型改进

我们采用了官方的 `EEGNetv1` 模型用在我们的**跨被试**实验上，整体表现远远优于之前的几个传统机器学习模型

<img src="https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251128133049948.png" alt="EEGNetv1" style="zoom:50%;" />

但是显然还是有些被试比较难处理，这也是未来需要继续挑战的难题。

同时在**不跨被试**上我们也尝试应用了 `EEGNetv1` 模型，但是感觉效果相对提升的话没有前者那么明显：

- 跨session实验

  <img src="https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251128164800813.png" alt="神经网络模型下每个被试的表现(EEGNetv1)" style="zoom:50%;" />

  可以看出，每个被试的最好 Acc 几乎都大量分布在0.6以下，相对不如之前的传统机器学习方法。

  解释：可能是由于数据量太少，模型欠拟合导致的。

- 不跨session实验，感觉没有必要做实验，数据量更少，重点放在如何在数据处理层面来优化这个实验的性能。
