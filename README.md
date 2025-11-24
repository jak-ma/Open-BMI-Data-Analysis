# OpenBMI 运动想象脑电数据二分类问题

> **报告基本要求**
> (1)分析 OpenBMI 运动想象脑电数据集；
>
> (2)利用 CSP、FBCSP 等方法提取运动想象信号的分类特征；
>
> (3) 实现至少两种分类器设计(FLDA 分类器，朴素贝叶斯分类器, SVM 分类器，深度神经
> 网络等），对不少于10个被试的运动想象脑电数据进行跨被试或不跨被试的分析；
> **加分点**
> (1)考虑如何改进系统和算法；
>
> (2) 实现多个分类方法的对比分析；
>
> (3) 对该数据集所有 54个被试数据进行分析；
>
> (4) 同时进行跨被试和不跨被试分析。

## 数据集介绍

![img](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251124194434003.png)

> 引用 

## 数据预处理

#### 基础数据预处理流程

raw MI data -> resample to 250 -> choose 20 special channels -> filter (8-30Hz) -> split -> csp -> model input data

1. 原始运动想象脑电数据
2. 时间域上进行降采样，从1000Hz采样至250Hz
3. 选择特定的通道 `['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']`，为了只保留与运动想象（MI）相关的脑区，去掉无关通道
4. 频域滤波，只保留8-30Hz段的波形
5. 分割，将原始数据拼接成一整块的MI EEG数据，按照触发点切分成一段一段的trial，用于分类
6. CSP 方法进行特征提取，只提取了2对特征
7. 得到可以输入模型训练的数据

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

   (由可视化的结果直观能够体现被试在不同模型上面的表现具有相似性，说明被试数据间存在简单/复杂之分)

   各模型平均表现

   | accuracy\model | SVM  | Logistic Regression | KNN  | 朴素贝叶斯 | RandomForest |
   | :------------: | :--: | :-----------------: | :--: | :--------: | :----------: |
   |      mean      | 0.68 |        0.68         | 0.67 |    0.63    |     0.18     |
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

   同上可视化展示在文件夹templates/cross_subject_model_accuracy_comparison

   各模型平均表现

   | accuracy\model | SVM  | Logistic Regression | KNN  | 朴素贝叶斯 | RandomForest |
   | :------------: | :--: | :-----------------: | :--: | :--------: | :----------: |
   |      mean      | 0.52 |        0.56         | 0.53 |    0.51    |     0.55     |
   |      var       | 0.06 |        0.08         | 0.06 |    0.04    |     0.08     |

   但是整体效果极差，精度大都接近0.5，几乎和猜差不多

> 至此，以满足除了加分点1的所有任务点

## 算法改进

