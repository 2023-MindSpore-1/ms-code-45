# 目录

# 模型名称

Correlation Networks(CorNet)为极端多标签文本分类( XMTC )任务的网络架构。极端多标签文本分类任务的目标是从一个非常大的标签集合中，用最相关的标签子集对输入文本序列进行标记，可以在许多实际应用中找到，例如文档标注和产品标注。

CorNet通过在深度模型的预测层添加额外的CorNet模块表示不同标签之间有用的相关性信息，利用相关性知识增强原始标签预测并输出增强的标签预测。了解更多的网络细节，请参考[CorNet论文](https://dl.acm.org/doi/pdf/10.1145/3394486.3403151) 。

## 模型架构

通过为极端多标签文本分类架构--XMLCNN增加CorNet模块，获取不同标签之间的相关性，实现多标签文本分类任务性能的提升。


## 数据集

[EUR-Lex 数据集](https://drive.google.com/file/d/15WSOexahaC-5kIcraYReFXR84TSuTejc/view?usp=sharing)是有关**欧盟法律的文件**集合。它包含许多不同类型的文件，包括条约、立法、判例法和立法提案，这些文件根据**几个正交分类方案**编制索引，共3801个类别，涉及欧洲法律的不同方面。对于多标签分类问题而言，一个样本可能同时属于多个类别。

此外，还需要下载预训练的[gensim模型](https://drive.google.com/file/d/1A_jGmpsq7dVAN0-eHZ3RZaPNL-ZdViIr/view)进行数据预处理。

* 目录结构如下：

  ```
  ├── deep_data
      ├── EUR-Lex
          ├─ train_texts.txt
          ├─ train_labels.txt
          ├─ test_texts.txt
          ├─ test_labels.txt
          ├─ test.txt
          └─ train.txt
      ├── glove.840B.300d.gensim.vectors.npy
      ├── glove.840B.300d.gensim
  ```

* 执行数据预处理：

  ```
  bash ./scripts/preprocess_eurlex.sh
  ```

* 数据预处理后的目录结构：

  ```
  ├── deep_data
      ├── EUR-Lex
          ├─ train_texts.txt
          ├─ train_labels.txt
          ├─ test_texts.txt
          ├─ test_labels.txt
          ├─ test.txt
          ├─ train.txt
          ├─ vocab.npy             
          ├─ train_texts.npy
          ├─ train_labels.npy
          ├─ test_texts.npy
          ├─ test_labels.npy
          ├─ labels_binarizer
          └─ emb_init.npy
      ├── glove.840B.300d.gensim.vectors.npy
      ├── glove.840B.300d.gensim
  ```

## 环境要求

* 硬件（Ascend）
  - 使用Ascend处理器来搭建硬件环境。

- 框架
  - [MindSpore](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Finstall)

- 如需查看详情，请参见如下资源
  - [MindSpore教程](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Ftutorials%2Fzh-CN%2Fmaster%2Findex.html)
  - [MindSpore Python API](https://gitee.com/link?target=https%3A%2F%2Fwww.mindspore.cn%2Fdocs%2Fzh-CN%2Fmaster%2Findex.html)

* 安装依赖的环境

  ```
  pip install requirements.txt
  ```

## 快速入门

* 本地训练：

  ```
  # 单卡训练
  python train.py --run_modelarts=Flase --is_distributed=False
  ```

  ```
  # 通过shell脚本进行8卡训练
  bash ./scripts/run_distribute_train.sh	
  ```

* 本地评估：

  ```
  python eval.py
  ```

* 在 ModelArts 进行训练 (如果你想在modelarts上运行，可以参考以下文档 [modelarts](https://gitee.com/link?target=https%3A%2F%2Fsupport.huaweicloud.com%2Fmodelarts%2F))

  * 在 ModelArts 上使用单卡训练 

    ```
    # (1) 在网页上设置 
    # (2) 执行a或者b
    # (3) 讲预处理好的数据集并压缩为.zip上传到桶上
    # (4) 在网页上设置启动文件为 "train.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    #     在网页上设置 "run_modelarts=True"
    #     在网页上设置 "is_distributed=False"
    ```

  * 在 ModelArts 上使用多卡训练

    ```
    # (1) 在网页上设置 
    # (2) 执行a或者b
    # (3) 讲预处理好的数据集并压缩为.zip上传到桶上
    # (4) 在网页上设置启动文件为 "train.py"
    # (5) 在网页上设置"训练数据集"、"训练输出文件路径"、"作业日志路径"等
    # (6) 创建训练作业
    #     在网页上设置 "run_modelarts=True"
    #     在网页上设置 "is_distributed=True"
    ```

## 脚本说明

### 脚本和样例代码

```
├── CorNet
    ├── ascend310_infer  #用于310推理
        ├─ inc
        	├─ utils.h
        ├─ src
        	├─ main.cc
        	├─ utils.c
        ├─ build.sh
        └─ CMakeLists.txt
    ├── configure       #模型及数据集路径及参数配置
    	├─ datasets
    		├─ AmazonCat-13K.yaml
    		├─ EUR-Lex.yaml
    		└─ Wiki-500K.yaml
    	└─ models
    		└─ CorNetXMLCNN-EUR-Lex.yaml
    ├── deepxml         #网络代码部分
    	├─ _init_.py
    	├─ callback.py
    	├─ cornet.py
    	├─ data_preprocess.py  #原始数据预处理
    	├─ data_utils.py
    	├─ dataset.py
    	├─ evaluation.py
    	├─ trainonestep.py
    	└─ xmlcnn.py
    ├── scripts         #训练脚本
    	├─ preprocess_eurlex.sh
    	├─ preprocess_other.sh
    	├─ run_distribute_train.sh
    	├─ run_distribute_train_mpi.sh
    	├─ run_infer_310.sh
    	└─ run_models.sh
    ├── eval.py           #用于模型推理
    ├── evaluation.py
    ├── export.py         #由.ckpt模型导出.mindir模型
    ├── postprocess.py    #推理部分后处理，由推理结果与输入数据的标签计算推理结果
    ├── preprocess.py     #推理部分数据预处理，生成二进制数据
    ├── README.md
    ├── README_CN.md
    ├── requirements.txt
    └── train.py          #用于拉起模型训练
```

### 脚本参数

数据集配置：

```
run_modelarts：1                  # 是否云上训练

# modelarts云上参数
data_url: ""                      # S3 数据集路径
train_url: ""                     # S3 输出路径
checkpoint_url: ""                # S3 预训练模型路径
output_path: "/cache/train"       # 真实的云上机器路径，从train_url拷贝
dataset_path: "/cache/datasets/deep_data" # 真实的云上机器路径，从data_url拷贝
load_path: "/cache/model/best_38_0.5416439549567373.ckpt" #真实的云上机器路径，从checkpoint_url拷贝

# 训练参数
dynamic_pool_length: 8
bottleneck_dim: 512
num_filters: 128
dropout: 0.5
emb_trainable: False
batch_size: 32
nb_epoch: 45
swa_warmup: 10
embedding_size: 300
```

更多配置细节请参考脚本`train.py`, `eval.py`, `export.py` 和 `config/datasets/EUR-Lex.yaml`,`config/datasets/AmazonCat-13K.yaml`,`config/datasets/Wiki-500K.yaml`,`config/models/CorNetXMLCNN-EUR-Lex.yaml`。

## 训练过程

#### 单卡训练

在Ascend设备上，使用python脚本直接开始训练(单卡)

* python命令启动

  ```
  python train.py
  ```

* shell脚本启动

  ```
  bash run_models.sh
  ```

  训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果。训练过程日志：

  ```
  
  ```

#### 分布式训练

在Ascend设备上，使用shell脚本执行分布式训练示例(8卡)

* shell脚本启动

  ```
  bash run_distribute_train.sh
  ```

  训练checkpoint将被保存在`XXXX/ckpt_0`中，你可以从如下的log文件中获取结果。训练过程日志：

  ```
  
  ```

## 评估

### 评估过程

* python命令启动

  ```
  python eval.py
  ```

### 评估结果

上述python命令将在后台运行，您可以通过eval.log文件查看结果。测试数据集的准确性如下：

```log

```

## 导出

### 导出过程

```
#将.ckpt模型导出为mindir模型
python export.py
```

## 推理

### 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

###  用法

#### 相关说明

- 首先要通过执行export.py导出mindir文件
- 通过preprocess.py将数据集转为二进制文件
- 执行postprocess.py将根据mindir网络输出结果进行推理，并保存评估指标等结果

执行完整的推理脚本如下：

```
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATA_DIR] [DEVICE_ID]
```

### 推理结果

推理结果保存在当前路径，通过cat acc.log中看到最终精度结果。

```
```



## 性能

### 训练性能

CorNet应用于EUR-Lex训练数据集上：

| Parameters                 | Ascend 910                                                   | GPU |
| -------------------------- | ------------------------------------------------------------ | ----------------------------------------------|
| Model Version              | CorNet                                                |  CorNet                                |
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8  |  3090                             |
| uploaded Date              | 12/0425/2022                               | 12/04/2022                 |
| MindSpore Version          | 1.9                                                        | 1.3.0                                         |
| Dataset                    | EUR-Lex                                               | EUR-Lex                               |
| Training Parameters        | epoch=,  batch_size = 32             | epoch=30,  batch_size = 32 |
| Optimizer              | Adam                                                        | Adam                                  |
| Loss Function              | BCEWithLogitsLoss                       | BCEWithLogitsLoss        |
| outputs               | logit                                                       | logit                               |
| Loss                       |                                                  |                                   |
| Speed                      | ms/step（8pcs）                                             | 0.037 ms/step（8pcs）                      |
| Total time                 | mins                                                       | 16 mins                                   |
| Parameters (M)             | 1.03GB                                                   | 263.97MB                                 |

### 推理性能

> 提供推理性能的详细描述，包括耗时，精度等

你可以参照如下模板

| Parameters          | Ascend       |
| ------------------- | ------------ |
| Model Version       | CorNetXMLCNN |
| Resource            | Ascend 910   |
| Uploaded Date       | 12/4/2022    |
| MindSpore Version   | 1.9          |
| Dataset             | EUR-Lex      |
| batch_size          | 32           |
| outputs             | logit        |
| Accuracy            | %            |
| Model for inference | 1.03GB       |

## 随机情况说明



## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。
