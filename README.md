# MathorCup2021_B

## 环境
> * cuda==12.4
> * python==3.12.7
## 依赖
> * torch==2.5.1
> * torch-geometric==2.6.1
> * torch_cluster==1.6.3+pt25cu124
> * torch_scatter==2.1.2+pt25cu124
> * torch_sparse==0.6.18+pt25cu124
> * torch_spline_conv==1.2.2+pt25cu124
> * matplotlib==3.9.3
> * PyQt5==5.15.11
> * scipy==1.14.1
> * numpy==2.1.3

# Step
## 1. 安装依赖 & 激活环境
```
conda env create -f environment.yml
conda activate pytc
```

## 2. train
首先将1000个.xyz文件存放在`Q1/dimenet/data/au20`目录下

### 非GUI环境
使用`Q1/dimenet/train_server.py`脚本进行训练

```
cd Q1/dimenet
puyhon train_server.py
```
训练流程结束后将运行`Q1/dimenet/predict.py`脚本，
输入所有数据进行推论并输出*平均绝对排名误差*和*斯皮尔曼相关系数*

<br>

#### 超参数设置
在`train_server.py`中可以设置超参数，如下所示：

```python
train_and_evaluate(
        DATA_DIR = "data/au20",
        SAVEPATH = savepath,
        NUM_EPOCHS = 10,
        LEARNING_RATE = 1e-4,
        BATCH_SIZE = 32,
        CUTOFF_RADIUS = 6.0)
```
#### 结果保存
* 本次训练Val_loss最低的模型将保存在`Q1/dimenet/models/`目录下
* 最终Train_loss和Val_loss将保存在`Q1/dimenet/fig/`目录下
* 推理结果在`Q1/dimenet/predict_result`的csv文件中，时间标记与对应的模型相同


<br>

### GUI环境
使用`train.py`脚本进行训练

matplotlib使用交互模式，其余同上


