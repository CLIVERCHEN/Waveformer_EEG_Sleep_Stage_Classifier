# Waveformer: A Transformer based EEG Sleep Stage Classifier
### 项目背景
该项目为西安交通大学模式识别与机器学习BIME412113课程大作业，给出的数据集包括7个人睡眠时的大脑EEG信号，其中以30秒为一段，每段数据附有一个标签，该标签代表这30秒对应的睡眠分类。分类包括：
- 清醒期（Wake，W）

- 非快速眼动睡眠I期（N1）

- 非快速眼动睡眠II期（N2）

- 非快速眼动睡眠III期（N3）

- 快速眼动睡眠期（REM）

本次项目中，为了简化分类数目，将N1和N2期统称为浅睡期，进行睡眠分期的四分类，即清醒期、浅睡期（N1和N2）、深睡期（N3）、快速眼动期，在数据集中用0,(1.2),3,4表示。
### 文件结构
```bash
Wavefromer/
│
├── src/                        
│   ├── models/                 
│   │   ├── CAE.py             # CAE模型
│   │   └── main_model.py      # Wavefromer模型
│   ├── wavelet_transform.py   # 小波变换
│   ├── visualization.py       # 可视化
│   ├── energy.py              # 生成每段EEG的能量
│   ├── CAE_train.py           # CAE训练
│   └── train.py               # 训练代码
│
├── data/                       
│   ├── raw_data/              # 原始EEG数据
│   ├── energy_data/           # 能量数据
│   ├── pretreatment_data/     # 原始 EEG 数据
│   └── wavelet_data/          # 小波变换后的数据
│
├── results/                    
│
├── requirements.txt
├── README.md
└── .gitignore 

```
### 方法
对大脑EEG信号进行特征提取后，使用决策树、SVM等机器学习算法进行分类的方法，并未充分利用EEG信号的时序信息，因此本算法利用小波变换，同时捕获信号的时域及频域信息，并利用基于transformer的架构进行EEG信号的分类。
#### 小波变换
小波变化(Wavelet Transform，WT)是一种对信号进行时频域分析的方法，其继承了短时傅里叶变换(Short-time Fourier Transform, STFT)局部化分析信号的思想，同时又克服了窗口大小不随频率变化等缺点。对一维的信号进行小波变换后得到一个二维矩阵，其两个轴分别代表时间和反比于频率的尺度，其中点代表一个时间-频率对应的小波系数，其绝对值代表该时间点上，该尺度的特征的显著程度。\
对于原始信号连续信号 $x(t)$ 进行小波变换，可以通过以下公式描述：
$$W_x(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^* \left( \frac{t-b}{a} \right) dt$$\
其中 $W_x(a, b)$ 是小波变换的系数，表示信号在控制小波伸缩的尺度因子a和控制小波在时间轴上移动的平移因子b下的特性，$\psi^*(t)$ 是小波函数的复共轭，$\psi(t)$ 表示小波函数，考虑到大脑EEG信号具有较强的震荡属性，此处选用Morlet小波。\
Morlet 小波可以表示为：
$$\psi(t) = \pi^{-1/4} e^{i\omega_0 t} e^{-\frac{t^2}{2}}$$\
其中 $\pi^{-1/4}$ 是归一化因子，确保小波具有单位能量，$e^{i\omega_0t}$ 是复指数函数，代表一个频率为 $\omega_0$ 的正弦波， $e^{-\frac{t^2}{2}}$ 是高斯窗函数，确保小波具有有限的支持， $\omega_0$ 是小波的中心频率。
在小波变换时，根据先验知识选择变换的尺度，即EEG信号的不同频段有着不同的生理意义：

>δ (Delta) 频带：约 0.5-4 Hz，与深度睡眠和无意识状态有关。\
>θ (Theta) 频带：约 4-8 Hz，与轻度到中度的睡眠、放松、冥想、初级的认知过程和REM（快速眼动）睡眠阶段有关。\
>α (Alpha) 频带：约 8-14 Hz，与放松和闭眼休息有关。通常在一个人闭眼但醒着时被观察到。\
>β (Beta) 频带：约 14-30 Hz，与觉醒状态、思考、决策、数据分析和其他认知活动有关。\
>γ (Gamma) 频带：约 30-100 Hz，与更高级的认知过程、知觉和学习等活动有关。

考虑到需要对睡眠阶段进行分类，则需要对 $\delta,\theta,\alpha,\beta$ 这四个频带进行分析。在小波变换中，小波尺度a与频率f的关系可以由下式给出：
$$f = \frac{\omega_0}{2\pi a}$$\
其中 $\omega_0$ 是小波的中心频率，对于Morlet小波通常取5-6，由此可得需要使用的几个尺度。
#### 基于Transformer的模型
Transformer在NLP领域有着重要作用，是如今LLM的基础，其拥有强大的时序信息的处理能力。大脑EEG信号的信噪比低，但具有较强的时序特性，利用时间上相连的多个30s片段之间的联系进行分类，相比于分析一个30s片段的特征，会有更好的效果。本次实验使用其encoder部分，对小波变换后得到的矩阵进行分析，由网络最后的线性层输出分类结果。
#### 卷积自编码器CAE
为了对小波变换后产生的二维系数矩阵进行降维，本次实验采用两种处理方法。第一种方法将小波变换中的小波系数变换后直接求和，组成降维后的向量。第二种方法使用卷积自编码器，卷积自编码器将小波变换生成的二维矩阵降维至一维向量，再通过反卷积恢复回二维矩阵，通过最小化输入输出的差值，可以通过训练好的编码器用于二维矩阵的压缩。
### 步骤
首先对原始数据进行分析处理，在5个分类中各随机选择1个信号，对其波形进行可视化，结果如图：
![](https://github.com/CLIVERCHEN/Waveformer_EEG_Sleep_Stage_Classifier/blob/main/result/raw_data_visualization.png)
将原始数据进行0.5到30Hz的滤波，使用Butterworth数字滤波，之后将其还原到时域，下图展示了滤波前后时域的波形图对比：
![](https://github.com/CLIVERCHEN/Waveformer_EEG_Sleep_Stage_Classifier/blob/main/result/filter.png)
再将每条30s数据进行0.5-4, 4-8, 8-14, 14-30四个频率区间内各取8个频点的小波变换，得到若干尺寸为32 $\times$ 7680的矩阵，随机选取一个矩阵进行可视化，如图：
![](https://github.com/CLIVERCHEN/Waveformer_EEG_Sleep_Stage_Classifier/blob/main/result/wavelet_visualization.png)
接下来介绍使用第一种降维方式进行分类的方法，即将小波系数取绝对值后，进行对数归一化，对所有时间点，在每个频点上对小波系数求和，组成一个降维后的向量（简称为energy）。将64个向量分别作为token输入Transformer，输出64个分类。具体而言，输出64 $\times$ 4个分类信息，每个最终分类标签由4个输出中最大的决定。模型的结构如下图：
![](https://github.com/CLIVERCHEN/Waveformer_EEG_Sleep_Stage_Classifier/blob/main/result/Waveformer.png)
对该模型进行10折交叉验证，每折对数据集循环150个epoch，其准确率和loss的变化图如下：
![](https://github.com/CLIVERCHEN/Waveformer_EEG_Sleep_Stage_Classifier/blob/main/result/train_with_energy.png)
可见该模型的最优准确率约为75%，若加大训练的epoch数，则可能进一步提升。也可以更细致地调整学习率、正则化系数等超参数，实现更好的结果。但需要注意的是，该模型的训练并不稳定，训练结果在波动较大，意味着训练稳定性和泛化性有提高空间。\
第二种降维方式使用了卷积自编码器CAE，考虑到32 $\times$ 7680的矩阵过于扁平，不易设计卷积核大小、步长等参数，则增加0.5-4, 4-8, 8-14, 14-30四个频率区间内取样频点的个数至8个。通过多步的一维和二维卷积将小波变换产生64 $\times$ 7680的矩阵降维至32 $\times$ 1，送入transformer进行分类。CAE的结构设计如图：
![](https://github.com/CLIVERCHEN/Waveformer_EEG_Sleep_Stage_Classifier/blob/main/result/CAE.png)
但由于CAE的训练效果不佳，其loss的变化曲线如下：
![](https://github.com/CLIVERCHEN/Waveformer_EEG_Sleep_Stage_Classifier/blob/main/result/CAE_train_loss.png)\
原因可能由于内存容量所限，不能将所有数据一次装入，只能分批将输入送入，造成了数据集质量的下降。卷积设计可以也进一步优化，更多考虑小波变换后矩阵的特性。
因为训练出的CAE质量不高，其最终的分类结果差于第一种压缩方法，结果如下：
![](https://github.com/CLIVERCHEN/Waveformer_EEG_Sleep_Stage_Classifier/blob/main/result/train_with_CAEencoder.png)
但我认为该方案在优化后，会有超出第一种方法的准确率和泛化性，因为自编码器对小波变换得到的矩阵的表达能力超过简单在时间上求和，但这需要进一步设计CAE的结构。
### 总结
本次项目是我第一次尝试使用深度学习算法处理时序信息，所以很多训练细节尚不完善，尤其是对CAE的设计和训练，需要进一步学习提高。对于EEG信号的处理上，我最初的想法是对于一段30s长度的信号进行有交叠的切分，将切分后的矩阵降维后输入transformer，但效果并不好。经过王刚老师的提示，我意识到30s内的数据在时序上对分类任务并无帮助，且由于EEG信号信噪比较小，过分关注一段信号内的信息是不明智的，应该利用数据（即多段30s的数据）之间的时序联系。\
感谢白嫖的阿里云服务器（H100nb！），感谢[王刚老师](https://gr.xjtu.edu.cn/web/ggwang/home)对我的指导！

