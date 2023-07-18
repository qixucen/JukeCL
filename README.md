## Unsupervised-Rhythm-Clustering-Embedding

### encoder structure

<img src="png/encoder.png" style="zoom: 57%;" />

#### MCB block

![](png/MCB.png)

可以任意指定每层的输出维度（但由于底层是通过调节`nn.conv2d`的kernel_size实现的，所以当维度设置得不太合理时（比如在`MCB`中间层试图扩大后两维的维度，会导致kernel_size为负），可能会导致程序报错，所以尽量参考原论文的维度设置，如下图）

<img src="png/dims.png" style="zoom: 100%;" />

### two-step trainning

Unsupervised-Rhythm-Clustering-Embedding的本质，是在上游模型提取特征的基础上，再进行聚类的训练。

对于上游特征提取，可以有contrastive learning或者autoencoder这两种比较经典的无监督特征提取方法。

对于下游聚类训练，可见原论文中参考的[DEC](https://arxiv.org/pdf/1511.06335v2.pdf)，本质上是通过计算t分布来衡量嵌入点和聚类中心的相似度，再将这个t分布相似度和目标相似度计算kl散度作为损失函数。

#### contrastive learning

调用官方simclr实现，传入一个`encoder`即可，对于`encoder`的限制很宽松，只需要尾接一个命名为self.fc的层即可，在官方的simclr实现中会将这个层替换为self.

#### clustering learning

和传统对比学习不同的是，DEC不需要冻住上游模型，因为dec可训练的参数只有聚类中心。经试验发现，DEC需要的epoch并不多，否则聚类簇会倾向于聚类成线。

`class` `ClusteringLearning` need a `encoder` and `clustering ` 

#### Trainning data

数据可以在[此处]()下载，解压到data文件夹下即可