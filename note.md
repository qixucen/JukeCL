
labelsize: [B, 50]

datasize: [1, 59049]

the dataset will be paralleled by transform function in the loader. So when iterate the `dataloader`, the datasize will be [B, 2, 1, 59049]

### dino v2 compared to dino v1

#### can be applyed to audio domain

- V2中借鉴iBot方法引入mask机制，从而引导teacher和student网络在mask掉的区域上表达一致
- teacher网络将之前的softmax-centering方法替换为Sinkhorn-Knopp (SK) batch normalization，用以避免collapse。而student网络还是使用softmax归一化

#### maybe applyed to audio domain

- 为使batch中分布更加均匀使用KoLeo regularizer
- 对数据进行隐空间去重