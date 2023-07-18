import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random


def color():
    mycolor = []
    mycolor.append('#6A539D')
    mycolor.append('#FFFF00')
    mycolor.append('#99CCCC')
    mycolor.append('#FFCCCC')
    mycolor.append('#00FFFF')
    mycolor.append('#7D6C46')
    mycolor.append('#FF99CC')
    mycolor.append('#336699')
    mycolor.append('#FFCC99')
    mycolor.append('#993366')
    return mycolor


def T_SNE(data, labels=None, num_colors=10):
    # assert len(data) == len(labels)
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    labels = labels.reshape(-1)
    n_components = 2
    tsne = TSNE(n_components=n_components)
    data_tsne = tsne.fit_transform(data)

    colors = []
    # 定义颜色映射
    for _ in range(num_colors):
        r = random.randint(0, 255) / 255
        g = random.randint(0, 255) / 255
        b = random.randint(0, 255) / 255
        colors.append((r, g, b))
    colors = color()

    plt.figure(figsize=(8, 8))
    # 绘制散点图
    for i in range(num_colors):
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        plt.scatter(data_tsne[labels == i, 0],
                    data_tsne[labels == i, 1], c=colors[i], alpha=0.2)
    #     plt.show()
    # plt.scatter(data_tsne[:, 0],
    #             data_tsne[:, 1], alpha=0.2)
    plt.show()
    plt.savefig('png/show.png')
