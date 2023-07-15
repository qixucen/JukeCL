import torch
import torch.nn as nn


class MCBBlock(nn.Module):

    def __init__(self, input_shape, output_shape) -> None:
        super().__init__()
        # input_shape f.e.:(1,96,800)
        # output_shape  f.e.:(64,48,400)

        # 计算卷积层的参数
        in_channels = input_shape[0]
        out_channels = output_shape[0]
        kernel_size = (input_shape[1] // output_shape[1],
                       input_shape[2] // output_shape[2])
        stride = (input_shape[1] // output_shape[1],
                  input_shape[2] // output_shape[2])

        self.convlayer = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride)
        self.normlayer = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()

    def forward(self, input: torch.Tensor):
        assert len(input.shape) == 4, '4D input expected.'
        output = self.convlayer(input)
        output = self.normlayer(output)
        output = self.activation(output)
        return output


class MCB(nn.Module):

    def __init__(self, dims) -> None:
        super().__init__()
        self.MCBBlocks = nn.ModuleList(
            [MCBBlock(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, input):
        output = input
        for Block in self.MCBBlocks:
            output = Block(output)
        return output.flatten(-2)


class GRU(nn.Module):

    def __init__(self, dims) -> None:
        super().__init__()
        self.GRUBlocks = nn.ModuleList([
            nn.GRU(input_size=dims[i][0],
                   hidden_size=dims[i + 1][0],
                   num_layers=2,
                   batch_first=True) for i in range(len(dims) - 1)
        ])

    def forward(self, input):
        output = input.permute(0, 2, 1).unsqueeze(0)
        for Block in self.GRUBlocks:
            output = Block(output[0])
            print(output[0].shape)
        return output[0].permute(0, 2, 1)


if __name__ == '__main__':
    dims = [(1, 96, 800), (64, 48, 400), (128, 16, 200), (128, 4, 50)]
    m = MCB(dims)
    input = torch.randn(12, 1, 96, 800)
    output = m(input)
    # print(m.parameters)
    # print(output.shape)
    # g = nn.GRU(input_size=128,
    #            hidden_size=256,
    #            num_layers=2,
    #            batch_first=True,)
    # input = torch.randn(12, 128, 25).permute(0, 2, 1)
    # print(g(input)[0].shape)
    dims = [(128, 25), (256, 25), (128, 25)]
    g = GRU(dims)
    input = torch.randn(12, 128, 25)
    print(g(input).shape)
    # print(g.parameters)

    # rnn = nn.GRU(25, 25, 12)
    # input = torch.randn(12, 128, 25)
    # h0 = torch.randn(12, 128, 25)
    # output, hn = rnn(input, h0)
    # print(hn.shape)
