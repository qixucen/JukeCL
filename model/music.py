import torch
import torch.nn as nn


class MCBBolck(nn.Module):

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
        self.MCBBlocks = [
            MCBBolck(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ]

    def forward(self, input):
        output = input
        for MCBBlock in self.MCBBlocks:
            output = MCBBlock(output)
        return output


dims = [(1, 96, 800), (64, 48, 400), (128, 16, 200), (128, 4, 50)]
m = MCB(dims)
input = torch.randn(12, 1, 96, 800)
output = m(input)
print(output.shape)
