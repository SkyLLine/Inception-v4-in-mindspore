import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
import os
from inception_A import inception_A
from inception_B import inception_B
from inception_C import inception_C
from network import Stem
from reduction_A import reduction_A
from reduction_B import reduction_B
from reduction_C import reduction_C
import mindspore.dataset as ds

dict = {}
i = 0

for filename in os.listdir('./ImageNet_train'):
    dict[filename] = i
    i += 1


class InceptionV4(nn.Cell):
    def __init__(self, A, B, C):
        super().__init__()
        self.Stem = Stem(3)
        self.inception_A = self.generate_inception_module(384, 384, A, inception_A)
        self.reduction_A = reduction_A(384)
        self.inception_B = self.generate_inception_module(1024, 1024, B, inception_B)
        self.reduction_B = reduction_B(1024)
        self.inception_C = self.generate_inception_module(1536, 1536, C, inception_C)
        self.avgpool = nn.AvgPool2d(7)

        self.dropout = nn.Dropout(0.8)
        self.linear = nn.Dense(1536, 1000, activation='softmax')


    def forward(self, x):
        x = self.Stem(x)
        x = self.inception_A(x)
        x = self.reduction_A(x)
        x = self.inception_B(x)
        x = self.reduction_B(x)
        x = self.inception_C(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 1536)
        x = self.linear(x)
        return x

    def generate_inception_module(input_channels, output_channels, block_num, block):
        layers = nn.SequentialCell([block(input_channels)])
        for i in range(block_num):
            layers = nn.SequentialCell(block(input_channels), layers)
            input_channels = output_channels

        return layers

def train(epoch):
    pass

if __name__=='__main__':
    DATA_DIR = "dataset/"

    # imagenet = ds.(DATA_DIR)