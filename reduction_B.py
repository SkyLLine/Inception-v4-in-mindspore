import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
import mindspore.dataset.transforms.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.common.initializer import TruncatedNormal


class reduction_B(nn.Cell):
    def __init__(self, in_channle, bias=False):
        super().__init__()
        self.pool = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid"),
        ])

        self.conv1x1_conv3x3 = nn.SequentialCell([
            nn.Conv2d(in_channle, 192, 1, has_bias=bias),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=2, has_bias=bias, pad_mode="valid"),
            nn.BatchNorm2d(192),
            nn.ReLU(),
        ])
        self.conv1x1_conv1x7_conv_7x1_conv3x3 = nn.SequentialCell([
            nn.Conv2d(in_channle, 256, 1, has_bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (1, 7), has_bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 320, (7, 1), has_bias=bias),
            nn.BatchNorm2d(320),
            nn.ReLU(),
            nn.Conv2d(320, 320, 3, stride=2, has_bias=bias, pad_mode="valid"),
            nn.BatchNorm2d(320),
            nn.ReLU(),
        ])
        self.cat = operator.Concat(1)

    def construct(self, x):
        pool_out = self.pool(x)
        conv1x1_conv3x3_out = self.conv1x1_conv3x3(x)
        conv1x1_conv1x7_conv_7x1_conv3x3_out = self.conv1x1_conv1x7_conv_7x1_conv3x3(x)
        x = self.cat((
            pool_out,
            conv1x1_conv3x3_out,
            conv1x1_conv1x7_conv_7x1_conv3x3_out,
        ))
        return x
