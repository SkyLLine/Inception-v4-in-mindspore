import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
import mindspore.dataset.transforms.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.common.initializer import TruncatedNormal


class reduction_A(nn.Cell):
    def __init__(self, in_channle, bias=False):
        super().__init__()
        self.pool = nn.SequentialCell([
            nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid"),
        ])
        self.cov3x3 = nn.SequentialCell([
            nn.Conv2d(in_channle, 384, 3, stride=2, has_bias=bias, pad_mode="valid"),
            nn.BatchNorm2d(384),
            nn.ReLU()
        ])
        self.conv1x1_conv3x3_conv3x3 = nn.SequentialCell([
            nn.Conv2d(in_channle, 192, 1, has_bias=bias),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.Conv2d(192, 224, 3, has_bias=bias),
            nn.BatchNorm2d(224),
            nn.ReLU(),
            nn.Conv2d(224, 256, 3, stride=2, has_bias=bias, pad_mode="valid"),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.cat = operator.Concat()

    def construct(self, x):
        pool_out = self.pool(x)
        cov3x3_out = self.cov3x3(x)
        conv1x1_conv3x3_conv3x3 = self.conv1x1_conv3x3_conv3x3(x)
        x = self.cat([
            pool_out,
            cov3x3_out,
            conv1x1_conv3x3_conv3x3,
        ])
        return x
