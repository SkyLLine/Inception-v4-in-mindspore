import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
import mindspore.dataset.transforms.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.common.initializer import TruncatedNormal


class inception_A(nn.Cell):
    def __init__(self, in_channle, bias=False):
        super().__init__()
        self.pool_cov1x1 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode="same"),
            nn.Conv2d(in_channle, 96, kernel_size=1, has_bias=bias),
            nn.BatchNorm2d(96),
            nn.ReLU()
        ])
        self.cov1x1 = nn.SequentialCell([
            nn.Conv2d(in_channle, 96, has_bias=bias),
            nn.BatchNorm2d(96),
            nn.ReLU()
        ])
        self.conv1x1_conv3x3 = nn.SequentialCell([
            nn.Conv2d(in_channle, 64, 1, has_bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, has_bias=bias),
            nn.BatchNorm2d(96),
            nn.ReLU()
        ])
        self.conv1x1_conv3x3_conv3x3 = nn.SequentialCell([
            nn.Conv2d(in_channle, 64, 1, has_bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 96, 3, has_bias=bias),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, has_bias=bias),
            nn.BatchNorm2d(96),
            nn.ReLU()
        ])
        self.cat = operator.Concat()

    def construct(self, x):
        pool_cov1x1_out = self.pool_cov1x1(x)
        cov1x1_out = self.cov1x1(x)
        conv1x1_conv3x3_out = self.conv1x1_conv3x3(x)
        conv1x1_conv3x3_conv3x3_out = self.conv1x1_conv3x3_conv3x3(x)
        x = self.cat([
            pool_cov1x1_out,
            cov1x1_out,
            conv1x1_conv3x3_out,
            conv1x1_conv3x3_conv3x3_out
        ])
        return x
