import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
import mindspore.dataset.transforms.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.common.initializer import TruncatedNormal


class inception_C(nn.Cell):
    def __init__(self, in_channle, bias=False):
        super().__init__()
        self.pool_cov1x1 = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode="same"),
            nn.Conv2d(in_channle, 256, kernel_size=1, has_bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.cov1x1 = nn.SequentialCell([
            nn.Conv2d(in_channle, 256, 1, has_bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ])
        self.conv1x1_pre = nn.SequentialCell([
            nn.Conv2d(in_channle, 384, 1, has_bias=bias),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ])
        self.conv1x1_next_conv1x3 = nn.SequentialCell([
            nn.Conv2d(384, 256, (1, 3), has_bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ])
        self.conv1x1_next_conv3x1 = nn.SequentialCell([
            nn.Conv2d(384, 256, (3, 1), has_bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ])

        self.conv1x1_conv1x3_conv3x1_pre = nn.SequentialCell([
            nn.Conv2d(in_channle, 384, 1, has_bias=bias),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(384, 448, (1, 3), has_bias=bias),
            nn.BatchNorm2d(448),
            nn.ReLU(),
            nn.Conv2d(448, 512, (3, 1), has_bias=bias),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ])
        self.conv1x1_conv1x3_conv3x1_next_conv1x3 = nn.SequentialCell([
            nn.Conv2d(512, 256, (1, 3), has_bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ])
        self.conv1x1_conv1x3_conv3x1_next_conv3x1 = nn.SequentialCell([
            nn.Conv2d(512, 256, (3, 1), has_bias=bias),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ])
        self.cat = operator.Concat()

    def construct(self, x):
        pool_cov1x1_out = self.pool_cov1x1(x)
        cov1x1_out = self.cov1x1(x)

        conv1x1_pre_out = self.conv1x1_pre(x)

        conv1x1_next_conv1x3_out = self.conv1x1_next_conv1x3(conv1x1_pre_out)
        conv1x1_next_conv3x1_out = self.conv1x1_next_conv3x1(conv1x1_pre_out)

        conv1x1_conv1x3_conv3x1_pre_out = self.conv1x1_conv1x3_conv3x1_pre_out(x)
        conv1x1_conv1x3_conv3x1_next_conv1x3_out = self.conv1x1_conv1x3_conv3x1_next_conv1x3(
            conv1x1_conv1x3_conv3x1_pre_out)
        conv1x1_conv1x3_conv3x1_next_conv3x1_out = self.conv1x1_conv1x3_conv3x1_next_conv3x1(
            conv1x1_conv1x3_conv3x1_pre_out)
        x = self.cat([
            pool_cov1x1_out,
            cov1x1_out,
            conv1x1_next_conv1x3_out,
            conv1x1_next_conv3x1_out,
            conv1x1_conv1x3_conv3x1_next_conv1x3_out,
            conv1x1_conv1x3_conv3x1_next_conv3x1_out
        ])
        return x
