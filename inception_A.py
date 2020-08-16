import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
from mindspore import context
import numpy as np
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
            nn.Conv2d(in_channle, 96, 1, has_bias=bias),
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
        self.cat = operator.Concat(1)

    def construct(self, x):
        pool_cov1x1_out = self.pool_cov1x1(x)
        cov1x1_out = self.cov1x1(x)
        conv1x1_conv3x3_out = self.conv1x1_conv3x3(x)
        conv1x1_conv3x3_conv3x3_out = self.conv1x1_conv3x3_conv3x3(x)
        x = self.cat((
            pool_cov1x1_out,
            cov1x1_out,
            conv1x1_conv3x3_out,
            conv1x1_conv3x3_conv3x3_out
        ))
        return x

if __name__=='__main__':

    # img = np.ones((1, 384, 35, 35))
    # img = ms.Tensor(img, ms.float32)
    # net = inception_A(384)
    # y = net(img)
    # print(y.shape)
    context.set_context(mode=context.GRAPH_MODE)
    pool = nn.AvgPool2d(kernel_size=3, stride=1)
    x = ms.Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), ms.float32)
    output = pool(x)
