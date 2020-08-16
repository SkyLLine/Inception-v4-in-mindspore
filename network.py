import mindspore as ms
import mindspore.nn as nn
from PIL import Image
from mindspore import context
import mindspore.ops.operations as P
import numpy as np
import mindspore.ops.operations as operator
import cv2 as cv
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset.transforms.vision.py_transforms as PY

class Stem(nn.Cell):
    def __init__(self, in_channels):
        super().__init__()
        self.conv2d_1a_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2,
                                       pad_mode='valid')
        self.conv2d_2a_3x3 = nn.Conv2d(32, 32, 3, stride=1, pad_mode='valid')
        self.conv2d_2b_3x3 = nn.Conv2d(32, 64, 3, stride=1)

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2, pad_mode='valid')
        self.mixed_3a_branch_1 = nn.Conv2d(64, 96, 3, stride=2, pad_mode='valid')
        self.cat = operator.Concat(1)

        self.mixed_4a_branch_0 = nn.SequentialCell([
            nn.Conv2d(160, 64, 1, stride=1),
            nn.Conv2d(64, 96, 3, stride=1, pad_mode='valid')
        ])

        self.mixed_4a_branch_1 = nn.SequentialCell([
            nn.Conv2d(160, 64, 1, stride=1),
            nn.Conv2d(64, 64, (1, 7), stride=1),
            nn.Conv2d(64, 64, (7, 1), stride=1),
            nn.Conv2d(64, 96, 3, stride=1, pad_mode='valid')
        ])

        self.mixed_5a_branch_0 = nn.Conv2d(192, 192, 3, stride=2, pad_mode='valid')
        self.mixed_5a_branch_1 = nn.MaxPool2d(2, stride=2, pad_mode='valid')

    def construct(self, x):
        x = self.conv2d_1a_3x3(x)
        x = self.conv2d_2a_3x3(x)
        x = self.conv2d_2b_3x3(x)
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = self.cat((x0, x1))

        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x = self.cat((x0, x1))

        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = self.cat((x0, x1))
        return x

if __name__=='__main__':
    context.set_context(mode=context.GRAPH_MODE)
    img = np.ones((1,3,299,299))
    img = ms.Tensor(img, ms.float32)
    net = Stem(3)
    y = net(img)
    print(y.shape)
