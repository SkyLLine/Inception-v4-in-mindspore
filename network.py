import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
class Stem:
    def __init__(self, in_channels):
        super(Stem).__init__()
        self.conv2d_1a_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2)
        self.conv2d_2a_3x3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv2d_2b_3x3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2, pad_mode='valid')
        self.mixed_3a_branch_1 = nn.Conv2d(64, 96, 3, stride=2, padding=0)
        self.cat_3a_branch = operator.Concat()

        self.mixed_4a_branch_0 = nn.SequentialCell([
            nn.Conv2d(160, 64, 1, stride=1),
            nn.Conv2d(64, 96, 3, stride=1)
        ])

        self.mixed_4a_branch_1 = nn.Conv2d(160, 64, 1, stride=1)
        #### 这里需要拼接
        self.layer1 = nn.Conv2d(64, 64, (1, 7), stride=1, pad_mode='same')
        #### 这里需要拼接
        self.layer2 = nn.SequentialCell([
            nn.Conv2d(64, 64, (7, 1), stride=1, pad_mode='same'),
            nn.Conv2d(64, 96, 3, stride=1, pad_mode='valid')
        ])
        self.cat_4a_branch = operator.Concat()

        self.mixed_5a_branch_0 = nn.Conv2d(192, 192, 3, stride=2, pad_mode='valid')
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2, pad_mode='valid')
        self.cat_5a_branch = operator.Concat()

    def forwad(self, x):
        x = self.conv2d_1a_3x3(x)
        x = self.conv2d_2a_3x3(x)
        x = self.conv2d_2b_3x3(x)
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x = self.cat_3a_branch([x0, x1])

        x0 = self.mixed_4a_branch_0(x)
        x1 = self.mixed_4a_branch_1(x)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x = self.cat_4a_branch([x0, x1])

        x0 = self.mixed_5a_branch_0(x)
        x1 = self.mixed_5a_branch_1(x)
        x = self.cat_5a_branch([x0, x1])
