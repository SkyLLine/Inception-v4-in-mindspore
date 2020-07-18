import mindspore as ms
import mindspore.nn as nn
class Stem:
    def __init__(self, in_channels):
        super(Stem).__init__()
        self.conv2d_1a_3x3 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2)
        self.conv2d_2a_3x3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv2d_2b_3x3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)

        self.mixed_3a_branch_0 = nn.MaxPool2d(3, stride=2, pad_mode='valid')
        self.mixed_3a_branch_1 = nn.Conv2d(64, 96, 3, stride=2, padding=0)

        self.mixed_4a_branch_0 = nn.SequentialCell([
            nn.Conv2d(160, 64, 1, stride=1),
            nn.Conv2d(64, 96, 3, stride=1)
        ])

        self.mixed_4a_branch_1 = nn.Conv2d(160, 64, 1, stride=1)
        #### 这里需要拼接
        self.layer1 = nn.Conv2d(64, 64, (1, 7), stride=1)
        #### 这里需要拼接
        self.layer2 = nn.SequentialCell([
            nn.Conv2d(64, 64, (7, 1), stride=1),
            nn.Conv2d(64, 96, 3, stride=1)
        ])
        self.mixed_5a_branch_0 = nn.Conv2d(192, 192, 3, stride=2)
        self.mixed_5a_branch_1 = nn.MaxPool2d(3, stride=2, pad_mode='valid')

    def forwad(self, x):
        x = self.conv2d_1a_3x3(x)
        x = self.conv2d_2a_3x3(x)
        x = self.conv2d_2b_3x3(x)
        x0 = self.mixed_3a_branch_0(x)
        x1 = self.mixed_3a_branch_1(x)
        x =