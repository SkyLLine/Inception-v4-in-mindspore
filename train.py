import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
import os
from lr_generator import get_lr
from preprocess import pre
from CrossEntropy import CrossEntropy
import argparse
from inception_A import inception_A
from inception_B import inception_B
from inception_C import inception_C
from network import Stem
from reduction_A import reduction_A
from reduction_B import reduction_B
from reduction_C import reduction_C
import mindspore.dataset as ds
from mindspore import context
from mindspore import Tensor
from mindspore.model_zoo.resnet import resnet50
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum

from mindspore.train.model import Model, ParallelMode
from config import config
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from dataloader import create_dataset

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=bool, default=True, help='Run distribute')
parser.add_argument('--device_num', type=int, default=8, help='Device num.')
parser.add_argument('--do_train', type=bool, default=True, help='Do train or not.')
parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')
parser.add_argument('--data_url', default=None, help='Location of data.')
parser.add_argument('--train_url', default=None, help='Location of training outputs.')

opt = parser.parse_args()

dict = {}
i = 0

for filename in os.listdir('./ImageNet_train'):
    dict[filename] = i
    i += 1


class InceptionV4(nn.Cell):
    def __init__(self, A, B, C):
        super().__init__()
        self.Stem = Stem(3)
        self.inception_A = self.generate_inception_module(384, 384, A, 1)
        self.reduction_A = reduction_A(384)
        self.inception_B = self.generate_inception_module(1024, 1024, B, 2)
        self.reduction_B = reduction_B(1024)
        self.inception_C = self.generate_inception_module(1536, 1536, C, 3)
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

    def generate_inception_module(self, input_channels, output_channels, block_num, block):
        if block == 1:
            layers = nn.SequentialCell([inception_A(input_channels)])
            for i in range(block_num):
                layers = nn.SequentialCell(inception_A(input_channels), layers)
                input_channels = output_channels

        if block == 2:
            layers = nn.SequentialCell([inception_B(input_channels)])
            for i in range(block_num):
                layers = nn.SequentialCell(inception_B(input_channels), layers)
                input_channels = output_channels

        if block == 3:
            layers = nn.SequentialCell([inception_C(input_channels)])
            for i in range(block_num):
                layers = nn.SequentialCell(inception_C(input_channels), layers)
                input_channels = output_channels

        return layers

def train(opt):
    # device_id = int(os.getenv('DEVICE_ID'))
    #
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=False)
    # context.set_context(enable_task_sink=True, device_id=device_id)
    # context.set_context(enable_loop_sink=True)
    # context.set_context(enable_mem_reuse=True)
    #
    # if not opt.do_eval and opt.run_distribute:
    #     context.set_auto_parallel_context(device_num=opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
    #                                       mirror_mean=True, parameter_broadcast=True)
    #     auto_parallel_context().set_all_reduce_fusion_split_indices([107, 160])
    #     init()

    loss = CrossEntropy(smooth_factor=config.label_smooth_factor, num_classes=config.class_num)

    pic, lab = pre()
    dataset = create_dataset(pic, lab, do_train=True)
    net = InceptionV4(4, 7, 3)

    stepsize = 32
    lr = Tensor(get_lr(global_step=0, lr_init=config.lr_init, lr_end=0.0, lr_max=config.lr_max,
                           warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=stepsize,
                           lr_decay_mode='cosine'))

    optt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, config.momentum,
                   config.weight_decay, config.loss_scale)

    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=optt, loss_scale_manager=loss_scale, metrics={'acc'})
    time_cb = TimeMonitor(data_size=stepsize)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    device_id = 0
    if config.save_checkpoint and device_id == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * stepsize,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="resnet", directory='./', config=config_ck)
        cb += [ckpt_cb]

    print("fuck")
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)


if __name__=='__main__':
    train(opt)

    # imagenet = ds.(DATA_DIR)