import mindspore as ms
import mindspore.nn as nn
import mindspore.ops.operations as operator
import os
from lr_generator import get_lr
from CrossEntropy import CrossEntropy
import argparse
from inception_A import inception_A
from inception_B import inception_B
import numpy as np
from inception_C import inception_C
from network import Stem
from reduction_A import reduction_A
from reduction_B import reduction_B
from reduction_C import reduction_C
import mindspore.dataset as ds
from mindspore import context
from mindspore import Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
import os
import urllib.request
from urllib.parse import urlparse
import gzip
import argparse
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train import Model
from mindspore.common.initializer import TruncatedNormal
import mindspore.dataset.transforms.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.transforms.vision import Inter
from mindspore.nn.metrics import Accuracy
from mindspore.common import dtype as mstype
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits

from mindspore.train.model import Model, ParallelMode
from config import config
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from dataloader import create_dataset


def unzipfile(gzip_path):
    """unzip dataset file
    Args:
        gzip_path: dataset file path
    """
    open_file = open(gzip_path.replace('.gz', ''), 'wb')
    gz_file = gzip.GzipFile(gzip_path)
    open_file.write(gz_file.read())
    gz_file.close()


def download_dataset():
    """Download the dataset from http://yann.lecun.com/exdb/mnist/."""
    print("******Downloading the MNIST dataset******")
    train_path = "./MNIST_Data/train/"
    test_path = "./MNIST_Data/test/"
    train_path_check = os.path.exists(train_path)
    test_path_check = os.path.exists(test_path)
    if train_path_check == False and test_path_check == False:
        os.makedirs(train_path)
        os.makedirs(test_path)
    train_url = {"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                 "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"}
    test_url = {"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"}
    for url in train_url:
        url_parse = urlparse(url)
        # split the file name from url
        file_name = os.path.join(train_path, url_parse.path.split('/')[-1])
        if not os.path.exists(file_name.replace('.gz', '')):
            file = urllib.request.urlretrieve(url, file_name)
            unzipfile(file_name)
            os.remove(file_name)
    for url in test_url:
        url_parse = urlparse(url)
        # split the file name from url
        file_name = os.path.join(test_path, url_parse.path.split('/')[-1])
        if not os.path.exists(file_name.replace('.gz', '')):
            file = urllib.request.urlretrieve(url, file_name)
            unzipfile(file_name)
            os.remove(file_name)


# def create_dataset(data_path, batch_size=32, repeat_size=1,
#                    num_parallel_workers=1):
#     """ create dataset for train or test
#     Args:
#         data_path: Data path
#         batch_size: The number of data records in each group
#         repeat_size: The number of replicated data records
#         num_parallel_workers: The number of parallel workers
#     """
#     # define dataset
#     mnist_ds = ds.MnistDataset(data_path)

#     # define operation parameters
#     resize_height, resize_width = 299, 299
#     rescale = 1.0 / 255.0
#     shift = 0.0
#     rescale_nml = 1 / 0.3081
#     shift_nml = -1 * 0.1307 / 0.3081

#     # define map operations
#     resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Resize images to (32, 32)
#     rescale_nml_op = CV.Rescale(rescale_nml, shift_nml) # normalize images
#     rescale_op = CV.Rescale(rescale, shift) # rescale images
#     hwc2chw_op = CV.HWC2CHW() # change shape from (height, width, channel) to (channel, height, width) to fit network.
#     type_cast_op = C.TypeCast(mstype.int32) # change data type of label to int32 to fit network

#     # apply map operations on images
#     mnist_ds = mnist_ds.map(input_columns="label", operations=type_cast_op, num_parallel_workers=num_parallel_workers)
#     mnist_ds = mnist_ds.map(input_columns="image", operations=resize_op, num_parallel_workers=num_parallel_workers)
#     mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_op, num_parallel_workers=num_parallel_workers)
#     mnist_ds = mnist_ds.map(input_columns="image", operations=rescale_nml_op, num_parallel_workers=num_parallel_workers)
#     mnist_ds = mnist_ds.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=num_parallel_workers)

#     # apply DatasetOps
#     buffer_size = 10000
#     mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
#     mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
#     mnist_ds = mnist_ds.repeat(repeat_size)

#     return mnist_ds

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


class InceptionV4(nn.Cell):
    def __init__(self, A, B, C):
        super().__init__()
        self.Stem = Stem(3)
        self.inception_A = inception_A(384)
        self.reduction_A = reduction_A(384)
        self.inception_B = inception_B(1024)
        self.reduction_B = reduction_B(1024)
        self.inception_C = inception_C(1536)
        self.avgpool = nn.AvgPool2d(8)

        #### reshape成2维
        self.dropout = nn.Dropout(0.8)
        self.linear = nn.Dense(1536, 1000, activation='softmax')

    def construct(self, x):
        x = self.Stem(x)
        x = self.inception_A(x)
        x = self.inception_A(x)
        x = self.inception_A(x)
        x = self.inception_A(x)
        x = self.reduction_A(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.inception_B(x)
        x = self.reduction_B(x)
        x = self.inception_C(x)
        x = self.inception_C(x)
        x = self.inception_C(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(-1, 35 * 35 * 384)
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

    mnist_path = "./MNIST_Data"
    download_dataset()
    dataset = create_dataset(os.path.join(mnist_path, "train"), 32, 1)
    net = InceptionV4(4, 7, 3)
    # net = LeNet5()

    stepsize = 32
    lr = 0.01

    optt = nn.Momentum(net.trainable_params(), lr, momentum=0.9)

    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # save the network model and parameters for subsequence fine-tuning
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    # group layers into an object with training and evaluation features

    net_loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')

    model = Model(net, net_loss, optt, metrics={"Accuracy": Accuracy()})

    model.train(config.epoch_size, dataset, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=False)


#########################################
def weight_variable():
    """Weight initial."""
    return TruncatedNormal(0.02)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Conv layer weight initial."""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """Fc layer weight initial."""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


class LeNet5(nn.Cell):
    """Lenet network structure."""

    # define the operator required
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def ans():
    context.set_context(mode=context.GRAPH_MODE)
    # net = InceptionV4(4, 7, 3)
    pic = []
    s = ms.Tensor(np.ones((1, 1, 299, 299)), ms.float32)
    a = s
    b = s
    pic.append(s)
    pic.append(a)
    pic.append(b)
    lab = []
    lab.append(0)
    lab.append(1)
    lab.append(2)
    print("start")
    ds = create_dataset(pic, lab, True)
    for data in ds.create_tuple_iterator():
        print(data[0].shape)
    # stepsize = 32
    # lr = 0.01
    #
    # optt = nn.Momentum(net.trainable_params(), lr, momentum=0.9)
    #
    # config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # # save the network model and parameters for subsequence fine-tuning
    # ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
    # # group layers into an object with training and evaluation features
    #
    # net_loss = SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True, reduction='mean')
    #
    # model = Model(net, net_loss, optt, metrics={"Accuracy": Accuracy()})
    #
    # model.train(config.epoch_size, ds, callbacks=[ckpoint_cb, LossMonitor()], dataset_sink_mode=False)


if __name__ == '__main__':
    ans()