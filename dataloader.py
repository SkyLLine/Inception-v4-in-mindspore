import numpy as np
import mindspore.dataset.transforms.vision.c_transforms as C
import mindspore.dataset as ds
from mindspore.dataset.transforms.vision import Inter
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.common.dtype as mstype
from pathlib2 import Path
import cv2 as cv


# 辅助生产数据集的类，接口不用调用

class Generator():

    def __init__(self, image, label):
        self.image = image
        self.label = label

    def __getitem__(self, item):
        return (np.array(self.image[item]),
                np.array(self.label[item]))  # Notice, tuple of only one element needs following a comma at the end.

    def __len__(self):
        return len(self.image)


# 调用这个函数就行，传入两个list 分别为image，label （label采用1,2,3这种离散标注，而非01矩阵，返回的label也是离散标注）
# 第三个参数是计算使用线程数，根据自己机器情况调整
#
def create_dataset(image, label, do_train, repeat_num=1, batch_size=4, num_parallel_workers=1, drop_remainder=False):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """

    dataset = ds.GeneratorDataset(source=Generator(image, label), column_names=["image", "label"], shuffle=True)

    image_size = 299
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # define map operations
    if do_train:
        trans = [
            # C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
            # C.Decode(),
            C.Resize((299, 299)),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
            # C.Resize(image_size)
        ]
    else:
        trans = [
            # C.Decode(),
            C.Resize((299, 299)),
            # C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)
    #
    dataset = dataset.map(input_columns=["image"], num_parallel_workers=num_parallel_workers, operations=trans)
    dataset = dataset.map(input_columns=["label"], num_parallel_workers=num_parallel_workers, operations=type_cast_op)
    #
    # # apply batch operations
    buffer_size = 10000
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    # apply dataset repeat operation
    dataset = dataset.repeat(repeat_num)

    return dataset


# 测试用函数，，，不用管
def get_image_iterator(dir, trans=False, size=None, begin=None):
    images = []
    name = []
    for cnt, i in enumerate(Path.iterdir(dir)):
        if i.is_file() and i.suffix == ".JPEG":
            if begin and cnt < begin:
                continue
            src = cv.imread(str(i))
            name.append(i.stem)
            # src = cv.bilateralFilter(src, 10, 50, 50)  # 训练1
            # src = cv.bilateralFilter(src, 7, 10, 10) #计算
            # src = cv.bilateralFilter(src, 25, 50, 50)  # 训练2
            # src = cv.bilateralFilter(src, 47, 100, 100)
            # src = cv.pyrMeanShiftFiltering(src, 10, 50)

            # src = cv.GaussianBlur(src, (5, 5), 100)
            # src = cv.GaussianBlur(src, (11, 11), 2.5)
            # src = np.array(cv.cvtColor(src, cv.COLOR_BGR2GRAY)).reshape((128, 128, 1))
            if trans:
                images.append(np.transpose(src, (2, 0, 1)))
            else:
                images.append(src)
            if size:
                if len(images) > size:
                    break
    return images, name


if __name__ == "__main__":
    image, image_name = get_image_iterator(Path("./ImageNet_train"))
    dataset = create_dataset(image, range(len(image)), True)
    # print(image)
    for data in dataset.create_tuple_iterator():  # each data is a sequence
        print(data)
