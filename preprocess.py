import mindspore as ms
import os

dict = {}
i = 0

for filename in os.listdir('./ImageNet_train'):
    dict[filename] = i
    i += 1