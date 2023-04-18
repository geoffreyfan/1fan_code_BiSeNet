# 导入库
import os
from torch.utils.data import Dataset

import warnings
import torch
warnings.filterwarnings("ignore")   # 忽略警告消息
from PIL import Image
import numpy as np
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A

torch.manual_seed(17)
# 设置 CPU 生成随机数的 种子 ，方便下次复现实验结果。
# 为 CPU 设置 种子 用于生成随机数，以使得结果是确定的。
# 当你设置一个随机种子时，接下来的随机算法生成数根据当前的随机种子按照一定规律生成。
# 随机种子作用域是在设置时到下一次设置时。要想重复实验结果，设置同样随机种子即可。


class CamVidDataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, masks_dir):      # import albumentations as A
        super(CamVidDataset, self).__init__()
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),   # 把做好了transform的数据转化成tensor
        ])
        self.ids = os.listdir(images_dir)
        # os.listdir(path)中有一个参数，就是传入相应的路径，将会返回那个目录下的所有文件名。这个函数在遍历文件操作时很常用。

        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('RGB'))
        image = self.transform(image=image, mask=mask)

        return image['image'], image['mask'][:, :, 0]

    def __len__(self):
        return len(self.ids)


