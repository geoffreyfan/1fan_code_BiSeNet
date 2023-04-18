
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="1"    # 也就是说这个东西不能出现在其它py文件里面

from dataset import CamVidDataset
from torch.utils.data import DataLoader
from BiSeNetV1 import BiSeNet
import torch
import matplotlib.pyplot as plt


from d2l import torch as d2l
import torch.nn as nn
import pandas as pd
import numpy as np
device = torch.device('cuda')
model = BiSeNet(num_classes=32)
model = model.to(device)
# 载入预训练模型
# model.load_state_dict(torch.load(r"BiSeNetV1_100.pth"), strict=False)
# 直接使用load_state_dict提供的参数strict=False，网络结构名字一致的会被导入，不一致的会被舍弃


# 损失函数选用多分类交叉熵损失函数
lossf = nn.CrossEntropyLoss(ignore_index=255)

# 使用ADE20K数据集进行验证的分割算法，因这个数据集是exausted annotated，
# 也就是图像中的每个像素都标注了类别，因此背景只占了很少的一部分，
# 因此训练时会设置ignore_index=255,在商汤的框架mmsegmentation中的ade20k.py中的educe_zero_label=True参数，
# 正是为了实现ignore index忽略背景

# 选用adam优化器来训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)

# 训练50轮
epochs_num = 100


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, scheduler,     # trainer 就是对应优化器
               devices=d2l.try_all_gpus()):  # devices=d2l.try_all_gpus() 获取GPU列表
    timer, num_batches = d2l.Timer(), len(train_iter)    # batch_size= 8时，num_batches = 45 ； batch_size= 16时，num_batches = 22
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1], legend=['train loss', 'train acc', 'test acc'])
    # Animator是一个基于终端的python动画库
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])    # 多GPU训练，将最终结果汇集到第0块GPU上

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)   # 创建四个个单位，储存训练损失，训练准确度，实例数，特点数
        print(enumerate(train_iter))    # 答应结果为 <enumerate object at 0x0000022624F47D80>
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(
                net, features, labels.long(), loss, trainer, devices)               # trainer就是优化器，optimizer
            fan1 = labels.shape[0]           # labels = (16, 224, 224) 第一维度代表，batch数
            fan2 = labels.numel()            # labels.numel() = 16 * 224 * 224 = 802,816 代表这批数据的像素点的总个数
            metric.add(l, acc, labels.shape[0], labels.numel())    #  img.shape[0]：图像的垂直尺寸（高度）
            # 训练损失，训练准确度，样本数，labels.numel()张量元素个数,numel就是"number of elements"的简写。
            # numel()可以直接返回int类型的元素个数
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:         # num_batches // 1   后面1这个数据说明了1个epoch保存一次
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],    # train loss = metric[0] / metric[2]  train acc = metric[1] / metric[3]
                              None))
        print(test_iter)
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)   # d2l.evaluate_accuracy_gpu()
        animator.add(epoch + 1, (None, None, test_acc))
        if np.mod(epoch + 1, 99) == 0:
            plt.gcf().set_size_inches(4.5, 4)    # 更改图片的大小
            plt.savefig("result.png", dpi=300)   # dpi 是像素
        animator.show()     # 添加


        # optimizer.step()通常用在每个mini - batch之中，而scheduler.step()
        # 通常用在epoch里面, 但是不绝对，可以根据具体的需求来做。只有用了optimizer.step()，模型才会更新，而scheduler.step()
        # 是对lr进行调整。
        scheduler.step()
        # 在scheduler的step_size表示scheduler.step()
        # 每调用step_size次，对应的学习率就会按照策略调整一次。所以如果scheduler.step()
        # 是放在mini - batch里面，那么step_size指的是经过这么多次迭代，学习率改变一次。
        print(
            f"epoch {epoch + 1} --- loss {metric[0] / metric[2]:.3f}"
            f" ---  train acc {metric[1] / metric[3]:.3f}"
            f" --- test acc {test_acc:.3f} --- cost time {timer.sum()}")

        # ---------保存训练数据---------------
        df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch + 1)
        time_list.append(timer.sum())

        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df['time'] = time_list
        df.to_excel("savefile/BiSeNetV1.xlsx")
        # ----------------保存模型-------------------
        if np.mod(epoch + 1, 10) == 0:    # mod(a,b) 求a/b的余数
            torch.save(model.state_dict(), f'BiSeNetV1_{epoch + 1}.pth')
            # model.state_dict() 返回的则是一个字典 {key:value}，key 是网络层名称，value 则是该层的参数。

# # 设置数据集路径
# DATA_DIR = r'CamVid'  # 根据自己的路径来设置
# x_train_dir = os.path.join(DATA_DIR, 'train')
# y_train_dir = os.path.join(DATA_DIR, 'trainannot')
# x_valid_dir = os.path.join(DATA_DIR, 'val')
# y_valid_dir = os.path.join(DATA_DIR, 'valannot')

# 设置数据集路径
DATA_DIR = r'CamVid'  # 根据自己的路径来设置
x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')
x_valid_dir = os.path.join(DATA_DIR, 'test')
y_valid_dir = os.path.join(DATA_DIR, 'testannot')

train_dataset = CamVidDataset(                    # from Data import CamVidDataset
    x_train_dir,       # images_dir
    y_train_dir,       # masks_dir
)
val_dataset = CamVidDataset(
    x_valid_dir,
    y_valid_dir,
)
# 这个batch_size的大小是要根据图片的数量设定的，因为要用图片的数量除以batch_size
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)   # from torch.utils.data import DataLoader
print(train_loader)          # 有367张图片，经过DataLoader变为 batch_size为16的22个BatchSampler
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, drop_last=True)
print(val_loader)

# 打印的两组数据  ：
# <torch.utils.data.dataloader.DataLoader object at 0x000001A11B812700>
# <torch.utils.data.dataloader.DataLoader object at 0x000001A122EAD730>

train_ch13(model, train_loader, val_loader, lossf, optimizer, epochs_num, scheduler)


