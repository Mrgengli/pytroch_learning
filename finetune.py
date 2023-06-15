


# ----------------------------------1.这里我们第一步要进行数据初始化---------------------------------------------
import torch
import torchvision
from torchvision import transformers,datasets
from torch.utils.data import Dataset,Dataloader
from torch import Dataloader
from Mydataset import Mydataset
import os
import nn.Module
import nn
import nn.functional as F
import numpy as np

train_path_txt = os.path.join('','')
valid_path_txt = os.path.join('','')
test_path_txt = os.path.join('','')
norm_mean= [22,33,55]
norm_std = [0.12,0.55,0.23]
lr_init = 0.01
num_epoch = 200

transformnorm = transformers.Normalize(norm_mean,norm_std)
train_transformer = transformers.Compose(
    transformers.Resize(32),
    transformers.RandomCrop(32,padding = 4),
    transformers.Totensor(),
    transformers.Normalize(norm_mean,norm_std)
)
valid_transformer = transformers.Compose(
    transformers.Totensor,
    transformnorm
)


# 在这里也可以加载线上的数据集
# train_dataset = datasets.CIFAR10(root = './Data',transformer = train_transformer,train = True,download = True)
# 创建mydataset实例

train_dataset = Mydataset(train_path_txt,transformer = train_transformer)
valid_dataset = Mydataset(valid_path_txt,transformer = valid_transformer)

train_dataloader = Dataloader(dataset = train_dataset,shuffle = True,batch_size = 16)
valid_dataloader = Dataloader(dataset = valid_dataset,shuffle = True,batch_size = 16)


# ------------------------------2.下面要开始定义模型,实例化对象并且权重初始化并把权重加载进去------------------------------------------

class self_Module(nn.Module):
    def __init__(self):
        super(self_Module,self).__init__()  #这里是继承父类的初始化函数，如果不继承父类的初始化函数，那么在调用父类的属性和方法时可能出错
        # 在这里我们每定义一个，相当于定义一个层
        self.conv1 = nn.Conv2d(3,6,5)#3:输入通道 ；6：输出通道；5：卷积窗口大小：5x5
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool1 = nn.MaxPool2d(2,2) # 一个池化核为2X2的池化层
        self.pool2 = nn.MaxPool2d(2,2) # 注意：这里定义了另一个大小的池化层，如果只定义一个，相当所有的池化只共享这一个参数
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #这里使用F.Relu，而不使用nn.Relu的原因：如果使用nn.Relu相当于加了一个层，加了一个层就相当于有可学习的参数
        #nn.Relu它包含了一个偏移量（bias）参数，可以通过反向传播算法对其进行训练优化
        #而F.relu是一个不可学习的函数，可以更加方便地使用
        x = x.view(-1,16*5*5)#这里将x这个tensor进行形状变换，—1表示第一个维度根据第二维度进行变化，变化过程中size不变
        #使用view方法时，该方法会返回一个新的张量，而不会修改原来的张量，因为PyTorch中的张量都是不可变对象。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def weight_init (self):    #这里权重的初始化 都是对他的weight和bias进行初始化
        for m in self.modules:
            if isinstance(m,nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data,0,0.01)
                m.bias.data.zero_()


net = self_Module()  #实例化一个对象

self_Module.weight_init()

pre_module_params = torch.load('')

net_state_dict = net.state_dict()

net_state_dict.update(pre_module_params)

net.load(net_state_dict)


#---------------------------------3.设置设置学习率-------------------------------------

# 在这里我们要对不同的网络层 设置不同的学习率  后面的网络更加接近结果  我们要对后面的学习学习率设置为前面的十倍

#我们首先要将最后一层的参数提取出来
ignore_model = list(map(id,net.fc3.parameters()))
based_model = filter(lambda p:id(p) not in ignore_model,net.parameters())

#对不同的设置不同的学习率
optimizer = optim.SGD([{'params':based_model},
                       {'params':net.fc3.parameters,'lr' :lr_init*10}],
                      lr = lr_init,momentum = 0.9,weight_decay = 0.0001)  #momentum是给梯度方向是施加的向量
#weight_decay为正则化项，默认为0，也就是没有正则化项，设置是用来防止过拟合的

criterion = nn.CrossEntropyLoss()
#设置学习率调整器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 50,gamma = 0.1)
#optimizer这个优化器里面每隔50个epoch学习率乘0.1

#--------------------------------------------开始训练----------------------------------------------------------

#使用 DataLoader 定义了一个训练数据加载器 train_loader，
#其中设定了 batch_size=32 和 shuffle=True，表示每个 mini-batch 中包含 32 个样本，并且在训练过程中打乱数据集顺序。
#在模型的训练循环中，我们使用 train_loader 对象迭代训练数据集，
#获取每一组输入和目标数据 inputs, targets，并执行前向传播、反向传播等操作，进行模型参数的更新。

for epoch in num_epoch:
    loss_sigma = 0.0

    scheduler.step() #更新学习率
    #这里它会自动跟踪epoch，可以根据上面定义的 step_size更新optimizer中的lr，每隔多少个epoch将lr乘gamma

    for i,data in enumerate(train_dataloader): #这里获得的train_dataloader是个可迭代对象，每一轮迭代的是一个batch的数据
        #这里的data的数据结构一个元组，元组中的每个元素是一个二元的元组，分别是输入数据和输入的标签（（），（），...，（））
        inputs ,labels = data
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward() #反向传播计算出梯度
        optimizer.step()  #优化器根据梯度更新参数


        _,predict = torch.max(outputs.detach(),dim = 1)  #outputs中会存储数值信息和梯度信息，outputs.detach()会只保留其中的数值，
        #torch.max函数返回的第一个是一个包含每一行中最大值的tensor，predict是这些最大值的在的索引，也就是类别。
        total = label.size(0)  #？？？？？？用于计算样本数
        correct += (predict == labels).squeeze().sum().numpy()  #这里为什么要进行squeeze()操作？？？？
        loss_sigma += loss.item()  #这句话是干什么？


        # 每10个iteration 打印一次训练信息，
        if i%10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0


# ----------------------------------------------5.绘制混淆矩阵-----------------------------------------------------------
clu_num = len(classname)
conf_mat = np.zeros([clu_num,clu_num])  #初始化混淆矩阵
net.eval()

for i ,data in enumerate(valid_dataloader):
    inputs,labels = data
    outputs = net(inputs)
    _,predicted = torch.nn.max(outputs.detach(),1)
    for j in range(len(batch_size)):
        conf_mat[labels[i],predicted[i]] +=1
        #这里的conf_mat就是混淆矩阵

















