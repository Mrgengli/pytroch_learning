```
def set_parameter_requires_grad(model, feature_extracting):
    """
    该函数用于将模型所有的梯度改为不可变
    :param model:要修改的模型
    :param feature_extracting:是否要改为不可变
    :return:
    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
```

上面的feature_extracting用于设置是否冻结参数   
在上面的for param in model.parameters()的迭代中它的迭代顺序是：  
考虑一个卷积神经网络（ConvNet）模型，它包括卷积层、全连接层和其他类型的层。当你使用 model.parameters() 迭代器时，它将**按照以下规则逐层迭代参数**：

1.先迭代卷积层的权重和偏置参数。  
2.然后是全连接层的权重和偏置参数。  
3.最后是其他类型的层，如果有的话。  

要**查看模型中的每个参数以及它们的值**，你可以使用以下代码来迭代参数并打印它们的名称和值：  
```
for name, param in model.named_parameters():
    print(f"Parameter Name: {name}")
    print(f"Parameter Value: {param.data}")
```

这段代码使用 model.named_parameters() 方法，它返回一个迭代器，每次迭代都会返回参数的名称和对应的值。你可以将这段代码添加到你的 Python 脚本中，然后运行它，就可以查看每个参数的名称和值。
