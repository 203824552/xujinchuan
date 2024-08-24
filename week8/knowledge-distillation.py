import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data
torch.manual_seed(0)        # 为CPU设置种子
torch.cuda.manual_seed(0)   # 为GPU设置种子
#定义教师模型
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)     # 卷积层
        self.conv2 = nn.Conv2d(32, 64, 3, 1)    # 卷积层
        self.dropout1 = nn.Dropout2d(0.3)       # dropout
        self.dropout2 = nn.Dropout2d(0.5)       # dropout
        self.fc1 = nn.Linear(9216, 128)         # 全连接层
        self.fc2 = nn.Linear(128, 10)           # 全连接层

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)           # 激活函数
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output
#定义训练教师模型方法
def train_teacher(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)   # 将数据转移到CPU/GPU
        optimizer.zero_grad()   # 优化器将梯度全部置为0
        output = model(data)    # 数据经过模型向前传播
        loss = F.cross_entropy(output, target)  # 计算损失函数
        loss.backward()         # 反向传播
        optimizer.step()        # 更新梯度

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50) # 计算训练进度
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')
#定义教师模型测试方法
def test_teacher(model, device, test_loader):
    model.eval()  # 设置为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 不计算梯度，减少计算量
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)  # 将数据转移到CPU/GPU
            output = model(data)  # 经过模型正向传播得到结果
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 计算总的损失函数
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率索引
            correct += pred.eq(target.view_as(pred)).sum().item()  # pred.eq(target.view_as(pred)) 会返回一个布尔张量，其中每个元素表示预测值是否等于目标值。然后，.sum().item() 会将所有为 True 的元素相加，从而得到正确分类的数量。

    test_loss /= len(test_loader.dataset)  # 计算损失函数

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)
#定义教师模型主函数
def teacher_main():
    epochs = 10
    batch_size = 64
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用的设备类型

    # 导入训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/MNIST', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))  # 数据正则化
                       ])),
        batch_size=batch_size, shuffle=True)

    # 导入测试集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/MNIST', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # 数据正则化
        ])),
        batch_size=1000, shuffle=True)

    model = TeacherNet().to(device)  # 传输经过教师模型网络
    optimizer = torch.optim.Adadelta(model.parameters())  # 使用Adadelta优化器

    teacher_history = []  # 记录教师得到结果的历史
    for epoch in range(1, epochs + 1):
        train_teacher(model, device, train_loader, optimizer, epoch)  # 开始训练模型
        loss, acc = test_teacher(model, device, test_loader)  # 计算损失函数和准确率
        teacher_history.append((loss, acc))  # 记录教师模型得到的历史数据

    torch.save(model.state_dict(), "teacher.pt")  # 保存到权重文件
    return model, teacher_history
#开始训练教师模型
# 训练教师网络
teacher_model, teacher_history = teacher_main()
#定义学生模型网络结构
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 64)       # 全连接层
        self.fc3 = nn.Linear(64, 10)        # 全连接层

    def forward(self, x):
        x = torch.flatten(x, 1)     # 将输入张量沿着第二维度平
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.relu(self.fc3(x))
        return output
#定义知识蒸馏方法，定义知识蒸馏主要是实现其损失函数。
def distillation(y, labels, teacher_scores, temp, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
#定义学生模型训练和测试方法
def train_student_kd(model, device, train_loader, optimizer, epoch):
    model.train()
    trained_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)  # 学生模型前向传播
        teacher_output = teacher_model(data)  # 教师模型前向传播
        teacher_output = teacher_output.detach()  # 切断老师网络的反向传播
        loss = distillation(output, target, teacher_output, temp=5.0, alpha=0.7)  # 计算总损失函数，这里使用的是知识蒸馏的损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')

def test_student_kd(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 计算总的损失函数
            pred = output.argmax(dim=1, keepdim=True)  # 获取最大对数概率索引
            correct += pred.eq(target.view_as(pred)).sum().item()  # 计算准确率
    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)
#定义学生模型主函数
def student_kd_main():
    epochs = 10
    batch_size = 64
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载训练集
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/MNIST', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True)
    # 加载测试集
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/MNIST', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=1000, shuffle=True)
    # 加载学生模型
    model = StudentNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters())

    student_history = []  # 记录学生训练的模型
    for epoch in range(1, epochs + 1):
        train_student_kd(model, device, train_loader, optimizer, epoch)
        loss, acc = test_student_kd(model, device, test_loader)
        student_history.append((loss, acc))

    torch.save(model.state_dict(), "student_kd.pt")
    return model, student_history
student_kd_model, student_kd_history = student_kd_main()