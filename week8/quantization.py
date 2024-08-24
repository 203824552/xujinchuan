import torchvision.models as models
model = models.resnet18(pretrained=True)
model.eval()
for param in model.parameters():
    param.requires_grad = False
from torch.quantization import QuantizedLinear

#量化模拟器
class QuantizedModel(nn.Module):
    def __init__(self):
        super(QuantizedModel, self).__init__()
        self.fc = QuantizedLinear(10, 10, dtype=torch.qint8)

    def forward(self, x):
        return self.fc(x)

#伪量化
from torch.quantization import QuantStub, DeQuantStub, fake_quantize, fake_dequantize

class FakeQuantizedModel(nn.Module):
    def __init__(self):
        super(FakeQuantizedModel, self).__init__()
        self.fc = nn.Linear(10, 10)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = fake_quantize(x, dtype=torch.qint8)
        x = self.fc(x)
        x = fake_dequantize(x, dtype=torch.qint8)
        x = self.dequant(x)
        return x

#量化模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def evaluate(model, criterion, test_loader):
    model.eval()
    total, correct = 0, 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy


# 使用伪量化评估模型性能
model = SimpleCNN()
model.eval()
accuracy = evaluate(model, criterion, test_loader)
print('Pre-quantization accuracy:', accuracy)

# 应用伪量化
model = FakeQuantizedModel()
accuracy = evaluate(model, criterion, test_loader)
print('Post-quantization accuracy:', accuracy)