import torch
import torchvision.models as models
# 加载预训练的ResNet18模型
model = models.resnet18(pretrained=True)
#Global Magnitude Pruning（全局幅度剪枝）
from torch.nn.utils.prune import global_unstructured
# 定义剪枝比例
pruning_rate = 0.5
# 对模型的全连接层进行剪枝
def prune_model(model, pruning_rate):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            global_unstructured(module, pruning_dim=0, amount=pruning_rate)

prune_model(model, pruning_rate)
# 查看剪枝后的模型结构
print(model)
#再进行重新训练和微调即可
