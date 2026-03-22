import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# 1. 定义适合同态加密的网络结构
class PolynomialNetwork(nn.Module):
    def __init__(self):
        super(PolynomialNetwork, self).__init__()
        # 为了降低后续密文矩阵乘法的计算延迟，网络被设计得很浅且参数较少
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        # 展平 28x28 的图像
        x = x.view(-1, 784) 
        
        # 第一层全连接
        x = self.fc1(x)
        
        # 使用平方函数作为激活函数
        x = x ** 2 
        
        x = self.fc2(x)
        return x

def train_model():
    # 2. 准备 MNIST 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 3. 初始化模型、损失函数和优化器
    model = PolynomialNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 训练模型
    epochs = 3
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx} | Loss: {loss.item():.4f}")
        print(f"Epoch {epoch+1} 耗时: {time.time() - start_time:.2f} 秒")
                
    # 5. 保存模型权重
    torch.save(model.state_dict(), "he_friendly_model.pth")

if __name__ == "__main__":
    train_model()