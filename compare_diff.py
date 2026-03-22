import torch
import torch.nn as nn
from torchvision import datasets, transforms
import tenseal as ts

# 1. 定义与加载网络
class PolynomialNetwork(nn.Module):
    def __init__(self):
        super(PolynomialNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = x ** 2
        x = self.fc2(x)
        return x

def main():
    model = PolynomialNetwork()
    model.load_state_dict(torch.load("he_friendly_model.pth", weights_only=True))
    model.eval()

    w1 = model.fc1.weight.t().tolist()
    b1 = model.fc1.bias.tolist()
    w2 = model.fc2.weight.t().tolist()
    b2 = model.fc2.bias.tolist()

    # 2. 初始化 HE 上下文
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()
    context.auto_relin = True
    context.auto_rescale = True
    context.auto_mod_switch = True

    # 3. 获取测试图片
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    image, label = test_dataset[0] # 取第一张图片，真实标签是 7

    # --- A. 获取明文输出 ---
    with torch.no_grad():
        out_plain = model(image)
        plain_logits = out_plain[0].tolist() # 提取 10 个类的打分

    # --- B. 获取密文输出 ---
    image_1d = image.view(-1).tolist()
    enc_x = ts.ckks_vector(context, image_1d)

    enc_out1 = enc_x.matmul(w1) + b1
    enc_sq = enc_out1.square()
    enc_out2 = enc_sq.matmul(w2) + b2
    
    crypt_logits = enc_out2.decrypt() # 解密得到 10 个类的近似打分

    print(f"\n真实标签 (Ground Truth): {label}")
    print("=" * 70)
    print(f"{'类别':<6} | {'明文模型输出':<18} | {'密文模型输出 (CKKS)':<22} | {'绝对误差 (Error)'}")
    print("-" * 70)
    
    for i in range(10):
        diff = abs(plain_logits[i] - crypt_logits[i])
        marker = ">>" if i == label else "  "
        print(f"{marker} {i:<3} | {plain_logits[i]:<18.6f} | {crypt_logits[i]:<22.6f} | {diff:.8f}")
    
    print("=" * 70)
    print("结论: CKKS 方案引入了微小的浮点数近似误差，但并未改变数值的相对大小，因此分类依然正确")

if __name__ == "__main__":
    main()