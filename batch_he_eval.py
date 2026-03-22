import torch
import torch.nn as nn
from torchvision import datasets, transforms
import tenseal as ts
import time

# 1. 定义网络结构
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
    # 2. 加载模型权重与提取网络参数
    model = PolynomialNetwork()
    model.load_state_dict(torch.load("he_friendly_model.pth", weights_only=True))
    model.eval()

    w1 = model.fc1.weight.t().tolist()
    b1 = model.fc1.bias.tolist()
    w2 = model.fc2.weight.t().tolist()
    b2 = model.fc2.bias.tolist()

    # 3. 设置 TenSEAL 同态加密上下文
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

    # 4. 准备测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # 5. 初始化统计变量
    num_tests = 100
    correct_plain = 0
    correct_crypt = 0
    total_plain_time = 0.0
    total_crypt_time = 0.0
    total_max_error = 0.0

    for i in range(num_tests):
        image, label = test_dataset[i]

        # 明文推理测试
        start_plain = time.time()
        with torch.no_grad():
            out_plain = model(image)
            pred_plain = out_plain.argmax(dim=1).item()
        plain_time = time.time() - start_plain
        total_plain_time += plain_time
        
        if pred_plain == label:
            correct_plain += 1

        # 密文推理测试
        image_1d = image.view(-1).tolist()
        enc_x = ts.ckks_vector(context, image_1d)

        start_crypt = time.time()
        enc_out1 = enc_x.matmul(w1) + b1
        enc_sq = enc_out1.square()
        enc_out2 = enc_sq.matmul(w2) + b2
        crypt_time = time.time() - start_crypt
        total_crypt_time += crypt_time

        decrypted_result = enc_out2.decrypt()
        pred_crypt = decrypted_result.index(max(decrypted_result))
        
        if pred_crypt == label:
            correct_crypt += 1

        # 计算明文与密文之间的绝对误差
        out_plain_list = out_plain[0].tolist()
        errors = [abs(p - c) for p, c in zip(out_plain_list, decrypted_result)]
        total_max_error += max(errors) 

        # 进度提示
        if (i + 1) % 10 == 0:
            print(f"已完成 {i + 1}/{num_tests} 张图片测试...")

    print("\n" + "="*45)
    print("批量测试结果统计 (明文 vs 密文) ")
    print("="*45)
    print(f"测试样本总数: {num_tests}")
    print(f"明文模型准确率: {correct_plain / num_tests * 100:.2f}%")
    print(f"密文模型准确率: {correct_crypt / num_tests * 100:.2f}%")
    print("-" * 45)
    print(f"明文平均推理耗时: {total_plain_time / num_tests * 1000:.2f} 毫秒/张")
    print(f"密文平均推理耗时: {total_crypt_time / num_tests:.2f} 秒/张")
    print("-" * 45)
    print(f"密文平均最大绝对误差: {total_max_error / num_tests:.8f}")
    print("="*45)

if __name__ == "__main__":
    main()