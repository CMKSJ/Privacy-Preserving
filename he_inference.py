import torch
import torch.nn as nn
from torchvision import datasets, transforms
import tenseal as ts
import time

# 1. 重新定义网络结构
class PolynomialNetwork(nn.Module):
    def __init__(self):
        super(PolynomialNetwork, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

def main():
    # 服务端加载模型
    model = PolynomialNetwork()
    model.load_state_dict(torch.load("he_friendly_model.pth", weights_only=True))
    model.eval()

    w1 = model.fc1.weight.t().tolist()
    b1 = model.fc1.bias.tolist()
    w2 = model.fc2.weight.t().tolist()
    b2 = model.fc2.bias.tolist()

    print("[客户端] 生成同态加密上下文和密钥")
    
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

    print("[客户端] 获取测试图片并加密")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    image, label = test_dataset[0]
    image_1d = image.view(-1).tolist()

    enc_x = ts.ckks_vector(context, image_1d)
    print(f"[客户端] 加密完成, 真实标签是: {label}")

    print("[服务端] 收到密文")
    start_time = time.time()

    enc_out1 = enc_x.matmul(w1) + b1
    
    enc_sq = enc_out1.square()
    
    enc_out2 = enc_sq.matmul(w2) + b2

    server_time = time.time() - start_time
    print(f"[服务端] 密文计算完成！耗时: {server_time:.2f} 秒")

    print("[客户端] 接收到加密结果，使用本地私钥解密")
    decrypted_result = enc_out2.decrypt()

    # 找出解密后 10 个概率值中最大的那个，即为预测结果
    predicted_class = decrypted_result.index(max(decrypted_result))
    
    print("\n==============================")
    print(f"真实标签: {label}")
    print(f"密文预测结果: {predicted_class}")
    print("==============================")

if __name__ == "__main__":
    main()