import torch
import torch.nn as nn
from torchvision import datasets, transforms
import tenseal as ts
import time

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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    num_samples = 5
    t_keygen = t_encrypt = t_infer = t_decrypt = 0.0
    ctx_bytes = ct_bytes = result_bytes = 0

    for i in range(num_samples):
        image, label = test_dataset[i]

        # Phase 1: key generation
        t0 = time.time()
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=16384,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
        )
        context.global_scale = 2 ** 40
        context.generate_galois_keys()
        context.generate_relin_keys()
        context.auto_relin = True
        context.auto_rescale = True
        context.auto_mod_switch = True
        t_keygen += time.time() - t0

        # Phase 2: encryption
        image_1d = image.view(-1).tolist()
        t0 = time.time()
        enc_x = ts.ckks_vector(context, image_1d)
        t_encrypt += time.time() - t0

        # Serialized sizes (first sample only)
        if i == 0:
            ctx_bytes = len(context.serialize())
            ct_bytes = len(enc_x.serialize())

        # Phase 3: server-side inference (public context only)
        pub_context = context.copy()
        pub_context.make_context_public()
        enc_x_pub = ts.ckks_vector_from(pub_context, enc_x.serialize())

        t0 = time.time()
        enc_out1 = enc_x_pub.matmul(w1) + b1
        enc_sq = enc_out1.square()
        enc_out2 = enc_sq.matmul(w2) + b2
        t_infer += time.time() - t0

        if i == 0:
            result_bytes = len(enc_out2.serialize())

        # Phase 4: decryption (back on client with secret key)
        enc_result = ts.ckks_vector_from(context, enc_out2.serialize())
        t0 = time.time()
        decrypted = enc_result.decrypt()
        t_decrypt += time.time() - t0

        pred = decrypted.index(max(decrypted))
        print(f"[{i+1}/{num_samples}] label={label}, pred={pred}, correct={pred==label}")

    n = num_samples
    total = (t_keygen + t_encrypt + t_infer + t_decrypt) / n

    print("\n" + "=" * 50)
    print("端到端时延拆分 (avg over {} samples)".format(n))
    print("=" * 50)
    print(f"  密钥生成   : {t_keygen/n:.3f} s")
    print(f"  输入加密   : {t_encrypt/n:.4f} s")
    print(f"  服务端推理 : {t_infer/n:.3f} s")
    print(f"  结果解密   : {t_decrypt/n:.4f} s")
    print(f"  总计       : {total:.3f} s")
    print("-" * 50)
    print(f"  公开上下文大小 : {ctx_bytes/1024:.1f} KB")
    print(f"  输入密文大小   : {ct_bytes/1024:.1f} KB")
    print(f"  输出密文大小   : {result_bytes/1024:.1f} KB")
    print("=" * 50)

if __name__ == "__main__":
    main()
