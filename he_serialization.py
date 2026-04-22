"""
Demonstrates the privacy model using TenSEAL serialization:
  - Client holds the secret key and serializes only ciphertext + public context
  - Server deserializes, runs inference, serializes the encrypted result
  - Client deserializes and decrypts — server never sees raw data or secret key
"""
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


# ── Client side ─────────────────────────────────────────────────────────────

def client_encrypt(image_1d):
    """Generate context+keys, encrypt image, return public context bytes + ciphertext bytes."""
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

    enc_x = ts.ckks_vector(context, image_1d)

    # Serialize ciphertext
    ct_bytes = enc_x.serialize()

    # Make a public copy of the context (drops secret key) and serialize
    pub_ctx = context.copy()
    pub_ctx.make_context_public()
    pub_ctx_bytes = pub_ctx.serialize()

    return context, ct_bytes, pub_ctx_bytes


def client_decrypt(context, result_bytes):
    """Deserialize encrypted result and decrypt using the secret key."""
    enc_result = ts.ckks_vector_from(context, result_bytes)
    return enc_result.decrypt()


# ── Server side ──────────────────────────────────────────────────────────────

def server_infer(pub_ctx_bytes, ct_bytes, w1, b1, w2, b2):
    """Deserialize public context + ciphertext, run inference, return encrypted result bytes."""
    pub_ctx = ts.context_from(pub_ctx_bytes)
    enc_x = ts.ckks_vector_from(pub_ctx, ct_bytes)

    enc_out1 = enc_x.matmul(w1) + b1
    enc_sq = enc_out1.square()
    enc_out2 = enc_sq.matmul(w2) + b2

    return enc_out2.serialize()


# ── Main ─────────────────────────────────────────────────────────────────────

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
    image, label = test_dataset[0]
    image_1d = image.view(-1).tolist()

    print("=" * 55)
    print("  Privacy Model: Serialization-Based HE Inference")
    print("=" * 55)

    # Client: encrypt
    print("\n[Client] Generating keys and encrypting input...")
    t0 = time.time()
    context, ct_bytes, pub_ctx_bytes = client_encrypt(image_1d)
    client_enc_time = time.time() - t0
    print(f"[Client] Done. ({client_enc_time:.3f} s)")
    print(f"[Client] Public context size : {len(pub_ctx_bytes)/1024:.1f} KB")
    print(f"[Client] Ciphertext size     : {len(ct_bytes)/1024:.1f} KB")
    print(f"[Client] Sending public context + ciphertext to server...")

    # Server: infer
    print("\n[Server] Received public context and ciphertext.")
    print("[Server] Running encrypted inference (no secret key)...")
    t0 = time.time()
    result_bytes = server_infer(pub_ctx_bytes, ct_bytes, w1, b1, w2, b2)
    server_time = time.time() - t0
    print(f"[Server] Done. ({server_time:.3f} s)")
    print(f"[Server] Encrypted result size: {len(result_bytes)/1024:.1f} KB")
    print(f"[Server] Sending encrypted result back to client...")

    # Client: decrypt
    print("\n[Client] Received encrypted result. Decrypting with secret key...")
    t0 = time.time()
    logits = client_decrypt(context, result_bytes)
    client_dec_time = time.time() - t0
    pred = logits.index(max(logits))
    print(f"[Client] Done. ({client_dec_time:.4f} s)")

    print("\n" + "=" * 55)
    print(f"  Ground truth : {label}")
    print(f"  Prediction   : {pred}  ({'CORRECT' if pred == label else 'WRONG'})")
    print("-" * 55)
    total = client_enc_time + server_time + client_dec_time
    print(f"  Client encrypt : {client_enc_time:.3f} s")
    print(f"  Server infer   : {server_time:.3f} s")
    print(f"  Client decrypt : {client_dec_time:.4f} s")
    print(f"  Total          : {total:.3f} s")
    print("=" * 55)

if __name__ == "__main__":
    main()
