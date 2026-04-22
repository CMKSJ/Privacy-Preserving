# Presentation Script — Privacy-Preserving Inference with CKKS

---

## Slide 1 — Title

Good morning everyone. Our project is on privacy-preserving AI inference using Homomorphic Encryption. I'm Wei Xiao, and my partner is Feng Hao.

---

## Slide 2 — Agenda

We'll cover the problem motivation, our technical design, experimental results with deeper analysis, and a conclusion.

---

## Slide 3 — Motivation and Scope

In cloud AI services, users must send raw data to a server for inference — traditional encryption can't protect data while it's being processed. Our solution: use Homomorphic Encryption so the server computes directly on encrypted data and never sees the plaintext. We built an end-to-end pipeline on MNIST to demonstrate this.

---

## Slide 4 — Research Question

Can CKKS-based inference preserve accuracy with acceptable error, and at what performance cost? Short answer: yes — 100% accuracy is maintained, with average error of 0.031. But the cost is steep: ciphertext inference is about **6,000× slower**.

---

## Slide 5 — System Design and Workflow

The workflow is: train in plaintext → client encrypts input → server infers on ciphertext → client decrypts result. The secret key never leaves the client. The server only ever sees ciphertext and a public context. Everything runs in Docker for reproducibility.

---

## Slide 6 — HE-Friendly Model and CKKS Setup

We use a two-layer network (784→64→10) with **x² activation** instead of ReLU — because CKKS only supports polynomial operations. CKKS parameters: poly modulus degree 16,384, coefficient modulus [60,40,40,40,60], global scale 2⁴⁰. The three intermediate primes exactly match our three-step inference pipeline.

---

## Slide 7 — Key Results

Over 100 MNIST test images: both plaintext and ciphertext achieved **100% accuracy**. Plaintext latency is 0.50 ms/image; ciphertext is 2.94 s/image. Average max absolute error is 0.031 — too small to affect classification.

---

## Slide 8 — Single-Image Consistency Example

For label 7: encrypted prediction is 7, correct. The largest per-class logit difference was 0.02582 (class 9). Rank order was identical across all 10 classes — CKKS noise never flips the argmax.

---

## Slide 9 — Training Pipeline Details

MNIST, 60k training images, Adam (lr=0.001), batch size 64, 3 epochs, CrossEntropyLoss. The model has only 50,890 parameters — intentionally shallow to minimise multiplicative depth and keep HE computation feasible.

---

## Slide 10 — HE Inference: Step-by-Step

Three operations on ciphertext: (1) matmul+bias for FC1, depth +1; (2) square activation, depth +1, auto-relin and rescale; (3) matmul+bias for FC2, depth +1. The three intermediate primes in [60,40,40,40,60] are fully consumed — no wasted budget.

---

## Slide 11 — Numerical Precision Analysis

CKKS introduces rounding noise ~2⁻⁴⁰ per multiplication, accumulating over 3 levels. For label 7, errors stayed below 0.026 per class. Since classification uses argmax — not absolute values — this noise never changes the prediction.

---

## Slide 12 — Latency Overhead

Plaintext matmul takes microseconds; encrypted matmul requires 64 polynomial multiplications plus ~64 rotations on degree-16,384 polynomials — on CPU only. The result: 6,000× overhead. Mitigations include CKKS SIMD batching, hardware acceleration, or reducing polynomial degree.

---

## Slide 13 — Limitations and Next Steps

Limitations: MNIST only, shallow model, high latency. Future directions: CKKS batching, deeper HE-friendly networks, and comparison with TEE/MPC approaches.

---

## Slide 14 — Work Division

Feng Hao and I jointly implemented the code and ran experiments. Feng Hao wrote the report; I prepared this presentation.

---

## Slide 15 — Final Answer

CKKS inference protects user privacy and preserves accuracy, with negligible approximation error. The barrier is latency — practical today for offline, security-critical scenarios, but not yet for real-time use.

---

## Slide 16 — References

Thank you. Source code is on GitHub. Happy to take questions.

---

*[Estimated speaking time: ~6–8 minutes]*
