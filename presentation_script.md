# Presentation Script — Privacy-Preserving Inference with CKKS

---

## Slide 1 — Title

Good morning / afternoon, everyone. Today we'll be presenting our project for SE6019: Security and Privacy in AI. Our topic is **Privacy-Preserving Inference with CKKS** — specifically, how we can run a neural network on encrypted data so that the cloud server never sees the actual user input. My name is Wei Xiao, and my partner is Feng Hao.

---

## Slide 2 — Agenda

We'll walk through four main parts today. First, we'll frame the problem and our research question. Then we'll cover the technical design and method. After that, we'll present our experimental results — including some deeper analysis. And finally, we'll discuss limitations and our conclusions.

---

## Slide 3 — Motivation and Scope

So why does this matter? When users send data to a cloud AI service — whether it's a medical image or a handwritten digit — that data has to be transmitted to a remote server for inference. Traditional encryption only protects data *in transit* and *at rest*. The moment the server needs to run the model, the data must be decrypted, which means the server can see everything.

Our project addresses exactly this gap. We built an end-to-end pipeline where inference happens **directly on encrypted data** — the server computes the result without ever decrypting the input. We used the MNIST handwritten digit dataset as our test case.

---

## Slide 4 — Research Question

Our core research question is: **Can CKKS-based encrypted inference preserve prediction correctness with acceptable numerical error, and what is the performance cost?**

And we can already give you the short answer from our experiments: yes, accuracy is fully preserved — 100 out of 100 test samples were classified correctly in both plaintext and ciphertext. The numerical error introduced by CKKS is very small, around 0.031 on average. However, the cost is significant — ciphertext inference is about **6,000 times slower** than plaintext.

---

## Slide 5 — System Design and Workflow

Here's how the system works at a high level. First, we train the model in plaintext as normal. Then, on the client side, the user encrypts their input image using CKKS and sends the ciphertext to the server. The server runs the neural network inference entirely on the encrypted data — it never decrypts anything. The encrypted result is sent back to the client, who then decrypts it locally using their own private key.

One important architectural detail: the **secret key never leaves the client**. The server only receives a public context and the ciphertext. Everything in our implementation runs inside Docker for full reproducibility.

---

## Slide 6 — HE-Friendly Model and CKKS Setup

Now let's look at the model and encryption setup.

For the model, we use what we call a **PolynomialNetwork** — a simple two-layer fully connected network: 784 inputs, 64 hidden units, and 10 output classes. The key design choice is the activation function. Standard activations like ReLU involve conditional operations which are not compatible with homomorphic encryption. So we replace it with **x squared** — a polynomial function that works perfectly in the encrypted domain.

For the CKKS parameters, we use a polynomial modulus degree of 16,384, which gives us enough capacity for our computation. The coefficient modulus is set to [60, 40, 40, 40, 60], and the global scale is 2 to the power of 40. We'll explain why these specific values shortly.

---

## Slide 7 — Key Results (Batch: 100 Samples)

Here are the key numbers from our batch evaluation of 100 MNIST test images.

Both plaintext and ciphertext models achieved **100% accuracy** on this sample. The average plaintext latency is just 0.50 milliseconds per image — extremely fast. The ciphertext latency is 2.94 seconds per image. And the average maximum absolute error between plaintext and ciphertext logits is 0.031 — small enough to never affect the classification outcome.

The bottom line: CKKS preserved classification quality perfectly, but at a very large computational cost.

---

## Slide 8 — Single-Image Consistency Example

To make this more concrete, let's look at a single example. We took the first image in the MNIST test set — ground truth label 7. Running it through our encrypted inference pipeline, the server computed entirely on ciphertext and the prediction came out as 7 — correct.

Looking at the per-class logit comparison, the largest absolute difference between plaintext and ciphertext outputs was just 0.02582, for class 9. Critically, the rank order of all 10 classes was identical — the top class in both cases was class 7, by a wide margin. So the small approximation noise introduced by CKKS did not change the final answer at all.

---

## Slide 9 — Training Pipeline Details

Let me now go a bit deeper into how we built this system. The model was trained on MNIST — 60,000 training images — using the Adam optimizer with a learning rate of 0.001, a batch size of 64, and cross-entropy loss, for 3 epochs. The model is deliberately small: it has only 50,890 parameters in total, split between the two fully connected layers.

The reason for the shallow design is not a limitation — it's intentional. Fewer parameters means fewer multiplicative operations during encrypted inference, which means lower depth consumption in CKKS. This is the fundamental design trade-off in privacy-preserving machine learning.

---

## Slide 10 — HE Inference: Step-by-Step

Here's the exact sequence of operations that happen on the server during inference. There are three steps.

First, we compute the first encrypted fully-connected layer: encrypted x times W1 plus b1. This consumes one level of the modulus chain.

Second, we apply the encrypted square activation, which also consumes one level, followed by automatic relinearization and rescaling.

Third, we compute the second encrypted layer: encrypted squared output times W2 plus b2. This produces the final 10 encrypted logits and consumes the last depth level.

Notice that our coefficient modulus [60, 40, 40, 40, 60] has exactly **three intermediate primes** — which perfectly matches our three-step pipeline. There's no wasted depth, and no budget overflow.

---

## Slide 11 — Numerical Precision Analysis

The CKKS scheme is an *approximate* computation method — it deliberately trades exact precision for the ability to handle floating-point numbers efficiently. After each multiplication, a small amount of rounding noise is introduced, proportional to 2 to the negative 40 — our global scale. Over three levels, this noise accumulates.

However, as we showed, the key insight is that image classification depends on *argmax* — which class has the highest score — not the exact values. As long as the noise doesn't flip the ranking, the classification is correct. And in all 100 of our test cases, the ranking was preserved.

---

## Slide 12 — Latency Overhead

So why is ciphertext inference 6,000 times slower? Let's break it down.

In plaintext, a matrix multiplication takes microseconds. In ciphertext, each matmul on a 16,384-degree polynomial requires 64 ciphertext-plaintext multiplications plus roughly 64 ciphertext rotations for the inner sum — all operating on large polynomial objects. Similarly, the square activation, which is trivial in plaintext, requires a full ciphertext-by-ciphertext multiplication with relinearization.

On top of this, our Docker environment uses CPU only — no AVX-512 or GPU acceleration. 

There are known mitigation strategies: CKKS supports SIMD-style batching to pack multiple inputs into a single ciphertext, hardware acceleration can provide significant speedups, and reducing the polynomial modulus degree would help — at the cost of security or depth.

---

## Slide 13 — Limitations and Next Steps

Our current implementation has a few clear limitations. We only tested on MNIST, which is a simple dataset. The model is intentionally shallow, which limits accuracy on harder tasks. And the inference latency is still far from practical for real-time use.

Looking ahead, the most promising directions are exploring CKKS batching and packing strategies, testing deeper HE-friendly approximations, and comparing CKKS with complementary approaches like Trusted Execution Environments or Multi-Party Computation.

---

## Slide 14 — Work Division

In terms of work division: both Feng Hao and I collaborated on writing the code, engineering the pipeline, and running all the experiments together. Feng Hao took the lead on writing the project report, and I prepared this presentation.

---

## Slide 15 — Final Answer

So, to close the loop on our research question: **yes**, CKKS-based inference successfully protects input confidentiality and preserves prediction correctness, with only a small and inconsequential approximation error. The main practical barrier today is latency overhead.

Privacy-preserving AI inference is technically feasible right now for offline or low-throughput scenarios — for example, batch medical screening where security is paramount. But for real-time applications, significant optimization is still required before it becomes practical at scale.

---

## Slide 16 — References

That's our presentation. The source code is publicly available on GitHub. Thank you — we're happy to take any questions.

---

*[Estimated speaking time: ~12–15 minutes]*
