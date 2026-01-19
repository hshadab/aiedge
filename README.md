# JOLT Atlas

JOLT Atlas is a zero-knowledge machine learning (zkML) framework that extends the [JOLT](https://github.com/a16z/jolt) proving system to support ML inference verification from ONNX models. 

Made with ❤️ by [ICME Labs](https://blog.icme.io/).

<img width="983" height="394" alt="icme_labs" src="https://github.com/user-attachments/assets/ffc334ed-c301-4ce6-8ca3-a565328904fe" />

## Overview

JOLT Atlas enables practical zero-knowledge machine learning by leveraging Just One Lookup Table (JOLT) technology. Traditional circuit-based approaches are prohibitively expensive when representing non-linear functions like ReLU and SoftMax. Lookups eliminate the need for circuit representation entirely.

In JOLT Atlas, we eliminate the complexity that plagues other approaches: no quotient polynomials, no byte decomposition, no grand products, no permutation checks, and most importantly — no complicated circuits.

## Examples

The `examples/` directory contains practical demonstrations of zkML models:

### Article Classification

A text classification model that categorizes articles into business, tech, sport, entertainment, and politics.

```bash
cargo run --release --example article_classification
```

This example:
- Tests model accuracy on sample texts
- Generates a SNARK proof for one classification
- Verifies the proof cryptographically

### Transaction Authorization

A financial transaction authorization model that decides whether to approve or deny transactions based on features like budget, trust score, amount, etc.

```bash
cargo run --release --example authorization
```

This example:
- Tests the model on various transaction scenarios
- Shows authorization decisions with confidence scores
- Generates and verifies a SNARK proof for one transaction

### MediaPipe Text Classifier (Sentiment Analysis)

A sentiment analysis model based on Google's MediaPipe Average Word Embedding classifier. This example proves the MLP portion of the classifier in zero-knowledge.

```bash
cargo run --release --example mediapipe_mlp_proof
```

#### How It Works (Plain English)

**The Problem**: You want to prove that a piece of text is "positive" or "negative" sentiment without revealing your ML model's internal computations to a verifier.

**The Solution**: Zero-knowledge proofs allow you to prove "I ran this neural network correctly and got this result" without the verifier needing to re-run the computation or see the intermediate values.

**What happens step by step**:

1. **Input Preparation**: The text is first converted to a numerical embedding (a list of 16 numbers representing the meaning of the text). This embedding is computed outside the ZK circuit using word embeddings and average pooling.

2. **The MLP Circuit**: The proof covers the Multi-Layer Perceptron (MLP) portion:
   - **Layer 1**: Takes the 16-number embedding, multiplies it by a 16×16 weight matrix, adds a bias → produces 16 numbers
   - **ReLU**: Sets any negative numbers to zero (the non-linear "activation")
   - **Layer 2**: Multiplies by a 16×8 weight matrix, adds bias → produces 8 numbers (padded from 2 classes)

3. **Proof Generation (~5 seconds)**: The prover executes the neural network and generates a cryptographic proof that:
   - All matrix multiplications were done correctly
   - The ReLU was applied correctly
   - The final output genuinely came from running the model on that input

4. **Verification (~2 seconds)**: The verifier checks the proof using only:
   - The input embedding
   - The final classification output
   - The cryptographic proof

   The verifier does NOT need to re-run the neural network or see any intermediate values.

**Why this matters**:
- **Privacy**: Your model weights and intermediate computations stay private
- **Trust**: Anyone can verify you ran the real model, not a fake one
- **Efficiency**: Verification is much faster than re-running the model

**Technical note**: The model dimensions are padded to powers of 2 (required by JOLT's lookup tables), and the output is padded from 2 classes to 8 to meet these requirements.

## Benchmarks

### Transformer (self-attention) profile

Latest run (`cargo run -r -- profile --name self-attention --format default`):

| Stage  | Wall clock |
| ------ | ----------- |
| Prove  | 20.8 s |
| Verify | 143 ms |
| End-to-end CLI run | 25.8 s |

The prover hit a peak allocated footprint of roughly 5.6 GB during sumcheck round 10, which matches what we have seen in the integration test harness. Numbers were collected from this workstation; expect ±10% variance depending on CPU, memory bandwidth.

### MediaPipe MLP Sentiment Classifier

| Stage  | Wall clock |
| ------ | ----------- |
| Prove  | ~5.0 s |
| Verify | ~2.2 s |

Model: Dense(16→16) → ReLU → Dense(16→8), proving sentiment classification on pre-computed word embeddings.

### Cross-project snapshot

Article-classification workload comparison

| Project    | Latency | Notes                        |
| ---------- | ------- | ---------------------------- |
| zkml-jolt  | ~0.7s   | in-tree article-classification bench |
| mina-zkml  | ~2.0s   |                              |
| ezkl       | 4–5s    |                              |
| deep-prove | N/A     | missing gather primitive     |
| zk-torch   | N/A     | missing reduceSum primitive  |

Perceptron MLP baseline (easy sanity workload):

| Project    | Latency | Notes                |
| ---------- | ------- | -------------------- |
| zkml-jolt  | ~800ms  |                      |
| deep-prove | ~200ms  | lacks MCC            |

### How to reproduce locally

```bash
# from repo root
cd zkml-jolt-core

cargo run -r -- profile --name article-classification --format default
cargo run -r -- profile --name self-attention --format default
cargo run -r -- profile --name mlp --format default
```

Add `--format chrome` if you want a tracing JSON for Chrome's `chrome://tracing` viewer instead of plain-text timings.

## Getting Started

1. Clone the repository
2. Install Rust and Cargo
3. Run the examples:
   ```bash
   cargo run --example article_classification
   cargo run --example authorization
   ```

## Acknowledgments

Thanks to the Jolt team for their foundational work. We are standing on the shoulders of giants.