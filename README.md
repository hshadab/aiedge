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

### Enterprise Agentic Spending Guardrails

Cryptographic policy enforcement for B2B AI agents. This demo shows how to extend [Google AI Edge function calling](https://ai.google.dev/edge/mediapipe/solutions/genai/function_calling) with zero-knowledge spending guardrails.

```bash
# First, build the spending classifier model
python3 scripts/build_spending_classifier.py

# Run the demo
cargo run --release --example agentic_spending_guardrails
```

**Interactive Demo**: See [demos/ai-edge-ap2](demos/ai-edge-ap2) for a full web UI integrating Google AI Edge, zkML proofs, and Google AP2 payments.

#### The Problem

Enterprise AI agents (procurement bots, fleet robots, supply chain AI) need to autonomously purchase cloud resources, API credits, and vendor services. But how do you ensure they follow corporate spending policies without trusting the AI?

#### The Solution: zkML Spending Proofs

Every spending decision generates a cryptographic proof that:
1. The intent was correctly classified by an approved ML model
2. The spend complies with corporate procurement policy
3. Any auditor can verify without re-running the model

```
┌─────────────────────────────────────────────────────────────────┐
│                    B2B AGENT SPENDING FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  On-Device LLM         zkML Classifier        Policy Engine      │
│  (Gemma/LiteRT)        (this demo)            (this demo)        │
│  ───────────────       ────────────────       ──────────────     │
│                                                                  │
│  Agent emits:          Classifies intent:     Checks compliance: │
│  purchase_api_credits  → "data_services"      → Category allowed │
│  vendor: clearbit      → Proof generated      → Under budget     │
│  amount: $150                                 → Vendor approved   │
│                                                                  │
│           Only if BOTH proofs verify → Payment executes          │
└─────────────────────────────────────────────────────────────────┘
```

#### Spending Categories

| Category | Description | Per-Transaction Limit |
|----------|-------------|----------------------|
| `data_services` | API subscriptions, market data feeds | $500 |
| `cloud_compute` | GPU/CPU rental, inference endpoints | $1,000 |
| `logistics_data` | Route optimization, fleet APIs | $250 |
| `saas_licenses` | Enterprise software subscriptions | $2,000 |
| `blocked` | Unauthorized procurement | $0 (always denied) |

#### Sample B2B Spending Receipt

```json
{
  "receipt_id": "rcpt_188c3f3d88d24ea5",
  "organization_id": "org_acme_corp",
  "agent_id": "agent_fleet_001",
  "function_call": {
    "function_name": "purchase_api_credits",
    "vendor_id": "vendor:clearbit_api",
    "amount_cents": 15000,
    "cost_center": "CC-4521-SALES"
  },
  "zkml_classification": {
    "category": "data_services",
    "proof_verified": true
  },
  "policy_evaluation": {
    "allowed": true,
    "checks_passed": ["category_allowed", "category_limit", "vendor_approved", "monthly_budget"]
  },
  "payment_status": {
    "authorized": true,
    "payment_rail": "x402_usdc"
  },
  "proof_metadata": {
    "proving_time_ms": 3447,
    "verification_time_ms": 1433,
    "proof_system": "jolt_dory"
  }
}
```

#### Enterprise Value Proposition

**For Procurement/Finance:**
- AI agents operate within pre-approved spending policies
- Every transaction has cryptographic proof of compliance
- Audit trail that can't be forged or disputed

**For IT/Security:**
- Classification model runs on-device (no data exfiltration)
- Proofs verify without exposing model weights
- Vendor allowlists enforced at cryptographic level

**For Compliance/Legal:**
- Every spending decision is auditable
- Proofs provide non-repudiation
- Policy enforcement is mathematically verifiable

#### Integration with Google AI Edge & Agentic Commerce

This demo is designed to integrate with:
- [Google AI Edge Function Calling](https://ai.google.dev/edge/mediapipe/solutions/genai/function_calling) - On-device LLM emits structured function calls
- [Google A2A Protocol](https://blog.google/products/ads-commerce/agentic-commerce-ai-tools-protocol-retailers-platforms/) - Agent-to-agent commerce
- [x402 Payment Protocol](https://www.x402.org/) - Machine-to-machine payments with USDC
- [SpendingProofs](https://www.spendingproofs.com/) - Cryptographic policy enforcement

**The key insight**: Instead of trusting AI agent decisions, you VERIFY them cryptographically before releasing funds.

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

### Enterprise Spending Guardrails

| Stage  | Wall clock |
| ------ | ----------- |
| Prove  | ~3.5 s |
| Verify | ~1.4 s |

Model: Dense(16→16) → ReLU → Dense(16→8), classifying B2B procurement intents into spending categories.

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