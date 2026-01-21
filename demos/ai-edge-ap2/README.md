# AI Edge → AP2 Payment Demo

> **Powered by [Jolt Atlas zkML](https://github.com/ICME-Lab/jolt-atlas)** — Real SNARK proofs for verifiable ML inference

**Real Zero-Knowledge Proofs for Autonomous Agent Spending**

This demo shows how on-device AI agents can make autonomous purchases with cryptographic proof of correct behavior. The zkML proofs ensure that spending classifications are verifiable without re-running inference.

### Key Technologies
- **[Jolt Atlas zkML](https://github.com/ICME-Lab/jolt-atlas)** - Real SNARK proofs of ML inference (~5s prove, ~2s verify)
- **[Google AI Edge](https://ai.google.dev/edge)** - Real MediaPipe text classifier for spending categorization (~1ms inference)
- **[Google AP2](https://github.com/google-agentic-commerce/AP2)** - Real Google Agent Payments Protocol types

## What's Real vs Simulated

| Component | Status | Description |
|-----------|--------|-------------|
| **Google AI Edge** | ✅ REAL | MediaPipe average word embedding classifier (~1-2ms inference) |
| **Google AP2 Types** | ✅ REAL | Official types from google-agentic-commerce/AP2 |
| **zkML Proofs** | ✅ REAL | Jolt Atlas SNARK proofs (real times shown in demo) |
| **Policy Engine** | ✅ REAL | Enterprise spending rule enforcement |
| **Payment Settlement** | ⚠️ SIMULATED | Actual money transfer (sandbox mode) |
| **Gemma LLM Agent** | ⚠️ SIMULATED | The AI agent making purchase decisions |

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  AI Edge Agent  │────▶│ Spending Classifier  │────▶│ Policy Engine   │────▶│ AP2 Payment  │
│ (Gemma/LiteRT)  │     │ (zkML Proved)        │     │ (Enterprise)    │     │ (Google)     │
└─────────────────┘     └──────────────────────┘     └─────────────────┘     └──────────────┘
     │                        │                           │                       │
     │                        │                           │                       │
  Generates              Categorizes &                Checks rules           Payment with
  function call          creates SNARK proof          (limits, vendors)      SpendingProof
  (NOT proved)           (real times in UI)
```

## What Exactly is Being Proved?

### Important Clarification

The zkML does **NOT** prove the on-device Gemma AI. Here's why:

| Component | Size | Proved by zkML? | Why? |
|-----------|------|-----------------|------|
| **Gemma (AI Edge)** | ~2B parameters | ❌ No | Too large - would take hours/days |
| **Spending Classifier** | ~400 parameters | ✅ Yes | Small enough to prove in ~5 seconds |

### The Trust Model

```
TRUSTED (not proved):
├── Google AI Edge (Gemma) runs correctly on device
├── The function call output is authentic
└── Device hasn't been tampered with

PROVED (with zkML):
├── The spending classifier neural network ran correctly
├── The classification output (e.g., "data_services") is genuine
├── The exact model (spending_classifier_v1) was used
└── Anyone can verify without re-running inference
```

### What the Spending Classifier Does

The AI agent (Gemma) outputs a purchase request like:
```json
{
  "function": "purchase_data_service",
  "vendor": "HERE Maps",
  "amount_cents": 1200
}
```

The **spending classifier** (a small MLP neural network) categorizes this into:
- `data_services` - Maps, APIs, data feeds
- `cloud_compute` - GPU, inference
- `saas_licenses` - Software subscriptions
- `blocked` - Unauthorized transactions

The zkML proof guarantees: *"This classifier really did output 'data_services' for this input."*

### Why This Architecture?

Think of it like a **provable expense auditor**:
- The AI makes spending decisions (like an employee)
- The classifier audits those decisions (like an expense system)
- The zkML proves the audit was done correctly (like a cryptographic receipt)

## Real zkML Performance

| Metric | Value |
|--------|-------|
| **Proof Generation** | Real time shown in demo UI |
| **Proof Verification** | Real time shown in demo UI |
| **Commitment Scheme** | Dory (Polynomial Commitment) |
| **Transcript** | Keccak256 |
| **Field** | BN254 (ark-bn254) |
| **Classifier Architecture** | MLP: 16→16→8 with ReLU |

## What the Policy Engine Does

The Policy Engine enforces **enterprise spending rules**:

```
ACME Corp Spending Policy
├── Categories
│   ├── data_services: ✅ Allowed up to $500/day
│   ├── cloud_compute: ✅ Allowed up to $1000/day
│   ├── saas_licenses: ✅ Approved vendors only
│   └── blocked: ❌ Never allowed
│
├── Approved Vendors
│   ├── HERE Maps ✅
│   ├── Google Maps ✅
│   ├── AWS Bedrock ✅
│   └── "SketchyData Inc" ❌
│
└── Rules
    ├── Single transaction limit: $500
    ├── Daily spending limit: $2000
    └── Require approval above: $1000
```

For each purchase request, it checks:
1. **Category allowed?** - Is `data_services` permitted?
2. **Under limit?** - Is $12 under the $500 category limit?
3. **Approved vendor?** - Is HERE Maps on the whitelist?

All checks pass → Payment proceeds with SpendingProof attached.

## Demo Scenarios

### 1. Approved Purchase (Route Data)
```
AI Agent: "I need route data to avoid traffic delay"
Function Call: purchase_data_service($12, HERE Maps)
Classifier: data_services (89% confidence)
zkML Proof: ✅ Generated and verified (real times shown in UI)
Policy: ✅ Category allowed, under limit, approved vendor
Result: Payment processed via AP2 with SpendingProof
```

### 2. Blocked Transfer
```
AI Agent: "Transfer funds to external account"
Function Call: transfer_external($5000, unknown)
Classifier: blocked (95% confidence)
zkML Proof: ✅ Generated and verified (proves correct classification)
Policy: ❌ Category "blocked" is never allowed
Result: Payment rejected, audit logged with cryptographic proof
```

## Running the Demo

### Prerequisites
- Rust nightly (for Jolt Atlas)
- Python 3.x (for AP2 service)

### Quick Start

```bash
# From jolt-atlas root directory
cd demos/ai-edge-ap2

# Start AP2 service (Python FastAPI)
cd ap2_service && source venv/bin/activate && python ap2_service.py &

# Start backend (Rust/Axum with real zkML)
cd .. && RUST_LOG=info cargo run -p ai-edge-ap2-demo --release &

# Start frontend
cd frontend && python3 -m http.server 3000 &
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| **Backend** | 3001 | Rust API with real Jolt Atlas zkML |
| **AP2 Service** | 3002 | Python FastAPI (AP2 protocol simulation) |
| **Frontend** | 3000 | Interactive demo UI |

### Demo URL
Open http://localhost:3000 and press **Space** or click **Run Demo**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/classify` | POST | Classify function call + generate zkML proof |
| `/api/v1/verify` | POST | Verify a zkML proof bundle |
| `/api/v1/policy/:id` | GET | Get spending policy |
| `/api/v1/ap2/pay` | POST | Process payment via AP2 with SpendingProof |
| `/api/v1/transactions` | GET | List transaction history |
| `/api/v1/demo/trigger/:scenario` | POST | Trigger demo scenarios |

## Technical Implementation

### Jolt Atlas Integration

```rust
// Real zkML proof generation
use zkml_jolt_core::jolt::{JoltProverPreprocessing, JoltSNARK};
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;

type PCS = DoryCommitmentScheme;

// Preprocessing (done once at startup, ~1s with cached SRS)
let preprocessing = JoltSNARK::<Fr, PCS, KeccakTranscript>
    ::prover_preprocess(model_fn, 1 << 20);

// Proof generation
let (snark, program_io, _) = JoltSNARK::prove(&preprocessing, model_fn, &input);

// Verification
snark.verify(&(&preprocessing).into(), program_io, None)?;
```

### SpendingProof in AP2 Payment

```json
{
  "ap2_version": "1.0",
  "payment": {
    "amount_cents": 1200,
    "payer_id": "ap2://acme-corp-agent",
    "payee_id": "ap2://here-maps"
  },
  "spending_proof": {
    "classification_proof": "0x4a4f4c545f534e41524b5f50524f4f465f...",
    "policy_attestation": "0x504f4c4943595f4154544553544154494f4e...",
    "model_hash": "sha256:spending_classifier_v1_jolt_atlas",
    "proving_time_ms": 4715,
    "verification_time_ms": 1258
  }
}
```

## Project Structure

```
demos/ai-edge-ap2/
├── backend/
│   └── src/
│       ├── main.rs          # Axum server
│       ├── api/mod.rs       # API handlers
│       ├── zkml/mod.rs      # Real Jolt Atlas zkML prover
│       └── ap2/mod.rs       # AP2 gateway integration
├── ap2_service/
│   └── ap2_service.py       # Python FastAPI AP2 simulation
├── frontend/
│   └── index.html           # Single-page demo UI
└── README.md
```

## Future: Proving the On-Device AI

To actually prove Gemma or other on-device models, potential approaches:

| Approach | Description | Feasibility |
|----------|-------------|-------------|
| **Smaller models** | Use a tiny model (~1M params) that can be proved | Near-term |
| **TEE + zkML** | Run Gemma in secure hardware (SGX/TrustZone), combine attestation with zkML | Medium-term |
| **Recursive proofs** | Break model into chunks, prove each, aggregate | Research |
| **Hardware acceleration** | Custom ZK hardware for faster proving | Long-term |

## Google AI Edge Integration

This demo is designed to integrate with Google's agentic commerce stack:

| Component | Role | Integration |
|-----------|------|-------------|
| **Google AI Edge** | On-device AI (Gemma on LiteRT) | Generates purchase requests |
| **Spending Classifier** | Categorizes requests | Proved by Jolt Atlas zkML |
| **Google A2A** | Agent-to-Agent protocol | Multi-agent coordination |
| **Google AP2** | Agent-to-Payment protocol | Receives SpendingProof |

The zkML proof from Jolt Atlas provides cryptographic evidence that the spending classification was done correctly, enabling trustless autonomous commerce.

## Status

- [x] Real Jolt Atlas zkML proofs (Dory commitment scheme)
- [x] End-to-end demo flow with real proof generation
- [x] Real Google AP2 types integration (from official repo)
- [x] Real Google AI Edge integration (MediaPipe classifier)
- [x] Enterprise policy enforcement
- [x] Integration status API (`/api/v1/integration/status`)
- [ ] Multi-agent A2A coordination demo
- [ ] Spending analytics dashboard

## API Endpoints - AP2 Service

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/integration/status` | GET | Full integration status (real vs simulated) |
| `/api/v1/ai-edge/status` | GET | Google AI Edge (MediaPipe) status |
| `/api/v1/ai-edge/classify` | POST | Classify function call using real MediaPipe |
| `/api/v1/ap2/status` | GET | Google AP2 integration status |
| `/api/v1/payments/process` | POST | Process payment with Google AP2 receipt |
| `/api/v1/demo/quick-pay` | POST | Simplified payment endpoint |
