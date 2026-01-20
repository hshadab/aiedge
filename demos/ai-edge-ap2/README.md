# AI Edge → AP2 Payment Demo

**Real Zero-Knowledge Proofs for Autonomous Agent Spending**

This demo demonstrates end-to-end integration of:
- **Jolt Atlas ZKML** - Real SNARK proofs of ML inference (~5s proving, ~1.3s verification)
- **Google AI Edge** - On-device AI agents (Gemma on LiteRT)
- **Google AP2** - Agent-to-Payment protocol with SpendingProof

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  AI Edge Agent  │────▶│ Jolt Atlas ZKML      │────▶│ Policy Engine   │────▶│ AP2 Payment  │
│ (Gemma/LiteRT)  │     │ (Real SNARK Proofs)  │     │ (Enterprise)    │     │ (Google)     │
└─────────────────┘     └──────────────────────┘     └─────────────────┘     └──────────────┘
     Function Call      Classification +            Policy Compliance       Payment with
                        Dory Commitment             Verification            SpendingProof
                        ~5s prove, ~1.3s verify
```

## Real ZKML Performance

| Metric | Value |
|--------|-------|
| **Proof Generation** | ~4.7 seconds |
| **Proof Verification** | ~1.3 seconds |
| **Commitment Scheme** | Dory (Polynomial Commitment) |
| **Transcript** | Keccak256 |
| **Field** | BN254 (ark-bn254) |
| **Model Architecture** | MLP: 16→16→8 with ReLU |

## What This Demo Proves

### Cryptographic Guarantees
1. **Verifiable Classification** - SNARK proof that the neural network classified the function call
2. **Model Integrity** - Proof tied to specific model hash (spending_classifier_v1_jolt_atlas)
3. **Non-repudiation** - Anyone can verify the proof without re-running inference

### Enterprise Value
- **Policy Enforcement** - Spending categories, vendor whitelists, amount limits
- **Audit Trail** - Every autonomous purchase has cryptographic proof of compliance
- **Zero Trust** - Proofs are verified independently, no trust in the agent

## Spending Categories

| Category | Description | Example Vendors |
|----------|-------------|-----------------|
| `data_services` | API calls, data feeds, routing | HERE Maps, Google Maps, Clearbit |
| `cloud_compute` | GPU, inference, compute | AWS Bedrock, Lambda, Azure |
| `saas_licenses` | Software subscriptions | Datadog, Splunk, PagerDuty |
| `blocked` | Prohibited transactions | External transfers, withdrawals |

## Demo Scenarios

### 1. Approved Purchase (Route Data)
```
Agent: "Purchase route optimization data from HERE Maps"
Classification: data_services (89% confidence)
ZKML Proof: Generated in ~4700ms, verified in ~1300ms
Policy: ✅ Under $500 limit, approved vendor
Result: Payment processed via AP2 with SpendingProof
```

### 2. Blocked Transfer
```
Agent: "Transfer $5000 to external account"
Classification: blocked (95% confidence)
ZKML Proof: Generated and verified (proves correct classification)
Policy: ❌ Category blocked by enterprise policy
Result: Payment rejected, audit logged with proof
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

# Start backend (Rust/Axum with real ZKML)
cd .. && RUST_LOG=info cargo run -p ai-edge-ap2-demo --release &

# Start frontend
cd frontend && python3 -m http.server 3000 &
```

### Services

| Service | Port | Description |
|---------|------|-------------|
| **Backend** | 3001 | Rust API with real Jolt Atlas ZKML |
| **AP2 Service** | 3002 | Python FastAPI (AP2 protocol simulation) |
| **Frontend** | 3000 | Interactive demo UI |

### Demo URL
Open http://localhost:3000 and press **Space** or click **Run Demo**

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/classify` | POST | Classify function call + generate ZKML proof |
| `/api/v1/verify` | POST | Verify a ZKML proof bundle |
| `/api/v1/policy/:id` | GET | Get spending policy |
| `/api/v1/ap2/pay` | POST | Process payment via AP2 with SpendingProof |
| `/api/v1/transactions` | GET | List transaction history |
| `/api/v1/demo/trigger/:scenario` | POST | Trigger demo scenarios |

## Technical Implementation

### Jolt Atlas Integration

```rust
// Real ZKML proof generation
use zkml_jolt_core::jolt::{JoltProverPreprocessing, JoltSNARK};
use jolt_core::poly::commitment::dory::DoryCommitmentScheme;

type PCS = DoryCommitmentScheme;

// Preprocessing (done once at startup, ~1s with cached SRS)
let preprocessing = JoltSNARK::<Fr, PCS, KeccakTranscript>
    ::prover_preprocess(model_fn, 1 << 20);

// Proof generation (~5s per classification)
let (snark, program_io, _) = JoltSNARK::prove(&preprocessing, model_fn, &input);

// Verification (~1.3s)
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
│       ├── zkml/mod.rs      # Real Jolt Atlas ZKML prover
│       └── ap2/mod.rs       # AP2 gateway integration
├── ap2_service/
│   └── ap2_service.py       # Python FastAPI AP2 simulation
├── frontend/
│   └── index.html           # Single-page demo UI
└── README.md
```

## Google AI Edge Integration

This demo is designed to integrate with Google's agentic commerce stack:

| Component | Integration |
|-----------|-------------|
| **Google AI Edge** | On-device ML (Gemma on LiteRT) for autonomous decisions |
| **Google A2A** | Agent-to-Agent protocol for multi-agent coordination |
| **Google AP2** | Agent-to-Payment with SpendingProof field |

The ZKML proof from Jolt Atlas becomes the cryptographic guarantee that the on-device AI made a policy-compliant decision, enabling trustless autonomous commerce.

## Status

- [x] Real Jolt Atlas ZKML proofs (Dory commitment scheme)
- [x] End-to-end demo flow with real proof generation
- [x] AP2 protocol integration (simulated service)
- [x] Enterprise policy enforcement
- [ ] Production AP2 sandbox integration
- [ ] Multi-agent A2A coordination demo
- [ ] Spending analytics dashboard
