# AI Edge → AP2 Payment Demo

This demo shows how ZKML (Zero-Knowledge Machine Learning) enables secure, verifiable spending for autonomous AI agents in enterprise B2B scenarios.

## What This Demo Shows

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  AI Edge Agent  │────▶│ ZKML Prover  │────▶│ Policy Engine   │────▶│ AP2 Payment  │
│ (Device)        │     │ (Classify)   │     │ (Enterprise)    │     │ (Google)     │
└─────────────────┘     └──────────────┘     └─────────────────┘     └──────────────┘
     Function Call      Classification +    Policy Compliance      Payment with
                        SNARK Proof         Verification           SpendingProof
```

### The Problem
AI agents making autonomous purchasing decisions need:
1. **Verifiable intent classification** - What is the agent actually trying to buy?
2. **Policy enforcement** - Does this purchase comply with enterprise spending rules?
3. **Non-repudiation** - Cryptographic proof that the classification was done correctly

### The Solution
ZKML generates a SNARK proof that:
- The AI's function call was classified by a specific ML model
- The classification result is correct (verifiable by anyone)
- The proof is attached to the AP2 payment as a "SpendingProof"

## Spending Categories

| Category | Description | Example Vendors |
|----------|-------------|-----------------|
| `data_services` | API calls, data feeds | Clearbit, ZoomInfo |
| `cloud_compute` | GPU, inference, compute | AWS Bedrock, Lambda |
| `logistics_data` | Routing, maps, fleet | HERE Maps, Google Maps |
| `saas_licenses` | Software subscriptions | Datadog, Splunk |
| `blocked` | Prohibited transactions | External transfers |

## Demo Scenarios

### 1. Approved Purchase (Route Data)
- Agent: "Purchase route optimization data from HERE Maps"
- Classification: `logistics_data` (95% confidence)
- Policy: ✅ Under $500 limit, approved vendor
- Result: Payment processed via AP2

### 2. Blocked Transfer
- Agent: "Transfer $5000 to external account"
- Classification: `blocked` (98% confidence)
- Policy: ❌ Category blocked by policy
- Result: Payment rejected, audit logged

### 3. Over Limit
- Agent: "Purchase $2000 GPU compute from AWS Bedrock"
- Classification: `cloud_compute` (92% confidence)
- Policy: ❌ Exceeds $1000 category limit
- Result: Requires approval, escalated

### 4. Unapproved Vendor
- Agent: "Get company data from SketchyData Inc"
- Classification: `data_services` (78% confidence)
- Policy: ❌ Vendor not in approved list
- Result: Payment rejected

## Running the Demo

### Prerequisites
- Rust (nightly)
- Python 3.x

### Start the Demo

```bash
cd demos/ai-edge-ap2
./scripts/run-demo.sh
```

This starts:
- **Backend**: http://localhost:3001 (Rust/Axum API server)
- **Frontend**: http://localhost:3000 (Interactive demo UI)

### Using the Demo

1. Open http://localhost:3000 in your browser
2. **Device Simulator Tab**: Simulate AI agent function calls
3. **Enterprise Dashboard Tab**: View spending policies and transaction history
4. **Vendor Portal Tab**: See how vendors receive payments with proofs

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/classify` | POST | Classify function call and generate ZKML proof |
| `/api/verify` | POST | Verify a ZKML proof bundle |
| `/api/policy` | GET/PUT | Get/update spending policy |
| `/api/ap2/pay` | POST | Process payment via AP2 with SpendingProof |
| `/api/transactions` | GET | List transaction history |
| `/api/demo/trigger` | POST | Trigger demo scenarios |

## Architecture

```
demos/ai-edge-ap2/
├── backend/
│   └── src/
│       ├── main.rs          # Axum server
│       ├── api/mod.rs       # API handlers
│       ├── zkml/mod.rs      # ZKML prover (wraps Jolt Atlas)
│       └── ap2/mod.rs       # Mock AP2 gateway
├── frontend/
│   └── index.html           # Single-page demo UI
└── scripts/
    └── run-demo.sh          # Demo runner
```

## Integration with Google's Agentic Commerce Stack

This demo is designed to integrate with:

- **Google AI Edge**: On-device ML for function call classification
- **Google A2A**: Agent-to-Agent protocol for multi-agent coordination
- **Google AP2**: Agent-to-Payment protocol with SpendingProof support

The ZKML proof becomes the `spending_proof` field in AP2 payment requests:

```json
{
  "ap2_version": "1.0",
  "payment": {
    "amount_cents": 9900,
    "payer_id": "ap2://acme-corp-agent",
    "payee_id": "ap2://here-maps"
  },
  "spending_proof": {
    "classification_proof": "0x7f83b1657ff1...",
    "policy_attestation": "0xa4b2c9d8e7f6...",
    "model_hash": "sha256:7f83b1657ff1fc53..."
  }
}
```

## Next Steps

- [ ] Integrate real Jolt Atlas ZKML proofs (currently simulated)
- [ ] Add multi-agent A2A coordination demo
- [ ] Connect to actual AP2 sandbox
- [ ] Add spending analytics dashboard
