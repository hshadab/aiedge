# Implementation Plan: Real AI Edge → ZKML → AP2 Demo

## Status: ✅ IMPLEMENTED

This document describes the implementation plan. The following components have been implemented:

### ✅ Phase 1: Real ZKML Proofs (Jolt Atlas)
- **Cargo.toml**: Updated with feature-gated ZKML dependencies (`real-zkml` feature)
- **zkml/mod.rs**: Dual implementation with mock (default) and real ZKML prover
- **spending_classifier**: Model exists at `onnx-tracer/models/spending_classifier/`

### ✅ Phase 2: Real AP2 Protocol (Python Service)
- **ap2-service/**: Complete FastAPI implementation
- **models.py**: Full AP2 protocol types (PaymentRequest, PaymentReceipt, SpendingProof)
- **main.py**: Service running on port 3002 with quick-pay and verification endpoints
- **Rust backend**: Updated `ap2/mod.rs` to call Python service with fallback to mock

### ✅ Phase 3: Frontend & Run Script
- **run-demo.sh**: Updated to start all three services (frontend:3000, backend:3001, AP2:3002)
- **Frontend**: Existing visualization with MediaPipe integration notes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            BROWSER (port 3000)                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Demo UI with Flow Visualization                                  │   │
│  │  - Function call display                                          │   │
│  │  - SpendingProof visualization                                    │   │
│  │  - Activity log                                                   │   │
│  │  (MediaPipe can be added for on-device classification)            │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────────────┐
│                     RUST BACKEND (port 3001)                            │
│                                  ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  Axum API Server                                                │    │
│  │  - /api/v1/classify - Classification + proof generation        │    │
│  │  - /api/v1/verify - Proof verification                         │    │
│  │  - /api/v1/ap2/pay - Payment processing                        │    │
│  │  - /api/v1/demo/* - Demo control endpoints                     │    │
│  └────────────────────────────────────────────────────────────────┘    │
│           │                    │                       │                │
│           ▼                    ▼                       ▼                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐     │
│  │ ZKML Prover     │  │ Policy Engine   │  │ AP2 Client          │     │
│  │                 │  │                 │  │                     │     │
│  │ Mock (default): │  │ - Category      │  │ Calls Python AP2    │     │
│  │ - Fast proofs   │  │   limits        │  │ service on :3002    │     │
│  │ - ~300ms        │  │ - Vendor        │  │                     │     │
│  │                 │  │   approval      │  │ Falls back to mock  │     │
│  │ Real (feature): │  │ - Daily caps    │  │ if unavailable      │     │
│  │ - Jolt SNARK    │  │                 │  │                     │     │
│  │ - ~3-5s         │  │                 │  │                     │     │
│  │ - ~48KB proof   │  │                 │  │                     │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     AP2 PYTHON SERVICE (port 3002)                      │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │  FastAPI Server                                                 │    │
│  │  - /api/v1/demo/quick-pay - Simplified payment endpoint        │    │
│  │  - /api/v1/payments/process - Full AP2 protocol                │    │
│  │  - /api/v1/verify - SpendingProof verification                 │    │
│  │  - /api/v1/vendors - Approved vendor registry                  │    │
│  │                                                                 │    │
│  │  Uses Google AP2 protocol types:                               │    │
│  │  - PaymentRequest, PaymentReceipt                              │    │
│  │  - AgentIdentifier, MonetaryAmount                             │    │
│  │  - SpendingProof (Novanet extension)                           │    │
│  └────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
demos/ai-edge-ap2/
├── README.md
├── IMPLEMENTATION_PLAN.md (this file)
├── backend/
│   ├── Cargo.toml              # With feature flags for real-zkml
│   └── src/
│       ├── main.rs             # Axum server
│       ├── api/mod.rs          # API handlers
│       ├── zkml/mod.rs         # ZKML prover (mock + real)
│       └── ap2/mod.rs          # AP2 client (calls Python service)
├── ap2-service/
│   ├── requirements.txt        # Python dependencies
│   ├── config.py               # Service configuration
│   ├── models.py               # AP2 protocol types
│   └── main.py                 # FastAPI service
├── frontend/
│   └── index.html              # Demo UI
└── scripts/
    └── run-demo.sh             # Starts all 3 services
```

---

## Running the Demo

### Quick Start (Mock Mode)
```bash
cd demos/ai-edge-ap2
./scripts/run-demo.sh
```

This starts:
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:3001
- **AP2 Service**: http://localhost:3002

### With Real ZKML Proofs (Requires Nightly Rust)
```bash
# Ensure nightly Rust
rustup default nightly

# Build with real-zkml feature
cd demos/ai-edge-ap2/backend
cargo build --release --features real-zkml
cargo run --release --features real-zkml
```

---

## Feature Flags

### `real-zkml` (Rust Backend)
When enabled, the backend uses real Jolt Atlas SNARK proofs:
- Requires nightly Rust (edition 2024)
- Proof generation: ~3-5 seconds
- Proof size: ~48KB
- Uses spending_classifier ONNX model

Without this flag, mock proofs are used (fast, for development).

---

## API Endpoints

### Backend (port 3001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/classify` | POST | Classify + generate proof |
| `/api/v1/verify` | POST | Verify proof |
| `/api/v1/policy/:id` | GET | Get policy |
| `/api/v1/transactions` | GET | List transactions |
| `/api/v1/ap2/pay` | POST | Process payment |
| `/api/v1/demo/trigger/:scenario` | POST | Trigger demo scenario |
| `/api/v1/demo/reset` | POST | Reset demo state |

### AP2 Service (port 3002)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/demo/quick-pay` | POST | Simplified payment |
| `/api/v1/payments/process` | POST | Full AP2 payment |
| `/api/v1/verify` | POST | Verify SpendingProof |
| `/api/v1/vendors` | GET | List approved vendors |
| `/api/v1/demo/reset` | POST | Reset demo state |

---

## SpendingProof Format

The SpendingProof attached to AP2 payments:

```json
{
  "classification_proof": "0x7f83b165...",  // SNARK proof (hex)
  "policy_attestation": "0xa4b2c9d8...",   // Policy check attestation
  "model_hash": "sha256:7f83b1657ff1...",  // Model identifier
  "classification_result": "logistics_data", // Category
  "confidence": 0.95,                       // Confidence score
  "timestamp": "2025-01-19T15:30:00Z"       // When proof was generated
}
```

---

## Next Steps

### To Enable Real ZKML
1. Ensure nightly Rust: `rustup default nightly`
2. Build with feature: `cargo build --release --features real-zkml`
3. First startup takes ~30s for model preprocessing

### To Add Real MediaPipe (Browser)
1. Add MediaPipe Tasks JS SDK to frontend
2. Load spending_classifier.tflite model
3. Run classification locally before calling backend

### To Integrate Google AP2 SDK
1. Install: `pip install git+https://github.com/google-agentic-commerce/AP2.git@main`
2. Import official types in ap2-service/main.py
3. Use official PaymentRequest/PaymentReceipt classes

---

## Implementation Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Rust Backend | ✅ Complete | Mock + real-zkml feature |
| Policy Engine | ✅ Complete | Category limits, vendor approval |
| AP2 Python Service | ✅ Complete | Full protocol implementation |
| AP2 Client (Rust) | ✅ Complete | HTTP client with fallback |
| Run Script | ✅ Complete | Starts all 3 services |
| Frontend | ✅ Complete | Visualization + annotations |
| Real ZKML | ⚙️ Feature-gated | Enable with `--features real-zkml` |
| MediaPipe Browser | 📝 Documented | Can be added to frontend |
