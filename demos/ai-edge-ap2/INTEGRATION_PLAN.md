# Real Google Integration Plan

## Overview

Integrate real Google AP2 types and Google AI Edge into the demo.

## Phase 1: Study AP2 Types ✅ COMPLETE

**Findings from `/home/hshadab/aiedge/AP2/`:**

### AP2 Core Types
```
src/ap2/types/
├── payment_request.py   # W3C Payment Request API types
│   ├── PaymentCurrencyAmount
│   ├── PaymentItem
│   ├── PaymentMethodData    # <-- SpendingProof goes in .data field
│   ├── PaymentRequest
│   └── PaymentResponse
├── payment_receipt.py   # Payment completion
│   ├── PaymentReceipt
│   ├── Success/Error/Failure
└── mandate.py           # Agent authorization
    ├── IntentMandate    # User's purchase intent
    ├── CartMandate      # Merchant-signed cart
    └── PaymentMandate   # User's payment authorization
```

### Key Integration Point
The `PaymentMethodData.data` field accepts arbitrary dict - we put SpendingProof here:
```python
PaymentMethodData(
    supported_methods="zkml-spending-proof",
    data={
        "classification_proof": "0x...",
        "policy_attestation": "0x...",
        "model_hash": "sha256:...",
        "proving_time_ms": 4715
    }
)
```

---

## Phase 2: Integrate Real AP2 Types

### Step 2.1: Install AP2 Package
```bash
cd demos/ai-edge-ap2/ap2_service
pip install git+https://github.com/google-agentic-commerce/AP2.git@main
```

### Step 2.2: Update ap2_service.py
Replace fake types with real AP2 types:

```python
# OLD (fake)
class QuickPayRequest(BaseModel):
    vendor: str
    amount_cents: int

# NEW (real AP2)
from ap2.types.payment_request import (
    PaymentRequest, PaymentMethodData, PaymentDetailsInit,
    PaymentItem, PaymentCurrencyAmount
)
from ap2.types.payment_receipt import PaymentReceipt, Success
from ap2.types.mandate import PaymentMandate, IntentMandate
```

### Step 2.3: Create SpendingProof Type
Add to our codebase:
```python
# ap2_service/spending_proof.py
from pydantic import BaseModel
from typing import Optional

class SpendingProof(BaseModel):
    """zkML proof attached to AP2 payments"""
    classification_proof: str  # Hex-encoded SNARK
    policy_attestation: str    # Hex-encoded attestation
    model_hash: str            # Model identifier
    classification: str        # Category result
    confidence: float          # Classification confidence
    proving_time_ms: int       # Proof generation time
    verification_time_ms: Optional[int] = None

# Register as payment method
zkML_PAYMENT_METHOD = "https://novanet.xyz/zkml-spending-proof"
```

### Step 2.4: Update Rust Backend
Map our ProofBundle to AP2 SpendingProof format.

---

## Phase 3: Add Real Google AI Edge Classifier

### Option A: MediaPipe Text Classifier (Recommended)
```python
# Install
pip install mediapipe

# Use pre-trained or custom classifier
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text

# Load classifier
classifier = text.TextClassifier.create_from_options(
    text.TextClassifierOptions(
        base_options=python.BaseOptions(model_asset_path="spending_classifier.tflite"),
        max_results=5
    )
)

# Classify
result = classifier.classify("purchase route data from HERE Maps")
# Returns: ClassificationResult with categories and scores
```

### Step 3.1: Download/Create TFLite Model
```bash
# Option 1: Use MediaPipe's average word embedding classifier as base
wget https://storage.googleapis.com/mediapipe-models/text_classifier/average_word_classifier/float32/latest/average_word_classifier.tflite

# Option 2: Convert our ONNX model to TFLite
python scripts/convert_to_tflite.py
```

### Step 3.2: Create AI Edge Service
```python
# ai_edge_service.py
from mediapipe.tasks.python import text

class AIEdgeClassifier:
    def __init__(self, model_path: str):
        self.classifier = text.TextClassifier.create_from_options(
            text.TextClassifierOptions(
                base_options=python.BaseOptions(model_asset_path=model_path)
            )
        )

    def classify(self, function_call: dict) -> tuple[str, float]:
        """Classify a function call using MediaPipe"""
        text_input = f"{function_call['function']} {function_call['vendor']}"
        result = self.classifier.classify(text_input)
        top = result.classifications[0].categories[0]
        return top.category_name, top.score
```

### Step 3.3: Architecture Decision
```
┌─────────────────────────┐
│  MediaPipe Classifier   │  <-- Real Google AI Edge
│  (spending_classifier)  │
│  - Runs locally         │
│  - ~10ms inference      │
└───────────┬─────────────┘
            │ Classification result
            ▼
┌─────────────────────────┐
│  Jolt Atlas zkML        │  <-- Real SNARK proof
│  - Proves the inference │
│  - ~5s proving          │
└───────────┬─────────────┘
            │ SpendingProof
            ▼
┌─────────────────────────┐
│  Real AP2 Types         │  <-- Real Google AP2
│  - PaymentRequest       │
│  - PaymentMethodData    │
└─────────────────────────┘
```

---

## Phase 4: Wire Up Full Integration

### Step 4.1: Update Backend Flow
```rust
// backend/src/api/mod.rs

pub async fn classify_and_prove(call: &FunctionCall) -> ClassifyResponse {
    // 1. Call AI Edge classifier (MediaPipe)
    let ai_edge_result = ai_edge_client.classify(&call).await;

    // 2. Generate zkML proof (Jolt Atlas)
    let (classification, proof_bundle) = zkml_prover.classify_and_prove(&call).await;

    // 3. Create AP2 SpendingProof
    let spending_proof = SpendingProof {
        classification_proof: proof_bundle.classification_proof,
        policy_attestation: proof_bundle.policy_attestation,
        model_hash: proof_bundle.model_hash,
        classification: classification.category,
        confidence: classification.confidence,
        proving_time_ms: proof_bundle.proving_time_ms,
    };

    // 4. Build AP2 PaymentMethodData
    let payment_method = PaymentMethodData {
        supported_methods: "zkml-spending-proof",
        data: spending_proof.to_dict(),
    };

    ClassifyResponse { ... }
}
```

### Step 4.2: Update AP2 Service
```python
# ap2_service.py

@app.post("/v1/payments/request")
async def create_payment_request(request: CreatePaymentRequest) -> PaymentRequest:
    """Create AP2 PaymentRequest with SpendingProof"""

    # Validate SpendingProof
    spending_proof = SpendingProof(**request.spending_proof)

    # Create real AP2 PaymentRequest
    return PaymentRequest(
        method_data=[
            PaymentMethodData(
                supported_methods="zkml-spending-proof",
                data=spending_proof.model_dump()
            )
        ],
        details=PaymentDetailsInit(
            id=str(uuid.uuid4()),
            display_items=[
                PaymentItem(
                    label=request.description,
                    amount=PaymentCurrencyAmount(
                        currency="USD",
                        value=request.amount_cents / 100
                    )
                )
            ],
            total=PaymentItem(
                label="Total",
                amount=PaymentCurrencyAmount(
                    currency="USD",
                    value=request.amount_cents / 100
                )
            )
        )
    )
```

### Step 4.3: Update UI Labels
```javascript
// Show real integration status
const integrationStatus = {
    aiEdge: "MediaPipe Text Classifier (Real)",
    zkml: "Jolt Atlas SNARK (Real)",
    ap2: "Google AP2 Types (Real)",
    payment: "Simulated Settlement"
};
```

---

## File Changes Summary

| File | Change |
|------|--------|
| `ap2_service/requirements.txt` | Add `ap2`, `mediapipe` |
| `ap2_service/ap2_service.py` | Use real AP2 types |
| `ap2_service/spending_proof.py` | New - SpendingProof type |
| `ap2_service/ai_edge_classifier.py` | New - MediaPipe classifier |
| `backend/src/api/mod.rs` | Return AP2-compatible format |
| `frontend/index.html` | Update labels to show "Real" |
| `README.md` | Update integration status |

---

## Timeline

| Phase | Task | Status |
|-------|------|--------|
| 1 | Clone AP2, study types | ✅ Complete |
| 2 | Integrate AP2 types | ✅ Complete |
| 3 | Add MediaPipe classifier | ✅ Complete |
| 4 | Wire up + update UI | ✅ Complete |

## Integration Complete!

### What's Now Real:

1. **Google AP2 Types** - Imported from `https://github.com/google-agentic-commerce/AP2`
   - `PaymentRequest`, `PaymentReceipt`, `PaymentMethodData`
   - `IntentMandate`, `CartMandate`, `PaymentMandate`
   - SpendingProof converts to/from `GooglePaymentMethodData`

2. **Google AI Edge** - MediaPipe text classifier
   - Model: `average_word_classifier.tflite` (~776KB)
   - Inference: ~1-2ms per classification
   - Categories: data_services, cloud_compute, ai_inference, analytics, etc.

3. **New API Endpoints**:
   - `GET /api/v1/integration/status` - Full integration status
   - `GET /api/v1/ai-edge/status` - MediaPipe classifier status
   - `POST /api/v1/ai-edge/classify` - Real MediaPipe classification
   - `GET /api/v1/ap2/status` - Google AP2 types status

### What Remains Simulated:
- Payment settlement (sandbox mode)
- Gemma LLM agent (the AI making decisions)
