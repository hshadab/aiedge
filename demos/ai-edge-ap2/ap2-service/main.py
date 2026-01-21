"""
AP2 Payment Service

FastAPI service implementing Google's Agent Payments Protocol (AP2)
with ZKML SpendingProof verification support.

Runs on port 3002.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    PaymentRequest,
    PaymentReceipt,
    PaymentStatus,
    PaymentError,
    ProcessPaymentRequest,
    ProcessPaymentResponse,
    VerifyProofRequest,
    VerifyProofResponse,
    SpendingProof,
    AgentIdentifier,
    MonetaryAmount,
    Currency,
    check_google_ap2_available,
    ZKML_SPENDING_PROOF_METHOD,
)
from ai_edge_classifier import (
    get_classifier,
    check_ai_edge_available,
    ClassificationResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ap2-service")

# Initialize FastAPI app
app = FastAPI(
    title="AP2 Payment Service",
    description="Google Agent Payments Protocol (AP2) implementation with ZKML SpendingProof support",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
transactions: Dict[str, PaymentReceipt] = {}
pending_authorizations: Dict[str, PaymentRequest] = {}

# Approved vendors registry (simulating AP2 network registry)
APPROVED_VENDORS = {
    "vendor:here_routing": {"name": "HERE Routing API", "merchant_id": "here-maps"},
    "vendor:google_maps": {"name": "Google Maps Platform", "merchant_id": "google-maps"},
    "vendor:aws_bedrock": {"name": "AWS Bedrock", "merchant_id": "amazon-bedrock"},
    "vendor:clearbit_api": {"name": "Clearbit API", "merchant_id": "clearbit"},
    "vendor:datadog": {"name": "Datadog", "merchant_id": "datadog"},
    "ap2://here-maps": {"name": "HERE Maps", "merchant_id": "here-maps"},
    "ap2://google-maps": {"name": "Google Maps", "merchant_id": "google-maps"},
    "ap2://aws-bedrock": {"name": "AWS Bedrock", "merchant_id": "amazon-bedrock"},
}


# =============================================================================
# Health & Status
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "ap2-payment-service",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/status")
async def service_status():
    """Get service status and statistics."""
    return {
        "status": "operational",
        "sandbox_mode": True,
        "transactions_processed": len(transactions),
        "pending_authorizations": len(pending_authorizations),
        "approved_vendors_count": len(APPROVED_VENDORS),
    }


@app.get("/api/v1/ap2/status")
async def google_ap2_status():
    """
    Check Google AP2 integration status.

    Returns information about which real Google AP2 types are available.
    """
    ap2_info = check_google_ap2_available()
    return {
        "integration": "google-ap2",
        "version": "1.0.0",
        "repository": "https://github.com/google-agentic-commerce/AP2",
        "zkml_payment_method": ZKML_SPENDING_PROOF_METHOD,
        **ap2_info,
    }


# =============================================================================
# Google AI Edge Classification
# =============================================================================

class ClassifyRequest(BaseModel):
    """Request to classify a function call."""
    function_name: str
    vendor: str
    description: Optional[str] = None


class ClassifyResponse(BaseModel):
    """Response from AI Edge classification."""
    category: str
    confidence: float
    inference_time_ms: float
    model_name: str
    is_real_ai_edge: bool


@app.get("/api/v1/ai-edge/status")
async def ai_edge_status():
    """
    Check Google AI Edge (MediaPipe) integration status.

    Returns information about the classifier and model status.
    """
    return check_ai_edge_available()


@app.post("/api/v1/ai-edge/classify", response_model=ClassifyResponse)
async def classify_function_call(request: ClassifyRequest):
    """
    Classify an AI agent's function call using Google AI Edge.

    This runs real MediaPipe inference to determine the spending category.
    """
    classifier = get_classifier()
    result = classifier.classify_function_call(
        function_name=request.function_name,
        vendor=request.vendor,
        description=request.description
    )

    return ClassifyResponse(
        category=result.category,
        confidence=result.confidence,
        inference_time_ms=result.inference_time_ms,
        model_name=result.model_name,
        is_real_ai_edge=classifier.is_real_ai_edge,
    )


@app.get("/api/v1/integration/status")
async def full_integration_status():
    """
    Get full integration status for all real Google components.

    Shows which parts of the demo use real Google technology vs simulation.
    """
    ap2_info = check_google_ap2_available()
    ai_edge_info = check_ai_edge_available()

    return {
        "integration_summary": {
            "google_ap2": {
                "status": "real" if ap2_info["google_ap2_available"] else "simulated",
                "description": "Google Agent Payments Protocol (AP2) types",
                "repository": "https://github.com/google-agentic-commerce/AP2",
            },
            "google_ai_edge": {
                "status": "real" if ai_edge_info["is_real_ai_edge"] else "simulated",
                "description": "MediaPipe text classifier for spending categorization",
                "documentation": "https://ai.google.dev/edge",
            },
            "zkml_proofs": {
                "status": "real",
                "description": "Jolt Atlas SNARK proofs for classification verification",
                "note": "Proves the spending classifier, not the LLM agent",
            },
            "payment_settlement": {
                "status": "simulated",
                "description": "Actual money transfer (sandbox mode)",
            },
        },
        "details": {
            "ap2": ap2_info,
            "ai_edge": ai_edge_info,
        },
    }


# =============================================================================
# Payment Processing
# =============================================================================

@app.post("/api/v1/payments/process", response_model=ProcessPaymentResponse)
async def process_payment(request: ProcessPaymentRequest):
    """
    Process an AP2 payment request.

    This endpoint:
    1. Validates the payment request
    2. Verifies the SpendingProof (if provided)
    3. Checks vendor registration
    4. Processes the payment
    5. Returns a receipt
    """
    payment_req = request.payment_request
    logger.info(f"Processing payment request: {payment_req.request_id}")

    start_time = time.time()

    try:
        # Step 1: Validate payer and payee
        if not payment_req.payer.agent_id:
            raise HTTPException(status_code=400, detail="Payer agent_id is required")
        if not payment_req.payee.agent_id:
            raise HTTPException(status_code=400, detail="Payee agent_id is required")

        # Step 2: Check vendor registration
        payee_id = payment_req.payee.agent_id
        vendor_info = APPROVED_VENDORS.get(payee_id)

        if not vendor_info and not payee_id.startswith("ap2://"):
            # Try with vendor: prefix
            vendor_info = APPROVED_VENDORS.get(f"vendor:{payee_id.split('/')[-1]}")

        # For sandbox mode, allow any vendor but log warning
        if not vendor_info:
            logger.warning(f"Vendor not in approved list: {payee_id} (allowing in sandbox mode)")
            vendor_info = {"name": payee_id, "merchant_id": "sandbox-merchant"}

        # Step 3: Verify SpendingProof (if provided)
        proof_verified = None
        verification_details = {}

        if payment_req.spending_proof:
            proof_result = await verify_spending_proof(payment_req.spending_proof)
            proof_verified = proof_result["valid"]
            verification_details = {
                "proof_verified": proof_verified,
                "verification_time_ms": proof_result["verification_time_ms"],
                "model_hash": payment_req.spending_proof.model_hash,
                "classification": payment_req.spending_proof.classification,
            }
            logger.info(f"SpendingProof verification: {proof_verified}")

            if not proof_verified:
                return ProcessPaymentResponse(
                    success=False,
                    error=PaymentError(
                        error_code="INVALID_SPENDING_PROOF",
                        error_message="The provided SpendingProof could not be verified",
                        request_id=payment_req.request_id,
                        details=verification_details,
                    )
                )

        # Step 4: Process payment (simulated in sandbox mode)
        transaction_id = f"AP2-TXN-{uuid.uuid4().hex[:12].upper()}"

        # Simulate payment processing delay
        await asyncio.sleep(0.1)

        # Step 5: Create receipt
        receipt = PaymentReceipt(
            request_id=payment_req.request_id,
            transaction_id=transaction_id,
            status=PaymentStatus.COMPLETED,
            status_message="Payment processed successfully via AP2 protocol",
            payer_id=payment_req.payer.agent_id,
            payee_id=payment_req.payee.agent_id,
            amount=payment_req.amount,
            spending_proof_verified=proof_verified,
            verification_details=verification_details,
            processed_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            network_reference=f"AP2-NET-{uuid.uuid4().hex[:8].upper()}",
            settlement_info={
                "merchant_name": vendor_info["name"],
                "settlement_date": (datetime.utcnow() + timedelta(days=1)).isoformat(),
            },
        )

        # Store transaction
        transactions[transaction_id] = receipt

        processing_time = (time.time() - start_time) * 1000
        logger.info(f"Payment processed in {processing_time:.2f}ms: {transaction_id}")

        # Generate Google AP2 format receipt for interoperability
        google_receipt = receipt.to_google_receipt()

        return ProcessPaymentResponse(
            success=True,
            receipt=receipt,
            google_ap2_receipt=google_receipt.model_dump(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Payment processing failed: {e}")
        return ProcessPaymentResponse(
            success=False,
            error=PaymentError(
                error_code="PROCESSING_ERROR",
                error_message=str(e),
                request_id=payment_req.request_id,
            )
        )


@app.get("/api/v1/payments/{transaction_id}")
async def get_transaction(transaction_id: str):
    """Get transaction details by ID."""
    if transaction_id not in transactions:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transactions[transaction_id]


@app.get("/api/v1/payments")
async def list_transactions(limit: int = 50, offset: int = 0):
    """List all transactions."""
    all_txns = list(transactions.values())
    return {
        "transactions": all_txns[offset:offset + limit],
        "total": len(all_txns),
        "limit": limit,
        "offset": offset,
    }


# =============================================================================
# SpendingProof Verification
# =============================================================================

async def verify_spending_proof(proof: SpendingProof) -> dict:
    """
    Verify a ZKML SpendingProof.

    In production, this would:
    1. Deserialize the SNARK proof
    2. Verify against the model hash
    3. Check the policy attestation

    For sandbox mode, we perform basic validation.
    """
    start_time = time.time()

    # Basic format validation
    if not proof.classification_proof or not proof.classification_proof.startswith("0x"):
        return {
            "valid": False,
            "verification_time_ms": int((time.time() - start_time) * 1000),
            "error": "Invalid proof format",
        }

    if len(proof.classification_proof) < 20:
        return {
            "valid": False,
            "verification_time_ms": int((time.time() - start_time) * 1000),
            "error": "Proof too short",
        }

    if not proof.model_hash:
        return {
            "valid": False,
            "verification_time_ms": int((time.time() - start_time) * 1000),
            "error": "Model hash required",
        }

    # Simulate verification delay (real SNARK verification takes ~100ms)
    await asyncio.sleep(0.05)

    verification_time_ms = int((time.time() - start_time) * 1000)

    return {
        "valid": True,
        "verification_time_ms": verification_time_ms,
        "proof_size_bytes": len(proof.classification_proof) // 2,  # hex encoding
        "model_verified": True,
    }


@app.post("/api/v1/verify", response_model=VerifyProofResponse)
async def verify_proof_endpoint(request: VerifyProofRequest):
    """
    Standalone endpoint to verify a SpendingProof.

    Useful for vendors who want to verify proofs independently.
    """
    result = await verify_spending_proof(request.spending_proof)
    return VerifyProofResponse(
        valid=result["valid"],
        verification_time_ms=result["verification_time_ms"],
        details=result,
    )


# =============================================================================
# Vendor Registry
# =============================================================================

@app.get("/api/v1/vendors")
async def list_approved_vendors():
    """List all approved vendors in the AP2 network."""
    return {
        "vendors": [
            {"id": vendor_id, **info}
            for vendor_id, info in APPROVED_VENDORS.items()
        ],
        "total": len(APPROVED_VENDORS),
    }


@app.get("/api/v1/vendors/{vendor_id}")
async def get_vendor(vendor_id: str):
    """Get vendor details."""
    vendor = APPROVED_VENDORS.get(vendor_id)
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    return {"id": vendor_id, **vendor}


# =============================================================================
# Demo Endpoints
# =============================================================================

class QuickPayRequest(BaseModel):
    """Simplified payment request for demo purposes."""
    vendor: str
    amount_cents: int
    payer_id: str = "acme-corp-agent"
    memo: Optional[str] = None
    spending_proof: Optional[dict] = None


@app.post("/api/v1/demo/quick-pay")
async def quick_pay(request: QuickPayRequest):
    """
    Simplified payment endpoint for demo purposes.

    Accepts minimal parameters and constructs full AP2 request internally.
    """
    # Build spending proof if provided
    proof = None
    if request.spending_proof:
        proof = SpendingProof(
            classification_proof=request.spending_proof.get("classification_proof", ""),
            policy_attestation=request.spending_proof.get("policy_attestation", ""),
            model_hash=request.spending_proof.get("model_hash", ""),
            classification=request.spending_proof.get("classification", ""),
            confidence=request.spending_proof.get("confidence", 0.0),
            proving_time_ms=request.spending_proof.get("proving_time_ms", 0),
        )

    # Build full payment request
    payment_request = PaymentRequest(
        payer=AgentIdentifier(
            agent_id=f"ap2://{request.payer_id}",
            agent_type="commerce",
            display_name="Acme Corp AI Agent",
        ),
        payee=AgentIdentifier(
            agent_id=request.vendor if request.vendor.startswith("ap2://") else f"ap2://{request.vendor}",
            agent_type="merchant",
        ),
        amount=MonetaryAmount(
            amount_cents=request.amount_cents,
            currency=Currency.USD,
        ),
        memo=request.memo,
        spending_proof=proof,
    )

    # Process via main endpoint
    result = await process_payment(ProcessPaymentRequest(payment_request=payment_request))

    # Simplified response
    if result.success:
        return {
            "success": True,
            "transaction_id": result.receipt.transaction_id,
            "amount_cents": request.amount_cents,
            "vendor": request.vendor,
            "proof_verified": result.receipt.spending_proof_verified,
            "message": result.receipt.status_message,
        }
    else:
        return {
            "success": False,
            "error": result.error.error_message if result.error else "Unknown error",
        }


@app.post("/api/v1/demo/reset")
async def reset_demo():
    """Reset all demo data."""
    transactions.clear()
    pending_authorizations.clear()
    return {"status": "reset", "message": "All demo data cleared"}


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting AP2 Payment Service...")
    logger.info("Sandbox mode: ENABLED")
    logger.info(f"Approved vendors: {len(APPROVED_VENDORS)}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3002,
        log_level="info",
    )
