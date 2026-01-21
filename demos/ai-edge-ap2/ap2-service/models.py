"""
AP2 Protocol Data Models

Integrates with Google's official Agent Payments Protocol (AP2) types.
https://github.com/google-agentic-commerce/AP2
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid

# =============================================================================
# Import REAL Google AP2 Types
# =============================================================================
from ap2.types.payment_request import (
    PaymentCurrencyAmount as GooglePaymentCurrencyAmount,
    PaymentItem as GooglePaymentItem,
    PaymentMethodData as GooglePaymentMethodData,
    PaymentDetailsInit as GooglePaymentDetailsInit,
    PaymentRequest as GooglePaymentRequest,
    PaymentResponse as GooglePaymentResponse,
    PaymentOptions as GooglePaymentOptions,
)
from ap2.types.payment_receipt import (
    PaymentReceipt as GooglePaymentReceipt,
    Success as GoogleSuccess,
    Error as GoogleError,
    Failure as GoogleFailure,
)
from ap2.types.mandate import (
    IntentMandate as GoogleIntentMandate,
    CartMandate as GoogleCartMandate,
    PaymentMandate as GooglePaymentMandate,
)

# =============================================================================
# ZKML SpendingProof - Our Extension to AP2
# =============================================================================

# Payment method identifier for ZKML SpendingProof
ZKML_SPENDING_PROOF_METHOD = "https://novanet.xyz/zkml-spending-proof/v1"


class SpendingProof(BaseModel):
    """
    ZKML Spending Proof - Novanet's extension to Google AP2.

    This goes into PaymentMethodData.data field in AP2 PaymentRequest.

    Proves that an AI agent's spending decision was:
    1. Classified by a verifiable ML model (spending_classifier)
    2. The classification was done correctly (SNARK proof)
    3. Compliant with enterprise spending policy
    """
    # Core ZKML proof data
    classification_proof: str = Field(..., description="Hex-encoded Jolt Atlas SNARK proof")
    policy_attestation: str = Field(..., description="Hex-encoded policy compliance attestation")
    model_hash: str = Field(..., description="Hash of the ML model (e.g., sha256:spending_classifier_v1)")

    # Classification result
    classification: str = Field(..., description="Spending category (data_services, cloud_compute, etc.)")
    confidence: float = Field(..., description="Classification confidence (0.0-1.0)")

    # Timing metadata
    proving_time_ms: int = Field(..., description="Time to generate SNARK proof in milliseconds")
    verification_time_ms: Optional[int] = Field(None, description="Time to verify proof in milliseconds")

    # Timestamps
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_payment_method_data(self) -> GooglePaymentMethodData:
        """Convert to Google AP2 PaymentMethodData format."""
        return GooglePaymentMethodData(
            supported_methods=ZKML_SPENDING_PROOF_METHOD,
            data={
                "classification_proof": self.classification_proof,
                "policy_attestation": self.policy_attestation,
                "model_hash": self.model_hash,
                "classification": self.classification,
                "confidence": self.confidence,
                "proving_time_ms": self.proving_time_ms,
                "verification_time_ms": self.verification_time_ms,
                "timestamp": self.timestamp.isoformat(),
            }
        )

    @classmethod
    def from_payment_method_data(cls, data: GooglePaymentMethodData) -> "SpendingProof":
        """Create from Google AP2 PaymentMethodData."""
        if data.supported_methods != ZKML_SPENDING_PROOF_METHOD:
            raise ValueError(f"Invalid payment method: {data.supported_methods}")
        return cls(**data.data)


# =============================================================================
# Bridge Types - Our types that wrap/extend Google AP2
# =============================================================================

class PaymentStatus(str, Enum):
    """Payment transaction status."""
    PENDING = "pending"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class Currency(str, Enum):
    """Supported currencies (ISO 4217)."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


class AgentIdentifier(BaseModel):
    """Identifies an agent in the AP2 protocol."""
    agent_id: str = Field(..., description="Unique agent identifier (e.g., 'ap2://acme-corp-agent')")
    agent_type: Optional[str] = Field(None, description="Type of agent (e.g., 'commerce', 'payment')")
    display_name: Optional[str] = Field(None, description="Human-readable name")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MonetaryAmount(BaseModel):
    """Represents a monetary value."""
    amount_cents: int = Field(..., description="Amount in cents/minor units")
    currency: Currency = Field(default=Currency.USD)

    @property
    def amount_dollars(self) -> float:
        return self.amount_cents / 100.0

    def to_google_amount(self) -> GooglePaymentCurrencyAmount:
        """Convert to Google AP2 PaymentCurrencyAmount."""
        return GooglePaymentCurrencyAmount(
            currency=self.currency.value,
            value=self.amount_dollars
        )


class LineItem(BaseModel):
    """Individual item in a payment request."""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    quantity: int = 1
    unit_price_cents: int
    category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_google_item(self) -> GooglePaymentItem:
        """Convert to Google AP2 PaymentItem."""
        return GooglePaymentItem(
            label=self.description,
            amount=GooglePaymentCurrencyAmount(
                currency="USD",
                value=self.unit_price_cents * self.quantity / 100
            )
        )


# =============================================================================
# Payment Request/Response (wrapping Google AP2)
# =============================================================================

class PaymentRequest(BaseModel):
    """
    AP2 Payment Request - wraps Google's PaymentRequest with SpendingProof.
    """
    ap2_version: str = Field(default="1.0", description="AP2 protocol version")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Parties
    payer: AgentIdentifier = Field(..., description="Agent initiating payment")
    payee: AgentIdentifier = Field(..., description="Agent receiving payment")

    # Payment details
    amount: MonetaryAmount = Field(..., description="Total payment amount")
    line_items: List[LineItem] = Field(default_factory=list)
    memo: Optional[str] = Field(None, description="Payment description/memo")

    # ZKML Spending Proof (Novanet extension)
    spending_proof: Optional[SpendingProof] = Field(None, description="ZKML proof of compliant spending")

    # Metadata
    merchant_order_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def to_google_payment_request(self) -> GooglePaymentRequest:
        """Convert to official Google AP2 PaymentRequest."""
        method_data = []

        # Add ZKML SpendingProof as a payment method
        if self.spending_proof:
            method_data.append(self.spending_proof.to_payment_method_data())

        # Add a default payment method
        method_data.append(GooglePaymentMethodData(
            supported_methods="https://example.com/payment",
            data={"agent_payer_id": self.payer.agent_id}
        ))

        # Build display items
        display_items = [item.to_google_item() for item in self.line_items]
        if not display_items:
            display_items = [GooglePaymentItem(
                label=self.memo or "AI Agent Purchase",
                amount=self.amount.to_google_amount()
            )]

        return GooglePaymentRequest(
            method_data=method_data,
            details=GooglePaymentDetailsInit(
                id=self.request_id,
                display_items=display_items,
                total=GooglePaymentItem(
                    label="Total",
                    amount=self.amount.to_google_amount()
                )
            )
        )


class PaymentReceipt(BaseModel):
    """
    AP2 Payment Receipt - returned after successful payment.
    """
    ap2_version: str = "1.0"
    receipt_id: str = Field(default_factory=lambda: f"AP2-RCP-{uuid.uuid4().hex[:12].upper()}")

    # Reference to original request
    request_id: str
    transaction_id: str = Field(default_factory=lambda: f"AP2-TXN-{uuid.uuid4().hex[:12].upper()}")

    # Status
    status: PaymentStatus = PaymentStatus.PENDING
    status_message: Optional[str] = None

    # Payment details (echoed from request)
    payer_id: str
    payee_id: str
    amount: MonetaryAmount

    # Spending proof verification (if provided)
    spending_proof_verified: Optional[bool] = None
    verification_details: Dict[str, Any] = Field(default_factory=dict)

    # Timestamps
    processed_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Network details
    network_reference: Optional[str] = None
    settlement_info: Dict[str, Any] = Field(default_factory=dict)

    def to_google_receipt(self) -> GooglePaymentReceipt:
        """Convert to official Google AP2 PaymentReceipt."""
        if self.status == PaymentStatus.COMPLETED:
            payment_status = GoogleSuccess(
                merchant_confirmation_id=self.transaction_id,
                psp_confirmation_id=self.network_reference,
            )
        elif self.status == PaymentStatus.FAILED:
            payment_status = GoogleFailure(
                failure_message=self.status_message or "Payment failed"
            )
        else:
            payment_status = GoogleSuccess(
                merchant_confirmation_id=self.transaction_id
            )

        return GooglePaymentReceipt(
            payment_mandate_id=self.request_id,
            payment_id=self.transaction_id,
            amount=self.amount.to_google_amount(),
            payment_status=payment_status,
            payment_method_details=self.verification_details
        )


class PaymentError(BaseModel):
    """Error response for failed payment operations."""
    error_code: str
    error_message: str
    request_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# API Request/Response Models
# =============================================================================

class ProcessPaymentRequest(BaseModel):
    """Request body for processing a payment."""
    payment_request: PaymentRequest


class ProcessPaymentResponse(BaseModel):
    """Response from payment processing."""
    success: bool
    receipt: Optional[PaymentReceipt] = None
    error: Optional[PaymentError] = None

    # Include Google AP2 format for interoperability
    google_ap2_receipt: Optional[Dict[str, Any]] = None


class VerifyProofRequest(BaseModel):
    """Request to verify a spending proof."""
    spending_proof: SpendingProof


class VerifyProofResponse(BaseModel):
    """Response from proof verification."""
    valid: bool
    verification_time_ms: int
    details: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Convenience function to check AP2 types are available
# =============================================================================

def check_google_ap2_available() -> dict:
    """Check that Google AP2 types are properly imported."""
    return {
        "google_ap2_available": True,
        "types_imported": [
            "PaymentCurrencyAmount",
            "PaymentItem",
            "PaymentMethodData",
            "PaymentDetailsInit",
            "PaymentRequest",
            "PaymentResponse",
            "PaymentReceipt",
            "IntentMandate",
            "CartMandate",
            "PaymentMandate",
        ],
        "zkml_payment_method": ZKML_SPENDING_PROOF_METHOD,
    }
