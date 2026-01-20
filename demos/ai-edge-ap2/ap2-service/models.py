"""
AP2 Protocol Data Models

Based on Google's Agent Payments Protocol (AP2) specification.
https://github.com/google-agentic-commerce/AP2
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid


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
    """Supported currencies."""
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


class SpendingProof(BaseModel):
    """
    ZKML Spending Proof attached to AP2 payment requests.

    This proves that an AI agent's spending decision was:
    1. Classified by an approved ML model
    2. Compliant with enterprise spending policy
    3. Cryptographically verifiable
    """
    classification_proof: str = Field(..., description="SNARK proof of intent classification")
    policy_attestation: str = Field(..., description="Attestation that policy was checked")
    model_hash: str = Field(..., description="Hash of the ML model used for classification")
    classification_result: Optional[str] = Field(None, description="The classified spending category")
    confidence: Optional[float] = Field(None, description="Classification confidence score")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class LineItem(BaseModel):
    """Individual item in a payment request."""
    item_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    quantity: int = 1
    unit_price_cents: int
    category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PaymentRequest(BaseModel):
    """
    AP2 Payment Request

    Represents a payment request from a payer agent to a payee agent.
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


class PaymentReceipt(BaseModel):
    """
    AP2 Payment Receipt

    Returned after successful payment processing.
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


class PaymentError(BaseModel):
    """Error response for failed payment operations."""
    error_code: str
    error_message: str
    request_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


# API Request/Response models

class ProcessPaymentRequest(BaseModel):
    """Request body for processing a payment."""
    payment_request: PaymentRequest


class ProcessPaymentResponse(BaseModel):
    """Response from payment processing."""
    success: bool
    receipt: Optional[PaymentReceipt] = None
    error: Optional[PaymentError] = None


class VerifyProofRequest(BaseModel):
    """Request to verify a spending proof."""
    spending_proof: SpendingProof


class VerifyProofResponse(BaseModel):
    """Response from proof verification."""
    valid: bool
    verification_time_ms: int
    details: Dict[str, Any] = Field(default_factory=dict)
