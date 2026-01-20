//! AP2 (Agent-to-Payment) Gateway Client
//!
//! Connects to the AP2 Python service (port 3002) for payment processing.
//! Falls back to mock implementation if service is unavailable.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// AP2 service endpoint
const AP2_SERVICE_URL: &str = "http://localhost:3002";

pub struct Ap2Gateway {
    client: reqwest::Client,
    service_available: bool,
}

#[derive(Debug, Clone)]
pub struct Ap2Result {
    pub success: bool,
    pub transaction_id: Option<String>,
    pub message: String,
    pub proof_verified: Option<bool>,
}

/// Request to the AP2 quick-pay endpoint
#[derive(Debug, Serialize)]
struct QuickPayRequest {
    vendor: String,
    amount_cents: u64,
    payer_id: String,
    memo: Option<String>,
    spending_proof: Option<SpendingProofPayload>,
}

#[derive(Debug, Serialize)]
struct SpendingProofPayload {
    classification_proof: String,
    policy_attestation: String,
    model_hash: String,
    classification_result: Option<String>,
    confidence: Option<f64>,
}

/// Response from AP2 quick-pay endpoint
#[derive(Debug, Deserialize)]
struct QuickPayResponse {
    success: bool,
    transaction_id: Option<String>,
    amount_cents: Option<u64>,
    vendor: Option<String>,
    proof_verified: Option<bool>,
    message: Option<String>,
    error: Option<String>,
}

impl Ap2Gateway {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
            service_available: true,
        }
    }

    /// Check if AP2 service is available
    pub async fn check_health(&mut self) -> bool {
        match self.client.get(format!("{}/health", AP2_SERVICE_URL)).send().await {
            Ok(resp) => {
                self.service_available = resp.status().is_success();
                self.service_available
            }
            Err(e) => {
                tracing::warn!("AP2 service not available: {}", e);
                self.service_available = false;
                false
            }
        }
    }

    /// Process a payment via the AP2 protocol
    ///
    /// If the AP2 Python service is running, uses the real AP2 protocol.
    /// Otherwise, falls back to mock implementation.
    pub fn process_payment(&self, vendor: &str, amount_cents: u64) -> Ap2Result {
        // Try async version in blocking context
        let rt = tokio::runtime::Handle::try_current();
        if let Ok(handle) = rt {
            let client = self.client.clone();
            let vendor = vendor.to_string();

            // Try to call AP2 service
            match handle.block_on(async {
                Self::call_ap2_service(&client, &vendor, amount_cents, None).await
            }) {
                Ok(result) => return result,
                Err(e) => {
                    tracing::warn!("AP2 service call failed, using mock: {}", e);
                }
            }
        }

        // Fallback to mock implementation
        self.mock_process_payment(vendor, amount_cents)
    }

    /// Process payment with spending proof
    pub async fn process_payment_with_proof(
        &self,
        vendor: &str,
        amount_cents: u64,
        proof: Option<Ap2SpendingProof>,
    ) -> Ap2Result {
        // Try AP2 service first
        match Self::call_ap2_service(&self.client, vendor, amount_cents, proof.clone()).await {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!("AP2 service call failed, using mock: {}", e);
                self.mock_process_payment(vendor, amount_cents)
            }
        }
    }

    /// Call the real AP2 Python service
    async fn call_ap2_service(
        client: &reqwest::Client,
        vendor: &str,
        amount_cents: u64,
        proof: Option<Ap2SpendingProof>,
    ) -> Result<Ap2Result, Box<dyn std::error::Error + Send + Sync>> {
        let spending_proof = proof.map(|p| SpendingProofPayload {
            classification_proof: p.classification_proof,
            policy_attestation: p.policy_attestation,
            model_hash: p.model_hash,
            classification_result: None,
            confidence: None,
        });

        let request = QuickPayRequest {
            vendor: vendor.to_string(),
            amount_cents,
            payer_id: "acme-corp-agent".to_string(),
            memo: Some(format!("Payment to {} for ${:.2}", vendor, amount_cents as f64 / 100.0)),
            spending_proof,
        };

        let response = client
            .post(format!("{}/api/v1/demo/quick-pay", AP2_SERVICE_URL))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("AP2 service error: {} - {}", status, body).into());
        }

        let result: QuickPayResponse = response.json().await?;

        Ok(Ap2Result {
            success: result.success,
            transaction_id: result.transaction_id,
            message: result.message.or(result.error).unwrap_or_else(|| "Unknown".to_string()),
            proof_verified: result.proof_verified,
        })
    }

    /// Mock payment processing (fallback when AP2 service unavailable)
    fn mock_process_payment(&self, vendor: &str, amount_cents: u64) -> Ap2Result {
        // Check for blocked vendors
        if vendor.contains("unknown") || vendor.contains("sketchy") {
            return Ap2Result {
                success: false,
                transaction_id: None,
                message: "AP2: Vendor not registered in payment network".to_string(),
                proof_verified: None,
            };
        }

        // Simulate successful payment
        let tx_id = format!(
            "AP2-TXN-{}",
            Uuid::new_v4().to_string().replace("-", "")[..12].to_uppercase()
        );

        Ap2Result {
            success: true,
            transaction_id: Some(tx_id),
            message: format!(
                "AP2 (mock): Payment of ${:.2} processed successfully",
                amount_cents as f64 / 100.0
            ),
            proof_verified: None,
        }
    }
}

/// AP2 Request format (matching Google's spec)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ap2Request {
    pub ap2_version: String,
    pub transaction_id: String,
    pub payment: Ap2PaymentDetails,
    pub spending_proof: Option<Ap2SpendingProof>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ap2PaymentDetails {
    pub amount_cents: u64,
    pub currency: String,
    pub payer_id: String,
    pub payee_id: String,
    pub memo: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ap2SpendingProof {
    pub classification_proof: String,
    pub policy_attestation: String,
    pub model_hash: String,
}

impl Ap2Request {
    pub fn new(
        transaction_id: &str,
        payer: &str,
        payee: &str,
        amount_cents: u64,
        memo: Option<String>,
        proof: Option<Ap2SpendingProof>,
    ) -> Self {
        Self {
            ap2_version: "1.0".to_string(),
            transaction_id: transaction_id.to_string(),
            payment: Ap2PaymentDetails {
                amount_cents,
                currency: "USD".to_string(),
                payer_id: format!("ap2://{}", payer),
                payee_id: format!("ap2://{}", payee),
                memo,
            },
            spending_proof: proof,
            metadata: HashMap::new(),
        }
    }
}
