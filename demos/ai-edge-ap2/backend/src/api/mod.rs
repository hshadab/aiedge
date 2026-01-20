//! API handlers and state management

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::zkml::ZkmlProver;
use crate::ap2::Ap2Gateway;

// ============================================================================
// State
// ============================================================================

pub struct AppState {
    pub zkml_prover: Option<ZkmlProver>,
    pub transactions: Vec<Transaction>,
    pub policies: HashMap<String, SpendingPolicy>,
    pub devices: HashMap<String, Device>,
    pub ap2_gateway: Ap2Gateway,
}

impl AppState {
    pub fn new() -> Self {
        let mut policies = HashMap::new();
        policies.insert(
            "acme-fleet-policy".to_string(),
            SpendingPolicy::default_fleet_policy(),
        );

        let mut devices = HashMap::new();
        devices.insert(
            "truck-127".to_string(),
            Device {
                id: "truck-127".to_string(),
                name: "Delivery Truck #127".to_string(),
                device_type: "truck".to_string(),
                location: Location { lat: 37.7749, lng: -122.4194 },
                policy_id: "acme-fleet-policy".to_string(),
                status: "active".to_string(),
                spent_today_cents: 84700,
            },
        );

        Self {
            zkml_prover: None,
            transactions: Vec::new(),
            policies,
            devices,
            ap2_gateway: Ap2Gateway::new(),
        }
    }

    pub async fn initialize_zkml(&mut self) {
        self.zkml_prover = Some(ZkmlProver::new().await);
    }
}

// ============================================================================
// Data Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub function: String,
    pub vendor: String,
    pub amount_cents: u64,
    pub service: Option<String>,
    pub context: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub category: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyCheckResult {
    pub allowed: bool,
    pub reason: String,
    pub checks_passed: Vec<String>,
    pub checks_failed: Vec<String>,
    pub requires_human_approval: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofBundle {
    pub classification_proof: String,
    pub policy_attestation: String,
    pub model_hash: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub device_id: String,
    pub device_name: String,
    pub function_call: FunctionCall,
    pub classification: ClassificationResult,
    pub policy_check: PolicyCheckResult,
    pub proof_bundle: Option<ProofBundle>,
    pub payment_status: String,
    pub ap2_transaction_id: Option<String>,
    pub proving_time_ms: u128,
    pub verification_time_ms: Option<u128>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendingPolicy {
    pub id: String,
    pub name: String,
    pub organization_id: String,
    pub categories: HashMap<String, CategoryConfig>,
    pub approved_vendors: Vec<String>,
    pub daily_limit_cents: u64,
    pub monthly_limit_cents: u64,
    pub human_approval_threshold_cents: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryConfig {
    pub allowed: bool,
    pub limit_cents: u64,
}

impl SpendingPolicy {
    pub fn default_fleet_policy() -> Self {
        let mut categories = HashMap::new();
        categories.insert("data_services".to_string(), CategoryConfig { allowed: true, limit_cents: 25000 });
        categories.insert("cloud_compute".to_string(), CategoryConfig { allowed: true, limit_cents: 100000 });
        categories.insert("logistics_data".to_string(), CategoryConfig { allowed: true, limit_cents: 25000 });
        categories.insert("saas_licenses".to_string(), CategoryConfig { allowed: true, limit_cents: 200000 });
        categories.insert("blocked".to_string(), CategoryConfig { allowed: false, limit_cents: 0 });

        Self {
            id: "acme-fleet-policy".to_string(),
            name: "Acme Fleet Spending Policy".to_string(),
            organization_id: "org_acme_logistics".to_string(),
            categories,
            approved_vendors: vec![
                "vendor:here_routing".to_string(),
                "vendor:google_maps".to_string(),
                "vendor:aws_bedrock".to_string(),
                "vendor:clearbit_api".to_string(),
                "vendor:datadog".to_string(),
            ],
            daily_limit_cents: 500000,
            monthly_limit_cents: 5000000,
            human_approval_threshold_cents: 100000,
        }
    }

    pub fn check_compliance(&self, category: &str, amount_cents: u64, vendor: &str) -> PolicyCheckResult {
        let mut checks_passed = Vec::new();
        let mut checks_failed = Vec::new();

        // Check 1: Category allowed
        let cat_config = self.categories.get(category);
        if let Some(config) = cat_config {
            if !config.allowed {
                checks_failed.push("category_allowed".to_string());
                return PolicyCheckResult {
                    allowed: false,
                    reason: format!("Category '{}' is blocked by policy", category),
                    checks_passed,
                    checks_failed,
                    requires_human_approval: false,
                };
            }
            checks_passed.push("category_allowed".to_string());

            // Check 2: Category limit
            if amount_cents > config.limit_cents {
                checks_failed.push("category_limit".to_string());
                return PolicyCheckResult {
                    allowed: false,
                    reason: format!(
                        "Amount ${:.2} exceeds category limit ${:.2}",
                        amount_cents as f64 / 100.0,
                        config.limit_cents as f64 / 100.0
                    ),
                    checks_passed,
                    checks_failed,
                    requires_human_approval: false,
                };
            }
            checks_passed.push("category_limit".to_string());
        } else {
            checks_failed.push("category_allowed".to_string());
            return PolicyCheckResult {
                allowed: false,
                reason: format!("Unknown category '{}'", category),
                checks_passed,
                checks_failed,
                requires_human_approval: false,
            };
        }

        // Check 3: Vendor approved
        if !self.approved_vendors.contains(&vendor.to_string()) {
            checks_failed.push("vendor_approved".to_string());
            return PolicyCheckResult {
                allowed: false,
                reason: format!("Vendor '{}' not on approved list", vendor),
                checks_passed,
                checks_failed,
                requires_human_approval: false,
            };
        }
        checks_passed.push("vendor_approved".to_string());

        // Check 4: Human approval threshold
        let requires_human = amount_cents > self.human_approval_threshold_cents;

        PolicyCheckResult {
            allowed: true,
            reason: if requires_human {
                format!("Approved - requires human confirmation (>${:.2})",
                    self.human_approval_threshold_cents as f64 / 100.0)
            } else {
                "Approved - all policy checks passed".to_string()
            },
            checks_passed,
            checks_failed,
            requires_human_approval: requires_human,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub device_type: String,
    pub location: Location,
    pub policy_id: String,
    pub status: String,
    pub spent_today_cents: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub lat: f64,
    pub lng: f64,
}

// ============================================================================
// Request/Response Types
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct ClassifyRequest {
    pub function_call: FunctionCall,
    pub device_id: String,
    pub policy_id: String,
}

#[derive(Debug, Serialize)]
pub struct ClassifyResponse {
    pub transaction_id: String,
    pub classification: ClassificationResult,
    pub policy_check: PolicyCheckResult,
    pub proof_bundle: Option<ProofBundle>,
    pub proving_time_ms: u128,
    pub payment_ready: bool,
}

#[derive(Debug, Deserialize)]
pub struct VerifyRequest {
    pub proof_bundle: ProofBundle,
}

#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub verification_time_ms: u128,
    pub verified_claims: VerifiedClaims,
}

#[derive(Debug, Serialize)]
pub struct VerifiedClaims {
    pub classification_correct: bool,
    pub policy_compliant: bool,
    pub model_approved: bool,
}

#[derive(Debug, Deserialize)]
pub struct Ap2PaymentRequest {
    pub transaction_id: String,
}

#[derive(Debug, Serialize)]
pub struct Ap2PaymentResponse {
    pub success: bool,
    pub ap2_transaction_id: Option<String>,
    pub message: String,
}

// ============================================================================
// Handlers
// ============================================================================

pub async fn health_check() -> &'static str {
    "OK"
}

pub async fn classify_intent(
    State(state): State<Arc<RwLock<AppState>>>,
    Json(request): Json<ClassifyRequest>,
) -> Result<Json<ClassifyResponse>, (StatusCode, String)> {
    let mut state = state.write().await;

    // Get policy
    let policy = state.policies.get(&request.policy_id)
        .ok_or((StatusCode::NOT_FOUND, "Policy not found".to_string()))?
        .clone();

    // Get device
    let device = state.devices.get(&request.device_id)
        .ok_or((StatusCode::NOT_FOUND, "Device not found".to_string()))?
        .clone();

    // Classify and generate proof
    let start = std::time::Instant::now();

    let (classification, proof_bundle) = if let Some(ref prover) = state.zkml_prover {
        let result = prover.classify_and_prove(&request.function_call).await;
        (result.0, Some(result.1))
    } else {
        // Fallback if ZKML not initialized yet
        (
            ClassificationResult {
                category: classify_by_keywords(&request.function_call),
                confidence: 0.85,
            },
            None,
        )
    };

    let proving_time_ms = start.elapsed().as_millis();

    // Check policy
    let policy_check = policy.check_compliance(
        &classification.category,
        request.function_call.amount_cents,
        &request.function_call.vendor,
    );

    // Create transaction
    let transaction_id = Uuid::new_v4().to_string();
    let transaction = Transaction {
        id: transaction_id.clone(),
        timestamp: Utc::now(),
        device_id: device.id.clone(),
        device_name: device.name.clone(),
        function_call: request.function_call,
        classification: classification.clone(),
        policy_check: policy_check.clone(),
        proof_bundle: proof_bundle.clone(),
        payment_status: if policy_check.allowed { "ready".to_string() } else { "denied".to_string() },
        ap2_transaction_id: None,
        proving_time_ms,
        verification_time_ms: None,
    };

    state.transactions.push(transaction);

    Ok(Json(ClassifyResponse {
        transaction_id,
        classification,
        policy_check: policy_check.clone(),
        proof_bundle,
        proving_time_ms,
        payment_ready: policy_check.allowed,
    }))
}

pub async fn verify_proof(
    State(state): State<Arc<RwLock<AppState>>>,
    Json(request): Json<VerifyRequest>,
) -> Json<VerifyResponse> {
    let state = state.read().await;

    let start = std::time::Instant::now();

    let valid = if let Some(ref prover) = state.zkml_prover {
        prover.verify_proof(&request.proof_bundle).await
    } else {
        // Fallback verification
        !request.proof_bundle.classification_proof.is_empty()
    };

    let verification_time_ms = start.elapsed().as_millis();

    Json(VerifyResponse {
        valid,
        verification_time_ms,
        verified_claims: VerifiedClaims {
            classification_correct: valid,
            policy_compliant: valid,
            model_approved: valid,
        },
    })
}

pub async fn get_policy(
    State(state): State<Arc<RwLock<AppState>>>,
    Path(policy_id): Path<String>,
) -> Result<Json<SpendingPolicy>, (StatusCode, String)> {
    let state = state.read().await;
    state.policies.get(&policy_id)
        .cloned()
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "Policy not found".to_string()))
}

pub async fn list_policies(
    State(state): State<Arc<RwLock<AppState>>>,
) -> Json<Vec<SpendingPolicy>> {
    let state = state.read().await;
    Json(state.policies.values().cloned().collect())
}

pub async fn list_transactions(
    State(state): State<Arc<RwLock<AppState>>>,
) -> Json<Vec<Transaction>> {
    let state = state.read().await;
    Json(state.transactions.clone())
}

pub async fn get_transaction(
    State(state): State<Arc<RwLock<AppState>>>,
    Path(tx_id): Path<String>,
) -> Result<Json<Transaction>, (StatusCode, String)> {
    let state = state.read().await;
    state.transactions.iter()
        .find(|t| t.id == tx_id)
        .cloned()
        .map(Json)
        .ok_or((StatusCode::NOT_FOUND, "Transaction not found".to_string()))
}

pub async fn process_ap2_payment(
    State(state): State<Arc<RwLock<AppState>>>,
    Json(request): Json<Ap2PaymentRequest>,
) -> Result<Json<Ap2PaymentResponse>, (StatusCode, String)> {
    let mut state = state.write().await;

    // Find transaction
    let tx_idx = state.transactions.iter()
        .position(|t| t.id == request.transaction_id)
        .ok_or((StatusCode::NOT_FOUND, "Transaction not found".to_string()))?;

    let transaction = &state.transactions[tx_idx];

    // Check if payment is allowed
    if !transaction.policy_check.allowed {
        return Ok(Json(Ap2PaymentResponse {
            success: false,
            ap2_transaction_id: None,
            message: format!("Payment denied: {}", transaction.policy_check.reason),
        }));
    }

    // Process via AP2 gateway
    let ap2_result = state.ap2_gateway.process_payment(
        &transaction.function_call.vendor,
        transaction.function_call.amount_cents,
    );

    // Update transaction
    state.transactions[tx_idx].payment_status = if ap2_result.success {
        "completed".to_string()
    } else {
        "failed".to_string()
    };
    state.transactions[tx_idx].ap2_transaction_id = ap2_result.transaction_id.clone();

    Ok(Json(Ap2PaymentResponse {
        success: ap2_result.success,
        ap2_transaction_id: ap2_result.transaction_id,
        message: ap2_result.message,
    }))
}

pub async fn trigger_scenario(
    State(state): State<Arc<RwLock<AppState>>>,
    Path(scenario): Path<String>,
) -> Json<ClassifyResponse> {
    let function_call = match scenario.as_str() {
        "route_purchase" => FunctionCall {
            function: "purchase_data_service".to_string(),
            vendor: "vendor:here_routing".to_string(),
            amount_cents: 1200,
            service: Some("premium_route_optimization".to_string()),
            context: [
                ("delivery_id".to_string(), serde_json::json!("DEL-4521")),
                ("urgency".to_string(), serde_json::json!("high")),
            ].into_iter().collect(),
        },
        "blocked_transfer" => FunctionCall {
            function: "transfer_external".to_string(),
            vendor: "vendor:unknown_external".to_string(),
            amount_cents: 500000,
            service: None,
            context: [
                ("reason".to_string(), serde_json::json!("external transfer")),
            ].into_iter().collect(),
        },
        "over_limit" => FunctionCall {
            function: "purchase_compute".to_string(),
            vendor: "vendor:aws_bedrock".to_string(),
            amount_cents: 150000,
            service: Some("gpu_cluster".to_string()),
            context: HashMap::new(),
        },
        "unapproved_vendor" => FunctionCall {
            function: "purchase_data_service".to_string(),
            vendor: "vendor:sketchy_api".to_string(),
            amount_cents: 5000,
            service: Some("data_feed".to_string()),
            context: HashMap::new(),
        },
        _ => FunctionCall {
            function: "purchase_data_service".to_string(),
            vendor: "vendor:here_routing".to_string(),
            amount_cents: 1200,
            service: Some("route_data".to_string()),
            context: HashMap::new(),
        },
    };

    let request = ClassifyRequest {
        function_call,
        device_id: "truck-127".to_string(),
        policy_id: "acme-fleet-policy".to_string(),
    };

    // Call classify_intent
    match classify_intent(State(state), Json(request)).await {
        Ok(response) => response,
        Err((_, msg)) => Json(ClassifyResponse {
            transaction_id: "error".to_string(),
            classification: ClassificationResult {
                category: "error".to_string(),
                confidence: 0.0,
            },
            policy_check: PolicyCheckResult {
                allowed: false,
                reason: msg,
                checks_passed: vec![],
                checks_failed: vec!["error".to_string()],
                requires_human_approval: false,
            },
            proof_bundle: None,
            proving_time_ms: 0,
            payment_ready: false,
        }),
    }
}

pub async fn reset_demo(
    State(state): State<Arc<RwLock<AppState>>>,
) -> &'static str {
    let mut state = state.write().await;
    state.transactions.clear();
    "Demo reset"
}

// Helper function for keyword-based classification fallback
fn classify_by_keywords(call: &FunctionCall) -> String {
    let text = format!("{} {} {:?}", call.function, call.vendor, call.service).to_lowercase();

    if text.contains("transfer") || text.contains("external") || text.contains("withdraw") {
        "blocked".to_string()
    } else if text.contains("route") || text.contains("maps") || text.contains("logistics") {
        "logistics_data".to_string()
    } else if text.contains("compute") || text.contains("gpu") || text.contains("inference") {
        "cloud_compute".to_string()
    } else if text.contains("api") || text.contains("data") || text.contains("feed") {
        "data_services".to_string()
    } else if text.contains("license") || text.contains("subscription") || text.contains("saas") {
        "saas_licenses".to_string()
    } else {
        "data_services".to_string()
    }
}
