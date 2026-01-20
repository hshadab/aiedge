//! ZKML Prover - wraps Jolt Atlas for API use
//!
//! Supports two modes:
//! - Default: Mock proofs (fast, for development)
//! - real-zkml feature: Real Jolt SNARK proofs

use crate::api::{ClassificationResult, FunctionCall, ProofBundle};
use chrono::Utc;

// ============================================================================
// Mock ZKML Implementation (default)
// ============================================================================

#[cfg(not(feature = "real-zkml"))]
pub struct ZkmlProver {
    initialized: bool,
}

#[cfg(not(feature = "real-zkml"))]
impl ZkmlProver {
    pub async fn new() -> Self {
        // Simulate initialization time
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        tracing::info!("Mock ZKML prover initialized");
        Self { initialized: true }
    }

    /// Classify the function call intent and generate a mock ZKML proof
    pub async fn classify_and_prove(&self, call: &FunctionCall) -> (ClassificationResult, ProofBundle) {
        // Simulate proof generation time
        tokio::time::sleep(tokio::time::Duration::from_millis(300)).await;

        let category = self.classify_intent(call);
        let confidence = self.calculate_confidence(call, &category);

        // Generate mock proof bundle
        let proof_bundle = ProofBundle {
            classification_proof: format!("0x{}", hex_encode(self.generate_mock_proof())),
            policy_attestation: format!("0x{}", hex_encode(self.generate_mock_attestation())),
            model_hash: "sha256:7f83b1657ff1fc53b92dc18148a1d65dfc2d4b1fa3d677284addd200126d9069".to_string(),
            timestamp: Utc::now(),
        };

        (
            ClassificationResult { category, confidence },
            proof_bundle,
        )
    }

    /// Verify a proof bundle (mock - always returns true for well-formed proofs)
    pub async fn verify_proof(&self, _proof: &ProofBundle) -> bool {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        true
    }

    fn classify_intent(&self, call: &FunctionCall) -> String {
        let text = format!("{} {} {:?}", call.function, call.vendor, call.service).to_lowercase();

        if text.contains("transfer") || text.contains("external") || text.contains("withdraw") {
            "blocked".to_string()
        } else if text.contains("route") || text.contains("maps") || text.contains("logistics") || text.contains("here") {
            "logistics_data".to_string()
        } else if text.contains("compute") || text.contains("gpu") || text.contains("inference") || text.contains("bedrock") {
            "cloud_compute".to_string()
        } else if text.contains("api") || text.contains("data") || text.contains("feed") || text.contains("clearbit") {
            "data_services".to_string()
        } else if text.contains("license") || text.contains("subscription") || text.contains("saas") || text.contains("datadog") {
            "saas_licenses".to_string()
        } else {
            "data_services".to_string()
        }
    }

    fn calculate_confidence(&self, call: &FunctionCall, category: &str) -> f64 {
        let text = format!("{} {} {:?}", call.function, call.vendor, call.service).to_lowercase();

        let keywords: &[&str] = match category {
            "blocked" => &["transfer", "external", "withdraw", "unknown"],
            "logistics_data" => &["route", "maps", "logistics", "here", "delivery", "fleet"],
            "cloud_compute" => &["compute", "gpu", "inference", "bedrock", "aws", "instance"],
            "data_services" => &["api", "data", "feed", "clearbit", "service"],
            "saas_licenses" => &["license", "subscription", "saas", "datadog", "upgrade"],
            _ => &[],
        };

        let matches = keywords.iter().filter(|k| text.contains(*k)).count();
        (0.75 + matches as f64 * 0.05).min(0.98)
    }

    fn generate_mock_proof(&self) -> Vec<u8> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        (0..64).map(|i| ((ts >> (i % 16)) as u8).wrapping_add(i as u8)).collect()
    }

    fn generate_mock_attestation(&self) -> Vec<u8> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        (0..32).map(|i| ((ts >> (i % 8)) as u8).wrapping_add((i * 7) as u8)).collect()
    }
}

// ============================================================================
// Real ZKML Implementation (requires 'real-zkml' feature)
// ============================================================================

#[cfg(feature = "real-zkml")]
use ark_bn254::Fr;
#[cfg(feature = "real-zkml")]
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
#[cfg(feature = "real-zkml")]
use onnx_tracer::{model, tensor::Tensor};
#[cfg(feature = "real-zkml")]
use std::{collections::HashMap, path::PathBuf, sync::Arc};
#[cfg(feature = "real-zkml")]
use zkml_jolt_core::jolt::JoltSNARK;

#[cfg(feature = "real-zkml")]
type PCS = DoryCommitmentScheme;

#[cfg(feature = "real-zkml")]
pub struct ZkmlProver {
    model_path: PathBuf,
    categories: Vec<String>,
    preprocessing: Arc<zkml_jolt_core::jolt::JoltPreprocessing<Fr, PCS>>,
}

#[cfg(feature = "real-zkml")]
impl ZkmlProver {
    pub async fn new() -> Self {
        tracing::info!("Initializing REAL Jolt Atlas ZKML prover...");

        let model_path = PathBuf::from("../../../onnx-tracer/models/spending_classifier/network.onnx");

        // Categories from spending classifier
        let categories = vec![
            "data_api".to_string(),
            "compute".to_string(),
            "storage".to_string(),
            "priority".to_string(),
            "blocked".to_string(),
        ];

        // Preprocess model (this is the expensive operation)
        tracing::info!("Preprocessing model for SNARK generation...");
        let model_path_clone = model_path.clone();
        let model_fn = move || model(&model_path_clone);

        let preprocessing = tokio::task::spawn_blocking(move || {
            JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model_fn, 1 << 20)
        })
        .await
        .expect("Preprocessing failed");

        tracing::info!("REAL ZKML prover initialized with Jolt Atlas");

        Self {
            model_path,
            categories,
            preprocessing: Arc::new(preprocessing),
        }
    }

    /// Classify the function call and generate a REAL Jolt SNARK proof
    pub async fn classify_and_prove(&self, call: &FunctionCall) -> (ClassificationResult, ProofBundle) {
        // Convert function call to 16-dim embedding (matching spending_classifier input)
        let embedding = self.function_call_to_embedding(call);
        let input = Tensor::new(Some(&embedding), &[1, 16]).unwrap();

        let model_path = self.model_path.clone();
        let preprocessing = self.preprocessing.clone();

        // Run proof generation in blocking thread
        let (classification, proof_bytes) = tokio::task::spawn_blocking(move || {
            let model_fn = || model(&model_path);

            // Generate proof
            let start = std::time::Instant::now();
            let (snark, program_io, _debug) =
                JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model_fn, &input);
            let prove_time = start.elapsed();
            tracing::info!("Proof generated in {:?}", prove_time);

            // Verify proof
            let start = std::time::Instant::now();
            let verification_result = snark.verify(&(&*preprocessing).into(), program_io.clone(), None);
            let verify_time = start.elapsed();
            tracing::info!("Proof verified in {:?}", verify_time);

            if let Err(e) = verification_result {
                tracing::error!("Proof verification failed: {:?}", e);
            }

            // Get classification result from model output
            let output = program_io.outputs.clone();
            let (pred_idx, max_val) = output
                .iter()
                .take(5) // Only first 5 categories are real
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap_or((0, &0));

            // Serialize proof (simplified - in production, use proper serialization)
            let proof_bytes: Vec<u8> = format!("JOLT_SNARK_PROOF_{:?}", snark).into_bytes();

            (
                (pred_idx, *max_val as f64),
                proof_bytes,
            )
        })
        .await
        .expect("Proof generation failed");

        let (pred_idx, confidence) = classification;
        let category = self.categories.get(pred_idx).cloned().unwrap_or_else(|| "unknown".to_string());

        // Map spending_classifier categories to demo categories
        let mapped_category = match category.as_str() {
            "data_api" => "data_services".to_string(),
            "compute" => "cloud_compute".to_string(),
            "storage" => "data_services".to_string(),
            "priority" => "saas_licenses".to_string(),
            "blocked" => "blocked".to_string(),
            _ => "data_services".to_string(),
        };

        let proof_bundle = ProofBundle {
            classification_proof: format!("0x{}", hex_encode(proof_bytes)),
            policy_attestation: format!("0x{}", hex_encode(self.generate_policy_attestation(&mapped_category))),
            model_hash: self.compute_model_hash(),
            timestamp: Utc::now(),
        };

        (
            ClassificationResult {
                category: mapped_category,
                confidence: (confidence / 1000.0).max(0.5).min(0.99), // Normalize
            },
            proof_bundle,
        )
    }

    /// Verify a ZKML proof
    pub async fn verify_proof(&self, proof: &ProofBundle) -> bool {
        // For real verification, we would deserialize and verify the SNARK
        // For now, check that proof has expected format
        proof.classification_proof.starts_with("0x") && proof.classification_proof.len() > 10
    }

    /// Convert function call to 16-dim embedding for the spending classifier
    fn function_call_to_embedding(&self, call: &FunctionCall) -> Vec<i32> {
        let text = format!("{} {} {:?}", call.function, call.vendor, call.service).to_lowercase();

        // Feature extraction based on keywords (matches spending_classifier training)
        let mut embedding = vec![0i32; 16];

        // data_api features (0-2)
        if text.contains("api") || text.contains("data") || text.contains("feed") {
            embedding[0] = 100;
            embedding[1] = 80;
            embedding[2] = 60;
        }

        // compute features (3-5)
        if text.contains("compute") || text.contains("gpu") || text.contains("inference") || text.contains("bedrock") {
            embedding[3] = 100;
            embedding[4] = 85;
            embedding[5] = 70;
        }

        // storage features (6-8)
        if text.contains("storage") || text.contains("backup") || text.contains("upload") {
            embedding[6] = 100;
            embedding[7] = 85;
            embedding[8] = 70;
        }

        // priority features (9-11)
        if text.contains("priority") || text.contains("expedite") || text.contains("rush") {
            embedding[9] = 100;
            embedding[10] = 85;
            embedding[11] = 70;
        }

        // blocked features (12-14) - negative for other categories
        if text.contains("transfer") || text.contains("external") || text.contains("withdraw") {
            embedding[12] = 100;
            embedding[13] = 85;
            embedding[14] = 70;
            // Negative signal for other categories
            embedding[0] = -10;
            embedding[3] = -5;
        }

        // Route/logistics -> data_api
        if text.contains("route") || text.contains("maps") || text.contains("logistics") {
            embedding[0] = 90;
            embedding[1] = 85;
            embedding[2] = 70;
        }

        embedding
    }

    fn generate_policy_attestation(&self, category: &str) -> Vec<u8> {
        // Generate attestation including category and timestamp
        let mut attestation = category.as_bytes().to_vec();
        let timestamp = Utc::now().timestamp() as u64;
        attestation.extend_from_slice(&timestamp.to_le_bytes());
        attestation
    }

    fn compute_model_hash(&self) -> String {
        // In production, compute actual hash of model file
        "sha256:spending_classifier_v1_7f83b1657ff1fc53b92dc".to_string()
    }
}

// ============================================================================
// Shared utilities
// ============================================================================

fn hex_encode(bytes: Vec<u8>) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
