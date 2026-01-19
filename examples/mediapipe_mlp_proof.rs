//! MediaPipe Text Classifier MLP Proof Example
//!
//! This example proves only the MLP portion of the MediaPipe classifier.
//! Embedding lookup and average pooling are computed outside the ZK circuit.
//!
//! Architecture:
//! - Client computes: Embedding lookup → Average pooling → pooled_embedding [1, 16]
//! - ZK Proof: Dense(16→16) → ReLU → Dense(16→2) → Softmax
#![allow(clippy::upper_case_acronyms)]

use ark_bn254::Fr;
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
use onnx_tracer::{model, tensor::Tensor};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};
use zkml_jolt_core::jolt::JoltSNARK;

type PCS = DoryCommitmentScheme;

const MAX_SEQ_LENGTH: usize = 256;
const EMBEDDING_DIM: usize = 16;
const PAD_TOKEN: i32 = 0;
const START_TOKEN: i32 = 1;
const UNKNOWN_TOKEN: i32 = 2;

/// Load vocab.json into HashMap<String, i32>
fn load_vocab(path: &str) -> Result<HashMap<String, i32>, Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let json_value: Value = serde_json::from_str(&contents)?;
    let mut vocab = HashMap::new();

    if let Value::Object(map) = json_value {
        for (word, data) in map {
            if let Some(index) = data.get("index").and_then(|v| v.as_i64()) {
                vocab.insert(word, index as i32);
            }
        }
    }

    Ok(vocab)
}

/// Load embedding weights from the full model
fn load_embeddings(model_path: &str) -> Vec<Vec<f32>> {
    // For simplicity, we'll compute embeddings using the full model
    // In production, you'd load the embedding matrix directly
    let model_fn = || model(&PathBuf::from(model_path));
    let model_instance = model_fn();

    // Get embedding dimension info from model
    // The embedding matrix is [vocab_size, embedding_dim]
    // For now, return empty - we'll compute via forward pass
    vec![]
}

/// Tokenize text
fn tokenize(text: &str, vocab: &HashMap<String, i32>) -> Vec<i32> {
    let mut tokens = vec![PAD_TOKEN; MAX_SEQ_LENGTH];
    tokens[0] = START_TOKEN;

    let re = regex::Regex::new(r"\w+|[^\w\s]").unwrap();
    let mut idx = 1;

    for cap in re.captures_iter(&text.to_lowercase()) {
        if idx >= MAX_SEQ_LENGTH {
            break;
        }
        let token = cap.get(0).unwrap().as_str();
        let token_id = vocab.get(token).copied().unwrap_or(UNKNOWN_TOKEN);
        tokens[idx] = token_id;
        idx += 1;
    }

    tokens
}

/// Compute pre-pooled embedding using full model's embedding layer
/// Returns the average pooled embedding [1, 16]
fn compute_pooled_embedding(tokens: &[i32], full_model_path: &str) -> Vec<f32> {
    // Use the full model to get the embedding output
    let input = Tensor::new(Some(tokens), &[1, MAX_SEQ_LENGTH]).unwrap();
    let model_fn = || model(&PathBuf::from(full_model_path));
    let model_instance = model_fn();
    let result = model_instance.forward(&[input]).unwrap();

    // The full model output is the final softmax output
    // We need to extract the pooled embedding before the dense layers
    // For now, compute manually using the embedding layer

    // Since we can't easily extract intermediate values, we'll use a workaround:
    // Use onnxruntime to get the pooled embedding from the full model
    vec![0.0; EMBEDDING_DIM] // Placeholder - will be filled by Python preprocessing
}

/// Scale float to i32 for tensor input (matching article_classification pattern)
fn scale_to_i32(x: f32, scale: f32) -> i32 {
    (x * scale).round() as i32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MediaPipe MLP Proof with Jolt SNARK");
    println!("====================================");
    println!("Proving: Dense(16) -> ReLU -> Dense(2) -> Softmax");
    println!();

    let working_dir = "onnx-tracer/models/mediapipe_text_classifier/";
    let mlp_model_path = format!("{working_dir}network_mlp_v4.onnx");

    // Check if MLP model exists
    if !std::path::Path::new(&mlp_model_path).exists() {
        println!("ERROR: MLP model not found at {}", mlp_model_path);
        println!("Please run: python3 scripts/build_mlp_only.py");
        return Ok(());
    }

    // Load vocab for reference
    let vocab_path = format!("{working_dir}vocab.json");
    let vocab = load_vocab(&vocab_path)?;
    println!("Loaded vocabulary with {} entries", vocab.len());

    // Test the MLP model with sample embeddings
    println!("\nTesting MLP model with sample inputs:");
    println!("======================================");

    let classes = ["negative", "positive"];

    // Pre-pooled embeddings extracted from TFLite model
    // These are real embeddings from average pooling of word embeddings, scaled by 1000
    let test_embeddings: Vec<Vec<i32>> = vec![
        // "This movie is great! I loved every moment of it." -> positive
        vec![10, 21, -256, -246, 78, 210, 49, -89, -127, -58, -11, -83, 0, -210, 32, -109],
        // "Terrible film. Complete waste of time." -> negative
        vec![-251, -86, -276, -272, 173, 170, -232, 194, 111, 85, 163, 187, 152, -226, -146, -182],
    ];
    let expected = ["positive", "negative"];

    for (i, (embedding, expected_label)) in test_embeddings.iter().zip(expected.iter()).enumerate() {
        let input = Tensor::new(Some(embedding), &[1, EMBEDDING_DIM]).unwrap();

        let model_fn = || model(&PathBuf::from(&mlp_model_path));
        let model_instance = model_fn();
        let result = model_instance.forward(&[input.clone()]).unwrap();
        let output = result.outputs[0].clone();

        let (pred_idx, max_val) = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let predicted = classes[pred_idx];
        println!(
            "Test {}: Expected {} | Predicted: {} | Confidence: {:.4}",
            i + 1,
            expected_label,
            predicted,
            max_val
        );
    }

    // Generate SNARK proof for the first embedding
    println!("\nGenerating SNARK proof for MLP:");
    println!("================================");

    let proof_embedding = &test_embeddings[0];
    let proof_input = Tensor::new(Some(proof_embedding), &[1, EMBEDDING_DIM]).unwrap();

    println!("Input: Pre-pooled embedding [1, 16]");
    println!("Preprocessing model...");

    let model_fn = || model(&PathBuf::from(&mlp_model_path));
    let preprocessing =
        JoltSNARK::<Fr, PCS, KeccakTranscript>::prover_preprocess(model_fn, 1 << 20);

    println!("Generating proof...");
    let start_time = std::time::Instant::now();
    let (snark, program_io, _debug_info) =
        JoltSNARK::<Fr, PCS, KeccakTranscript>::prove(&preprocessing, model_fn, &proof_input);
    let prove_time = start_time.elapsed();

    println!("Verifying proof...");
    let start_time = std::time::Instant::now();
    snark.verify(&(&preprocessing).into(), program_io, None)?;
    let verify_time = start_time.elapsed();

    println!("Proof verified successfully!");
    println!("Proving time: {prove_time:?}");
    println!("Verification time: {verify_time:?}");

    println!("\nMediaPipe MLP proof example completed successfully!");

    Ok(())
}
