//! MediaPipe Text Classifier Example
//!
//! This example demonstrates how to use the Jolt SNARK system with the MediaPipe
//! Average Word Embedding text classifier for sentiment analysis.
//!
//! Model: MediaPipe Average Word Embedding (float32)
//! - Input: [1, 256] token indices
//! - Output: [1, 2] probabilities (negative, positive)
#![allow(clippy::upper_case_acronyms)]

use ark_bn254::Fr;
use jolt_core::{poly::commitment::dory::DoryCommitmentScheme, transcripts::KeccakTranscript};
use onnx_tracer::{model, tensor::Tensor};
use serde_json::Value;
use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};
use zkml_jolt_core::jolt::JoltSNARK;

type PCS = DoryCommitmentScheme;

const MAX_SEQ_LENGTH: usize = 256;
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

/// Tokenize text using MediaPipe's tokenization approach:
/// - Lowercase
/// - Split on whitespace and punctuation
/// - Add <START> token at beginning
/// - Pad or truncate to MAX_SEQ_LENGTH
fn tokenize(text: &str, vocab: &HashMap<String, i32>) -> Vec<i32> {
    let mut tokens = vec![PAD_TOKEN; MAX_SEQ_LENGTH];

    // Start with <START> token
    tokens[0] = START_TOKEN;

    // Tokenize using regex (similar to MediaPipe)
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("MediaPipe Text Classifier with Jolt SNARK");
    println!("==========================================");

    let working_dir = "onnx-tracer/models/mediapipe_text_classifier/";

    // Load the vocab mapping from JSON
    let vocab_path = format!("{working_dir}/vocab.json");
    let vocab = load_vocab(&vocab_path).expect("Failed to load vocab");
    println!("Loaded vocabulary with {} entries", vocab.len());

    // Test inputs and expected outputs
    let test_cases = [
        ("This movie is great! I loved every moment of it.", "positive"),
        ("Terrible film. Complete waste of time.", "negative"),
        ("The acting was superb and the story was compelling.", "positive"),
        ("Boring and predictable. Would not recommend.", "negative"),
        ("An absolute masterpiece of cinema.", "positive"),
        ("The worst movie I have ever seen.", "negative"),
    ];

    let classes = ["negative", "positive"];
    println!("\nTesting model outputs:");
    println!("======================");

    // Test all inputs to verify model accuracy
    let mut correct_predictions = 0;
    for (i, (input_text, expected)) in test_cases.iter().enumerate() {
        let input_tokens = tokenize(input_text, &vocab);
        let input = Tensor::new(Some(&input_tokens), &[1, MAX_SEQ_LENGTH]).unwrap();

        // Load and run model
        let model_fn = || model(&PathBuf::from(format!("{working_dir}network.onnx")));
        let model_instance = model_fn();
        let result = model_instance.forward(&[input.clone()]).unwrap();
        let output = result.outputs[0].clone();

        // Get prediction (output is [1, 2] for [negative, positive])
        let (pred_idx, max_val) = output
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        let predicted = classes[pred_idx];
        let is_correct = predicted == *expected;

        if is_correct {
            correct_predictions += 1;
        }

        println!("\nTest {}: '{}'", i + 1, input_text);
        println!(
            "  Expected: {} | Predicted: {} | Confidence: {:.4} | {}",
            expected,
            predicted,
            max_val,
            if is_correct { "CORRECT" } else { "INCORRECT" }
        );
    }

    let accuracy = (correct_predictions as f32 / test_cases.len() as f32) * 100.0;
    println!(
        "\nModel Accuracy: {}/{} ({:.1}%)",
        correct_predictions,
        test_cases.len(),
        accuracy
    );

    // Generate a proof for the first test case
    println!("\nGenerating SNARK for first example:");
    println!("==========================================");

    let proof_text = test_cases[0].0;
    let proof_input_tokens = tokenize(proof_text, &vocab);
    let proof_input = Tensor::new(Some(&proof_input_tokens), &[1, MAX_SEQ_LENGTH]).unwrap();

    println!("Input: '{proof_text}'");
    println!("Preprocessing model...");

    let model_fn = || model(&PathBuf::from(format!("{working_dir}network.onnx")));
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

    println!("\nMediaPipe text classifier example completed successfully!");

    Ok(())
}
