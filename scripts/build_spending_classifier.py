#!/usr/bin/env python3
"""
Build a spending category classifier for agentic commerce.

This creates an ONNX model that classifies function call intents into
spending categories, which can then be verified via ZKML proofs.

Categories:
- data_api: API data requests (weather, routes, etc.)
- compute: Cloud compute resources
- storage: Storage services
- priority: Priority/expedited services
- blocked: Blocked/suspicious requests (always denied)

Architecture: Dense(16->16) -> ReLU -> Dense(16->8)
(8 outputs padded from 5 categories for power-of-2 JOLT compatibility)
"""

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import json
import os

# Spending categories
CATEGORIES = {
    0: "data_api",      # API data requests - allowed up to $5
    1: "compute",       # Cloud compute - allowed up to $10
    2: "storage",       # Storage services - allowed up to $2
    3: "priority",      # Priority services - allowed up to $20
    4: "blocked",       # Blocked - always denied
    # 5-7 are padding for power-of-2
}

CATEGORY_LIMITS_CENTS = {
    "data_api": 500,    # $5.00
    "compute": 1000,    # $10.00
    "storage": 200,     # $2.00
    "priority": 2000,   # $20.00
    "blocked": 0,       # Always denied
}

# Sample function call intents for training intuition
SAMPLE_INTENTS = {
    "data_api": [
        "fetch weather data for location",
        "get route optimization data",
        "query product catalog API",
        "retrieve user preferences",
        "lookup exchange rates",
    ],
    "compute": [
        "run inference on cloud GPU",
        "process batch job remotely",
        "execute ML pipeline",
        "analyze large dataset",
        "train model incrementally",
    ],
    "storage": [
        "upload logs to cloud",
        "backup sensor data",
        "store transaction history",
        "archive old records",
        "sync local cache",
    ],
    "priority": [
        "expedite delivery service",
        "fast lane access request",
        "priority queue placement",
        "rush order processing",
        "premium support ticket",
    ],
    "blocked": [
        "transfer funds externally",
        "withdraw to unknown wallet",
        "bypass security check",
        "access restricted endpoint",
        "override spending limit",
    ],
}


def create_spending_classifier_weights():
    """
    Create weights that classify spending intents.

    In a real scenario, these would be trained on actual data.
    For demo purposes, we create weights that respond to
    different input patterns representing different categories.
    """
    np.random.seed(42)

    # FC1: 16 -> 16 (feature extraction)
    # Create weights that extract features for each category
    fc1_weight = np.random.randn(16, 16).astype(np.float32) * 0.5
    fc1_bias = np.random.randn(16).astype(np.float32) * 0.1

    # FC2: 16 -> 8 (classification, 5 real + 3 padding)
    # Design weights so different feature patterns map to different categories
    fc2_weight = np.zeros((8, 16), dtype=np.float32)

    # Each category responds to different feature combinations
    # data_api (0): responds to features 0-2
    fc2_weight[0, 0:3] = [2.0, 1.5, 1.0]
    # compute (1): responds to features 3-5
    fc2_weight[1, 3:6] = [2.0, 1.5, 1.0]
    # storage (2): responds to features 6-8
    fc2_weight[2, 6:9] = [2.0, 1.5, 1.0]
    # priority (3): responds to features 9-11
    fc2_weight[3, 9:12] = [2.0, 1.5, 1.0]
    # blocked (4): responds to features 12-14
    fc2_weight[4, 12:15] = [2.0, 1.5, 1.0]

    # Add some noise for realism
    fc2_weight += np.random.randn(8, 16).astype(np.float32) * 0.1
    fc2_bias = np.random.randn(8).astype(np.float32) * 0.1

    return fc1_weight, fc1_bias, fc2_weight, fc2_bias


def build_spending_classifier_onnx(output_path: str):
    """Build the ONNX model for spending classification."""

    fc1_weight, fc1_bias, fc2_weight, fc2_bias = create_spending_classifier_weights()

    # Scale weights for integer computation (matching MediaPipe pattern)
    scale = 1000.0

    # Create ONNX graph
    # Input: [batch_size, 16] - embedding of function call intent
    input_tensor = helper.make_tensor_value_info(
        'input', TensorProto.FLOAT, ['batch_size', 16]
    )

    # Output: [batch_size, 8] - category scores (5 real + 3 padding)
    output_tensor = helper.make_tensor_value_info(
        'output', TensorProto.FLOAT, ['batch_size', 8]
    )

    # FC1 weights and bias
    fc1_weight_init = numpy_helper.from_array(
        fc1_weight.T,  # Transpose for MatMul convention
        name='fc1.weight'
    )
    fc1_bias_init = numpy_helper.from_array(fc1_bias, name='fc1.bias')

    # FC2 weights and bias
    fc2_weight_init = numpy_helper.from_array(
        fc2_weight.T,  # Transpose for MatMul convention
        name='fc2.weight'
    )
    fc2_bias_init = numpy_helper.from_array(fc2_bias, name='fc2.bias')

    # Create nodes
    # FC1: MatMul + Add
    fc1_matmul = helper.make_node(
        'MatMul',
        inputs=['input', 'fc1.weight'],
        outputs=['/fc1/MatMul_output'],
        name='/fc1/MatMul'
    )
    fc1_add = helper.make_node(
        'Add',
        inputs=['/fc1/MatMul_output', 'fc1.bias'],
        outputs=['/fc1/Add_output'],
        name='/fc1/Add'
    )

    # ReLU
    relu = helper.make_node(
        'Relu',
        inputs=['/fc1/Add_output'],
        outputs=['/relu/output'],
        name='/relu'
    )

    # FC2: MatMul + Add
    fc2_matmul = helper.make_node(
        'MatMul',
        inputs=['/relu/output', 'fc2.weight'],
        outputs=['/fc2/MatMul_output'],
        name='/fc2/MatMul'
    )
    fc2_add = helper.make_node(
        'Add',
        inputs=['/fc2/MatMul_output', 'fc2.bias'],
        outputs=['output'],
        name='/fc2/Add'
    )

    # Create graph
    graph = helper.make_graph(
        nodes=[fc1_matmul, fc1_add, relu, fc2_matmul, fc2_add],
        name='spending_classifier',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[fc1_weight_init, fc1_bias_init, fc2_weight_init, fc2_bias_init]
    )

    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 11)])
    model.ir_version = 8

    # Validate and save
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved spending classifier to {output_path}")

    return fc1_weight, fc1_bias, fc2_weight, fc2_bias


def create_sample_embeddings():
    """
    Create sample embeddings for each spending category.

    These simulate what a text embedding model would produce
    for different function call intents.
    """
    np.random.seed(123)

    embeddings = {}

    # data_api: high values in features 0-2
    embeddings["data_api"] = {
        "fetch_weather": [100, 80, 60, 10, 5, 3, 2, 1, 0, 0, 0, 0, -10, -5, -3, 0],
        "get_route": [90, 85, 70, 15, 8, 5, 3, 2, 1, 0, 0, 0, -8, -4, -2, 0],
    }

    # compute: high values in features 3-5
    embeddings["compute"] = {
        "run_inference": [5, 3, 2, 100, 85, 70, 10, 8, 5, 2, 1, 0, -5, -3, -1, 0],
        "process_batch": [8, 5, 3, 95, 80, 65, 12, 10, 7, 3, 2, 1, -7, -4, -2, 0],
    }

    # storage: high values in features 6-8
    embeddings["storage"] = {
        "upload_logs": [3, 2, 1, 5, 4, 3, 100, 85, 70, 8, 5, 3, -3, -2, -1, 0],
        "backup_data": [5, 3, 2, 8, 6, 4, 95, 80, 65, 10, 7, 4, -5, -3, -2, 0],
    }

    # priority: high values in features 9-11
    embeddings["priority"] = {
        "expedite_delivery": [2, 1, 0, 3, 2, 1, 5, 4, 3, 100, 85, 70, -2, -1, 0, 0],
        "fast_lane": [4, 2, 1, 5, 3, 2, 8, 6, 4, 95, 80, 65, -4, -2, -1, 0],
    }

    # blocked: high values in features 12-14
    embeddings["blocked"] = {
        "transfer_external": [-10, -8, -5, -5, -3, -2, -3, -2, -1, -2, -1, 0, 100, 85, 70, 0],
        "bypass_security": [-8, -6, -4, -4, -2, -1, -2, -1, 0, -1, 0, 0, 95, 80, 65, 0],
    }

    return embeddings


def save_model_artifacts(output_dir: str, weights):
    """Save model metadata and sample data."""

    fc1_weight, fc1_bias, fc2_weight, fc2_bias = weights

    # Categories metadata
    categories_meta = {
        "categories": CATEGORIES,
        "limits_cents": CATEGORY_LIMITS_CENTS,
        "num_categories": 5,
        "padded_size": 8,
    }

    with open(os.path.join(output_dir, "categories.json"), "w") as f:
        json.dump(categories_meta, f, indent=2)

    # Sample embeddings for testing
    embeddings = create_sample_embeddings()

    # Flatten for JSON
    test_cases = []
    for category, intents in embeddings.items():
        for intent_name, embedding in intents.items():
            test_cases.append({
                "name": intent_name,
                "category": category,
                "embedding": embedding,
                "expected_category_idx": list(CATEGORIES.keys())[
                    list(CATEGORIES.values()).index(category)
                ],
            })

    with open(os.path.join(output_dir, "test_cases.json"), "w") as f:
        json.dump(test_cases, f, indent=2)

    # Policy configuration
    policy = {
        "name": "default_spending_policy",
        "version": "1.0",
        "rules": [
            {
                "category": cat,
                "max_amount_cents": limit,
                "requires_approval": cat == "priority",
                "allowed": cat != "blocked",
            }
            for cat, limit in CATEGORY_LIMITS_CENTS.items()
        ],
        "default_action": "deny",
    }

    with open(os.path.join(output_dir, "policy.json"), "w") as f:
        json.dump(policy, f, indent=2)

    print(f"Saved model artifacts to {output_dir}")


def main():
    # Output directory
    output_dir = "onnx-tracer/models/spending_classifier"
    os.makedirs(output_dir, exist_ok=True)

    # Build model
    model_path = os.path.join(output_dir, "network.onnx")
    weights = build_spending_classifier_onnx(model_path)

    # Save artifacts
    save_model_artifacts(output_dir, weights)

    print("\nSpending classifier model created successfully!")
    print(f"\nCategories:")
    for idx, cat in CATEGORIES.items():
        if idx < 5:  # Only show real categories
            limit = CATEGORY_LIMITS_CENTS[cat]
            print(f"  {idx}: {cat} (limit: ${limit/100:.2f})")

    print(f"\nModel: Dense(16->16) -> ReLU -> Dense(16->8)")
    print(f"Input: 16-dim embedding of function call intent")
    print(f"Output: 8-dim category scores (5 real + 3 padding)")


if __name__ == "__main__":
    main()
