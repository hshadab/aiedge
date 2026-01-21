# AI Edge → AP2: Verifiable Autonomous Agent Payments

*Powered by [Jolt Atlas zkML](https://github.com/ICME-Lab/jolt-atlas)*

AI agents that can spend money autonomously—with cryptographic proof they followed your rules.

This demo shows an end-to-end flow: an on-device AI agent makes a purchase decision, a classifier categorizes the spend, a zkML proof verifies the classification was correct, and the payment executes via Google AP2—all with real proofs, real inference, and real Google types.

## Quick Start

```bash
# Terminal 1: Start backend (Rust + zkML)
cd demos/ai-edge-ap2
RUST_LOG=info cargo run -p ai-edge-ap2-demo --release

# Terminal 2: Start frontend
cd demos/ai-edge-ap2/frontend
python3 -m http.server 3000
```

Open http://localhost:3000 and press **Space** to run the demo.

## What You'll See

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐     ┌──────────────┐
│    AI Agent     │────▶│  AI Edge Classifier  │────▶│  zkML Prover    │────▶│  AP2 Payment │
│   (Simulated)   │     │   (Real MediaPipe)   │     │ (Real ~5s prove)│     │ (Real types) │
└─────────────────┘     └──────────────────────┘     └─────────────────┘     └──────────────┘
```

1. **AI Agent** decides to purchase route data ($12 from HERE Maps)
2. **AI Edge Classifier** categorizes it as "data_services" (~1ms inference)
3. **zkML Prover** generates a SNARK proof the classification was correct (~5s)
4. **Policy Engine** checks enterprise spending rules (category allowed, under limit, approved vendor)
5. **AP2 Payment** executes with the SpendingProof attached

## What's Real

| Component | Status | Details |
|-----------|--------|---------|
| Google AI Edge | **REAL** | MediaPipe text classifier, ~1ms inference |
| zkML Proofs | **REAL** | Jolt Atlas SNARK, ~5s prove, ~2s verify |
| Google AP2 Types | **REAL** | From [google-agentic-commerce/AP2](https://github.com/google-agentic-commerce/AP2) |
| Policy Engine | **REAL** | Enterprise spending rules |
| AI Agent (Gemma) | Simulated | The LLM making decisions |
| Payment Settlement | Simulated | Actual money transfer |

## Why This Matters

**The Problem**: AI agents need to spend money autonomously (buying API credits, cloud compute, vendor services). But how do you trust they're following corporate policies?

**The Solution**: Don't trust—verify. Every spending decision generates a cryptographic proof that:
- The classifier correctly categorized the purchase intent
- The exact approved ML model was used
- Anyone can verify without re-running inference

## The Trust Model

```
TRUSTED (not proved):
├── On-device Gemma runs correctly
└── Function call output is authentic

PROVED (with zkML):
├── Spending classifier ran correctly
├── Classification output is genuine
└── Approved model was used
```

The zkML proof is like a cryptographic receipt: *"This classifier really did output 'data_services' for this input."*

## Project Structure

```
demos/ai-edge-ap2/
├── backend/           # Rust API + Jolt Atlas zkML prover
├── frontend/          # Interactive demo UI
├── ap2-service/       # Python FastAPI (Google AI Edge + AP2)
└── README.md          # Detailed documentation
```

## Performance

| Metric | Value |
|--------|-------|
| ML Inference | ~1ms |
| Proof Generation | ~5s |
| Proof Verification | ~2s |
| Commitment Scheme | Dory |

## Learn More

- [Demo Documentation](demos/ai-edge-ap2/README.md) - Detailed architecture and API docs
- [Jolt Atlas](https://github.com/ICME-Lab/jolt-atlas) - The zkML framework powering this demo
- [Google AI Edge](https://ai.google.dev/edge) - On-device ML inference
- [Google AP2](https://github.com/google-agentic-commerce/AP2) - Agent Payments Protocol
