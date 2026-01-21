//! AI Edge → AP2 Payment Demo Backend
//!
//! API server that provides:
//! - zkML classification and proof generation
//! - Policy compliance checking
//! - Mock AP2 payment processing
//! - Transaction audit logging

mod api;
mod zkml;
mod ap2;

use axum::{
    routing::{get, post},
    Router,
    http::Method,
};
use tower_http::cors::{CorsLayer, Any};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::api::AppState;

#[tokio::main]
async fn main() {
    // Initialize logging
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .init();

    tracing::info!("Starting AI Edge → AP2 Demo Backend");

    // Initialize shared state
    let state = Arc::new(RwLock::new(AppState::new()));

    // Initialize zkML (preprocessing) in background
    let state_clone = state.clone();
    tokio::spawn(async move {
        tracing::info!("Initializing zkML preprocessing (this may take a moment)...");
        let mut state = state_clone.write().await;
        state.initialize_zkml().await;
        tracing::info!("zkML preprocessing complete!");
    });

    // CORS configuration for local development
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        // Health check
        .route("/health", get(api::health_check))

        // zkML endpoints
        .route("/api/v1/classify", post(api::classify_intent))
        .route("/api/v1/verify", post(api::verify_proof))

        // Policy endpoints
        .route("/api/v1/policy/:policy_id", get(api::get_policy))
        .route("/api/v1/policies", get(api::list_policies))

        // Transaction endpoints
        .route("/api/v1/transactions", get(api::list_transactions))
        .route("/api/v1/transactions/:tx_id", get(api::get_transaction))

        // AP2 mock endpoints
        .route("/api/v1/ap2/pay", post(api::process_ap2_payment))

        // Demo control endpoints
        .route("/api/v1/demo/trigger/:scenario", post(api::trigger_scenario))
        .route("/api/v1/demo/reset", post(api::reset_demo))

        .layer(cors)
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3001").await.unwrap();
    tracing::info!("Backend listening on http://localhost:3001");

    axum::serve(listener, app).await.unwrap();
}
