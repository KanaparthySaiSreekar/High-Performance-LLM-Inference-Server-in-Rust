pub mod handlers;
pub mod models;

use crate::batch::BatchProcessor;
use crate::config::ServerConfig;
use crate::metrics;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use std::sync::Arc;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    trace::TraceLayer,
};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub batch_processor: Arc<BatchProcessor>,
    pub config: Arc<ServerConfig>,
}

/// Build the HTTP router with all endpoints
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Health and status endpoints
        .route("/health", get(health_check))
        .route("/", get(root_handler))
        // OpenAI-compatible endpoints
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/models", get(handlers::list_models))
        // Metrics endpoint
        .route("/metrics", get(metrics_handler))
        // Add middleware
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

/// Root handler - returns server information
async fn root_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "name": "LLM Inference Server",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "running",
        "model": state.config.model.architecture,
        "endpoints": {
            "completions": "/v1/completions",
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "metrics": "/metrics"
        }
    }))
}

/// Health check endpoint
async fn health_check() -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "timestamp": chrono::Utc::now().to_rfc3339()
        })),
    )
}

/// Metrics endpoint - returns Prometheus metrics
async fn metrics_handler() -> String {
    metrics::gather_metrics()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        // This test just ensures the router can be created without panicking
        // Actual testing would require a full server setup
    }
}
