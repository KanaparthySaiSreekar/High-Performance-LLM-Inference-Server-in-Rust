mod api;
mod batch;
mod config;
mod error;
mod inference;
mod metrics;
mod model;

use crate::api::AppState;
use crate::batch::BatchProcessor;
use crate::config::{CliArgs, ServerConfig};
use crate::inference::InferenceEngine;
use crate::model::{tokenizer::Tokenizer, Model};
use clap::Parser;
use std::sync::Arc;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse CLI arguments
    let args = CliArgs::parse();

    // Initialize logging
    init_logging(&args.log_level)?;

    tracing::info!("Starting LLM Inference Server v{}", env!("CARGO_PKG_VERSION"));

    // Load configuration
    let config = load_config(args)?;

    tracing::info!("Configuration loaded: {:?}", config);

    // Register metrics
    metrics::register_metrics();

    // Initialize the server
    let state = initialize_server(config.clone()).await?;

    // Build the router
    let app = api::build_router(state);

    // Start the server
    let addr = format!("{}:{}", config.host, config.port);
    tracing::info!("Server listening on http://{}", addr);
    tracing::info!("Available endpoints:");
    tracing::info!("  - POST   http://{}/v1/completions", addr);
    tracing::info!("  - POST   http://{}/v1/chat/completions", addr);
    tracing::info!("  - GET    http://{}/v1/models", addr);
    tracing::info!("  - GET    http://{}/health", addr);
    tracing::info!("  - GET    http://{}/metrics", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn init_logging(level: &str) -> anyhow::Result<()> {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(level));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();

    Ok(())
}

fn load_config(args: CliArgs) -> anyhow::Result<ServerConfig> {
    if let Some(config_path) = &args.config {
        tracing::info!("Loading configuration from: {:?}", config_path);
        let config = ServerConfig::from_file(config_path)?;
        Ok(config.merge_with_cli(args))
    } else {
        Ok(ServerConfig::from(args))
    }
}

async fn initialize_server(config: ServerConfig) -> anyhow::Result<AppState> {
    tracing::info!("Initializing inference engine...");

    // Check if model path exists or needs to be downloaded
    let model_path = if config.model.path.starts_with("hf://") {
        // Download from Hugging Face
        let repo_id = config.model.path.trim_start_matches("hf://");
        tracing::info!("Downloading model from Hugging Face: {}", repo_id);

        // For now, we'll just use the path as-is
        // In a real implementation, this would download the model
        tracing::warn!("HuggingFace download not fully implemented, using path as-is");
        config.model.path.clone()
    } else {
        config.model.path.clone()
    };

    // Load model (this will fail in the demo since we don't have actual model files)
    // In production, you would download or point to a real model
    tracing::info!("Loading model from: {}", model_path);

    // For demo purposes, we'll create a placeholder that shows the architecture
    // In production, uncomment the following line:
    // let model = Model::load(&config.model)?;

    // DEMO MODE: Skip actual model loading
    tracing::warn!("===========================================");
    tracing::warn!("DEMO MODE: Skipping actual model loading");
    tracing::warn!("To use this server in production:");
    tracing::warn!("1. Download a model (e.g., from HuggingFace)");
    tracing::warn!("2. Point --model-path to the model directory");
    tracing::warn!("3. Uncomment model loading in main.rs");
    tracing::warn!("===========================================");

    // Create a mock inference engine for demo
    // In production, replace this with actual model/tokenizer loading
    let batch_processor = create_demo_batch_processor(&config);

    Ok(AppState {
        batch_processor: Arc::new(batch_processor),
        config: Arc::new(config),
    })
}

// Demo batch processor (replace with real one in production)
fn create_demo_batch_processor(config: &ServerConfig) -> BatchProcessor {
    use crate::error::ServerError;

    tracing::info!("Creating demo batch processor (no actual model)");

    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel(config.performance.queue_capacity);
    let capacity_semaphore = Arc::new(tokio::sync::Semaphore::new(config.performance.queue_capacity));

    // Spawn demo workers
    for worker_id in 0..config.performance.workers {
        let mut rx = request_rx.clone();
        let semaphore = Arc::clone(&capacity_semaphore);

        tokio::spawn(async move {
            tracing::info!("Demo worker {} started", worker_id);

            while let Some(request) = rx.recv().await {
                tracing::debug!("Demo worker {} processing request {}", worker_id, request.id);

                // Generate a demo response
                let response = format!(
                    "[DEMO MODE] Echo response for prompt: '{}' (max_tokens: {})",
                    request.prompt, request.params.max_tokens
                );

                // Simulate some processing time
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

                let _ = request.response_tx.send(Ok(response));
                semaphore.add_permits(1);
            }
        });
    }

    drop(request_rx);

    BatchProcessor {
        request_tx,
        capacity_semaphore,
    }
}

// Production initialization (commented out for demo)
/*
async fn initialize_production_server(config: ServerConfig) -> anyhow::Result<AppState> {
    tracing::info!("Loading model...");
    let model = Model::load(&config.model)?;

    tracing::info!("Loading tokenizer...");
    let tokenizer_path = config.model.tokenizer_path.as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer path required"))?;
    let tokenizer = Tokenizer::load(tokenizer_path)?;

    tracing::info!("Creating inference engine...");
    let engine = Arc::new(InferenceEngine::new(
        model,
        tokenizer,
        config.inference.clone(),
    ));

    tracing::info!("Creating batch processor...");
    let batch_processor = BatchProcessor::new(
        engine,
        config.performance.queue_capacity,
        config.performance.max_batch_size,
    );

    Ok(AppState {
        batch_processor: Arc::new(batch_processor),
        config: Arc::new(config),
    })
}
*/
