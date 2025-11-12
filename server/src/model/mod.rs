pub mod loader;
pub mod tokenizer;

use crate::config::ModelConfig;
use crate::error::{Result, ServerError};
use candle_core::{Device, Tensor};
use candle_transformers::models::llama as llama_model;
use std::sync::Arc;

/// Represents different model architectures
#[derive(Debug, Clone)]
pub enum ModelArchitecture {
    Llama,
    Mistral,
    Phi,
}

impl ModelArchitecture {
    pub fn from_string(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "llama" => Ok(Self::Llama),
            "mistral" => Ok(Self::Mistral),
            "phi" => Ok(Self::Phi),
            _ => Err(ServerError::ModelNotFound(format!(
                "Unsupported architecture: {}",
                s
            ))),
        }
    }
}

/// Model wrapper that holds the loaded model and its configuration
pub struct Model {
    pub architecture: ModelArchitecture,
    pub device: Device,
    pub config: llama_model::Config,
    pub cache: llama_model::Cache,
    inner: Arc<Box<dyn ModelInference + Send + Sync>>,
}

/// Trait for model inference operations
pub trait ModelInference {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor>;
}

/// Llama model implementation
pub struct LlamaModel {
    model: llama_model::Llama,
}

impl ModelInference for LlamaModel {
    fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        self.model
            .forward(input_ids, pos)
            .map_err(|e| ServerError::Inference(e.to_string()))
    }
}

impl Model {
    /// Load a model from the given configuration
    pub fn load(config: &ModelConfig) -> Result<Self> {
        let device = Self::get_device(&config.device)?;
        let architecture = ModelArchitecture::from_string(&config.architecture)?;

        match architecture {
            ModelArchitecture::Llama => Self::load_llama(config, device),
            _ => Err(ServerError::ModelNotFound(format!(
                "Architecture {:?} not yet implemented",
                architecture
            ))),
        }
    }

    fn get_device(device_str: &str) -> Result<Device> {
        match device_str.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    Ok(Device::new_cuda(0).map_err(|e| {
                        ServerError::ModelLoading(format!("CUDA initialization failed: {}", e))
                    })?)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(ServerError::ModelLoading(
                        "CUDA support not compiled".to_string(),
                    ))
                }
            }
            "metal" => {
                #[cfg(feature = "metal")]
                {
                    Ok(Device::new_metal(0).map_err(|e| {
                        ServerError::ModelLoading(format!("Metal initialization failed: {}", e))
                    })?)
                }
                #[cfg(not(feature = "metal"))]
                {
                    Err(ServerError::ModelLoading(
                        "Metal support not compiled".to_string(),
                    ))
                }
            }
            _ => Err(ServerError::Config(format!(
                "Unknown device: {}",
                device_str
            ))),
        }
    }

    fn load_llama(config: &ModelConfig, device: Device) -> Result<Self> {
        tracing::info!("Loading Llama model from: {}", config.path);

        // Load model configuration
        let model_config = llama_model::Config::config_7b_v2();

        // Create KV cache for efficient inference
        let cache = llama_model::Cache::new(true, &model_config, &device)
            .map_err(|e| ServerError::ModelLoading(format!("Failed to create cache: {}", e)))?;

        // Load model weights
        let vb = loader::load_weights(&config.path, &device)?;

        // Build the model
        let llama = llama_model::Llama::load(vb, &model_config)
            .map_err(|e| ServerError::ModelLoading(format!("Failed to load model: {}", e)))?;

        let llama_wrapper = LlamaModel { model: llama };

        Ok(Self {
            architecture: ModelArchitecture::Llama,
            device,
            config: model_config,
            cache,
            inner: Arc::new(Box::new(llama_wrapper)),
        })
    }

    /// Perform inference on the given input tokens
    pub fn forward(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        // Get mutable access to the inner model
        let inner = Arc::get_mut(&mut self.inner)
            .ok_or_else(|| ServerError::Internal("Multiple references to model".to_string()))?;

        inner.forward(input_ids, pos)
    }

    /// Get the embedding dimension
    pub fn embed_dim(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_architecture_parsing() {
        assert!(matches!(
            ModelArchitecture::from_string("llama").unwrap(),
            ModelArchitecture::Llama
        ));
        assert!(matches!(
            ModelArchitecture::from_string("LLAMA").unwrap(),
            ModelArchitecture::Llama
        ));
        assert!(ModelArchitecture::from_string("unknown").is_err());
    }

    #[test]
    fn test_device_parsing() {
        assert!(matches!(
            Model::get_device("cpu").unwrap(),
            Device::Cpu
        ));
    }
}
