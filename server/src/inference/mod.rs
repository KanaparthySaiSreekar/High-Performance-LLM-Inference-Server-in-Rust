pub mod sampler;
pub mod generator;

use crate::config::InferenceConfig;
use crate::error::{Result, ServerError};
use crate::model::{Model, tokenizer::Tokenizer};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Generation parameters for text completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationParams {
    /// Maximum number of tokens to generate
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Sampling temperature (0.0 - 2.0)
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Top-p (nucleus) sampling parameter
    #[serde(default = "default_top_p")]
    pub top_p: f64,

    /// Top-k sampling parameter
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Repetition penalty
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f64,

    /// Stop sequences
    #[serde(default)]
    pub stop: Vec<String>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> usize {
    256
}
fn default_temperature() -> f64 {
    0.7
}
fn default_top_p() -> f64 {
    0.9
}
fn default_top_k() -> usize {
    50
}
fn default_repetition_penalty() -> f64 {
    1.1
}

impl Default for GenerationParams {
    fn default() -> Self {
        Self {
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            top_k: default_top_k(),
            repetition_penalty: default_repetition_penalty(),
            stop: vec![],
            stream: false,
        }
    }
}

impl GenerationParams {
    pub fn with_defaults(self, config: &InferenceConfig) -> Self {
        Self {
            max_tokens: if self.max_tokens == 0 {
                config.default_max_tokens
            } else {
                self.max_tokens
            },
            temperature: if self.temperature == 0.0 {
                config.default_temperature
            } else {
                self.temperature
            },
            top_p: if self.top_p == 0.0 {
                config.default_top_p
            } else {
                self.top_p
            },
            top_k: if self.top_k == 0 {
                config.default_top_k
            } else {
                self.top_k
            },
            ..self
        }
    }
}

/// The inference engine manages model execution and text generation
pub struct InferenceEngine {
    model: Arc<RwLock<Model>>,
    tokenizer: Arc<Tokenizer>,
    config: InferenceConfig,
    device: Device,
}

impl InferenceEngine {
    pub fn new(
        model: Model,
        tokenizer: Tokenizer,
        config: InferenceConfig,
    ) -> Self {
        let device = model.device.clone();
        Self {
            model: Arc::new(RwLock::new(model)),
            tokenizer: Arc::new(tokenizer),
            config,
            device,
        }
    }

    /// Generate text from a prompt
    pub async fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<String> {
        let params = params.with_defaults(&self.config);

        // Tokenize input
        let input_tokens = self.tokenizer.encode(prompt, true)?;

        if input_tokens.len() > self.config.max_seq_len {
            return Err(ServerError::InvalidRequest(format!(
                "Input too long: {} tokens (max: {})",
                input_tokens.len(),
                self.config.max_seq_len
            )));
        }

        // Generate tokens
        let output_tokens = self.generate_tokens(&input_tokens, &params).await?;

        // Decode output
        let output_text = self.tokenizer.decode(&output_tokens)?;

        Ok(output_text)
    }

    async fn generate_tokens(
        &self,
        input_tokens: &[u32],
        params: &GenerationParams,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_tokens.to_vec();
        let mut sampler = sampler::Sampler::new(
            params.temperature,
            params.top_p,
            params.top_k,
            params.repetition_penalty,
        );

        let eos_token = self.tokenizer.eos_token_id();
        let max_len = input_tokens.len() + params.max_tokens;

        for pos in 0..params.max_tokens {
            if tokens.len() >= max_len {
                break;
            }

            // Get the last token
            let input_pos = input_tokens.len() + pos;
            let current_token = tokens[tokens.len() - 1];

            // Create input tensor
            let input_tensor = Tensor::new(&[current_token], &self.device)
                .map_err(|e| ServerError::Inference(e.to_string()))?;

            // Run model forward pass
            let logits = {
                let mut model = self.model.write().await;
                model.forward(&input_tensor, input_pos)?
            };

            // Sample next token
            let next_token = sampler.sample(&logits, &tokens)?;

            // Check for EOS or stop sequences
            if next_token == eos_token {
                break;
            }

            tokens.push(next_token);

            // Check stop sequences
            if !params.stop.is_empty() {
                let current_text = self.tokenizer.decode(&tokens)?;
                if params.stop.iter().any(|stop| current_text.contains(stop)) {
                    break;
                }
            }
        }

        // Return only the generated tokens (exclude input)
        Ok(tokens[input_tokens.len()..].to_vec())
    }

    pub fn tokenizer(&self) -> Arc<Tokenizer> {
        Arc::clone(&self.tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation_params_defaults() {
        let params = GenerationParams::default();
        assert_eq!(params.max_tokens, 256);
        assert_eq!(params.temperature, 0.7);
        assert_eq!(params.top_p, 0.9);
        assert_eq!(params.top_k, 50);
    }

    #[test]
    fn test_params_with_defaults() {
        let config = InferenceConfig::default();
        let params = GenerationParams {
            max_tokens: 0,
            temperature: 0.0,
            ..Default::default()
        };
        let merged = params.with_defaults(&config);
        assert_eq!(merged.max_tokens, config.default_max_tokens);
        assert_eq!(merged.temperature, config.default_temperature);
    }
}
