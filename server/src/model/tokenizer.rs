use crate::error::{Result, ServerError};
use candle_transformers::models::llama::Tokenizer as LlamaTokenizer;
use std::path::Path;

/// Tokenizer wrapper for different tokenizer types
pub struct Tokenizer {
    inner: TokenizerType,
}

enum TokenizerType {
    Llama(LlamaTokenizer),
}

impl Tokenizer {
    /// Load a tokenizer from the given path
    pub fn load(path: &str) -> Result<Self> {
        let tokenizer_path = Path::new(path);

        if !tokenizer_path.exists() {
            return Err(ServerError::ModelNotFound(format!(
                "Tokenizer path does not exist: {}",
                path
            )));
        }

        // For now, we'll use the Llama tokenizer as the default
        let tokenizer = LlamaTokenizer::from_file(tokenizer_path).map_err(|e| {
            ServerError::ModelLoading(format!("Failed to load tokenizer: {}", e))
        })?;

        Ok(Self {
            inner: TokenizerType::Llama(tokenizer),
        })
    }

    /// Encode text into token IDs
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<u32>> {
        match &self.inner {
            TokenizerType::Llama(tokenizer) => tokenizer
                .encode(text, add_bos)
                .map_err(|e| ServerError::Inference(format!("Tokenization failed: {}", e))),
        }
    }

    /// Decode token IDs back into text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        match &self.inner {
            TokenizerType::Llama(tokenizer) => tokenizer
                .decode(tokens)
                .map_err(|e| ServerError::Inference(format!("Detokenization failed: {}", e))),
        }
    }

    /// Get the vocabulary size
    pub fn vocab_size(&self) -> usize {
        match &self.inner {
            TokenizerType::Llama(tokenizer) => tokenizer.vocab_size(),
        }
    }

    /// Get the BOS (Beginning of Sequence) token ID
    pub fn bos_token_id(&self) -> u32 {
        match &self.inner {
            TokenizerType::Llama(_) => 1, // Standard BOS token for Llama
        }
    }

    /// Get the EOS (End of Sequence) token ID
    pub fn eos_token_id(&self) -> u32 {
        match &self.inner {
            TokenizerType::Llama(_) => 2, // Standard EOS token for Llama
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_load_invalid_path() {
        let result = Tokenizer::load("/nonexistent/tokenizer.json");
        assert!(result.is_err());
    }
}
