use crate::error::{Result, ServerError};
use candle_core::Tensor;
use rand::distributions::Distribution;
use rand::SeedableRng;

/// Sampler for generating next tokens from model logits
pub struct Sampler {
    temperature: f64,
    top_p: f64,
    top_k: usize,
    repetition_penalty: f64,
    rng: rand::rngs::StdRng,
}

impl Sampler {
    pub fn new(temperature: f64, top_p: f64, top_k: usize, repetition_penalty: f64) -> Self {
        Self {
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            rng: rand::rngs::StdRng::from_entropy(),
        }
    }

    /// Sample the next token from logits
    pub fn sample(&mut self, logits: &Tensor, context: &[u32]) -> Result<u32> {
        let logits = logits
            .to_vec1::<f32>()
            .map_err(|e| ServerError::Inference(e.to_string()))?;

        // Apply repetition penalty
        let mut logits = self.apply_repetition_penalty(logits, context);

        // Apply temperature
        if self.temperature > 0.0 {
            logits = logits
                .iter()
                .map(|&l| l / self.temperature as f32)
                .collect();
        }

        // Convert to probabilities
        let probs = self.softmax(&logits);

        // Apply top-k filtering
        let probs = if self.top_k > 0 && self.top_k < probs.len() {
            self.top_k_filtering(probs)
        } else {
            probs
        };

        // Apply top-p (nucleus) filtering
        let probs = if self.top_p < 1.0 {
            self.top_p_filtering(probs)
        } else {
            probs
        };

        // Sample from the distribution
        self.sample_from_probs(&probs)
    }

    fn apply_repetition_penalty(&self, mut logits: Vec<f32>, context: &[u32]) -> Vec<f32> {
        if self.repetition_penalty == 1.0 {
            return logits;
        }

        for &token in context {
            let token_idx = token as usize;
            if token_idx < logits.len() {
                if logits[token_idx] < 0.0 {
                    logits[token_idx] *= self.repetition_penalty as f32;
                } else {
                    logits[token_idx] /= self.repetition_penalty as f32;
                }
            }
        }

        logits
    }

    fn softmax(&self, logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    fn top_k_filtering(&self, probs: Vec<f32>) -> Vec<f32> {
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort by probability (descending)
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Zero out probabilities beyond top-k
        let mut result = vec![0.0; probs.len()];
        for (idx, prob) in indexed_probs.iter().take(self.top_k) {
            result[*idx] = *prob;
        }

        // Renormalize
        let sum: f32 = result.iter().sum();
        if sum > 0.0 {
            result.iter().map(|&p| p / sum).collect()
        } else {
            result
        }
    }

    fn top_p_filtering(&self, probs: Vec<f32>) -> Vec<f32> {
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();

        // Sort by probability (descending)
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find the cutoff point where cumulative probability exceeds top_p
        let mut cumulative = 0.0;
        let mut cutoff_idx = indexed_probs.len();

        for (i, (_, prob)) in indexed_probs.iter().enumerate() {
            cumulative += prob;
            if cumulative >= self.top_p as f32 {
                cutoff_idx = i + 1;
                break;
            }
        }

        // Zero out probabilities beyond cutoff
        let mut result = vec![0.0; probs.len()];
        for (idx, prob) in indexed_probs.iter().take(cutoff_idx) {
            result[*idx] = *prob;
        }

        // Renormalize
        let sum: f32 = result.iter().sum();
        if sum > 0.0 {
            result.iter().map(|&p| p / sum).collect()
        } else {
            result
        }
    }

    fn sample_from_probs(&mut self, probs: &[f32]) -> Result<u32> {
        if probs.is_empty() {
            return Err(ServerError::Inference(
                "Empty probability distribution".to_string(),
            ));
        }

        // Convert to f64 for better precision
        let probs_f64: Vec<f64> = probs.iter().map(|&p| p as f64).collect();

        // Create a weighted distribution
        let dist = rand::distributions::WeightedIndex::new(&probs_f64).map_err(|e| {
            ServerError::Inference(format!("Failed to create weighted distribution: {}", e))
        })?;

        let sampled_idx = dist.sample(&mut self.rng);
        Ok(sampled_idx as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let sampler = Sampler::new(1.0, 1.0, 0, 1.0);
        let logits = vec![1.0, 2.0, 3.0];
        let probs = sampler.softmax(&logits);

        // Check that probabilities sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that probabilities are in ascending order
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_top_k_filtering() {
        let mut sampler = Sampler::new(1.0, 1.0, 2, 1.0);
        let probs = vec![0.1, 0.3, 0.5, 0.05, 0.05];
        let filtered = sampler.top_k_filtering(probs);

        // Only top 2 should be non-zero
        let non_zero_count = filtered.iter().filter(|&&p| p > 0.0).count();
        assert_eq!(non_zero_count, 2);

        // Check sum is still 1
        let sum: f32 = filtered.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_repetition_penalty() {
        let sampler = Sampler::new(1.0, 1.0, 0, 1.5);
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let context = vec![1, 2]; // Tokens 1 and 2 were used
        let penalized = sampler.apply_repetition_penalty(logits.clone(), &context);

        // Penalized logits should be different for used tokens
        assert_ne!(penalized[1], logits[1]);
        assert_ne!(penalized[2], logits[2]);
        // Unused tokens should be unchanged
        assert_eq!(penalized[0], logits[0]);
        assert_eq!(penalized[3], logits[3]);
    }
}
